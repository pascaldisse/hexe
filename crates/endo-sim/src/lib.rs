//! endo-sim (L1): passive-ragdoll simulation of an Endorphin `Scene` on `rapier3d` (f32).
//!
//! Build: every `<MetaBody>` of a character becomes one dynamic rigid body whose frame is the owning joint's world frame
//! (collision surfaces `body.xf * surface.xf`, shapes Sphyl/Cylinder along local Z like `endorphin_geo`; `File`/`Mesh`/unknown
//! -> capsule fitted to the placeholder bbox). Mass = density x inertial volume (unknown = 1000 kg/m3 x collider volume).
//! Each joint-with-body is tied to its nearest ancestor-with-body: spherical (hard limits, else soft) when limit data exists,
//! fixed otherwise. Frames are centred on the POSED start configuration (scene pose at t = 0), twist about joint X (measured: knee/elbow
//! hinges carry twist 2.8 rad + swing 0.02), swing1/swing2 about joint Y/Z; limits are symmetric half-ranges. Joints that own no body (virtual muscles, nulls)
//! ride their nearest ancestor body at their start-pose offset. The `<Environment>` bodies are static. Ground: scene default
//! ground = static slab with top at y = 0. Gravity from the scene. Same-character colliders never collide.
//!
//! `Sim::from_scene` = PASSIVE ragdoll (no behaviours, no force events). `Sim::from_scene_active` adds the behaviour layer
//! (`behaviour.rs`: `Bob2: CatchFall` joint drives from the scene parameters) and the scene's `<ForceEvent>` pushes.
//! OUT OF SCOPE: every other behaviour (the `.nmb` networks are not decoded, NMB.md): scene instances of them are ignored.
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unimplemented,
    clippy::todo,
    clippy::unreachable
)]
mod behave;
pub mod behaviour;
pub use behave::BehaviourState;
use behaviour::{CatchFall, ForceRt, Gains};
use hexe_geo::endorphin::{Character, Geometry, Joint, Limits, Scene, Xf};
use rapier3d::glamx::{Mat3, Quat, Vec3};
use rapier3d::prelude::*;
use std::collections::{HashMap, HashSet};

/// Largest internal substep (s) of a passive / CatchFall-only sim: `step(dt)` splits `dt` so no substep exceeds it.
const MAX_SUBSTEP: f32 = 1.0 / 120.0;
/// Largest internal substep (s) with the `secondary` layer on: the joint motors need the shorter step to stay stable.
const MAX_SUBSTEP_PLAY: f32 = 1.0 / 240.0;
const MAX_SUBSTEPS: usize = 128;
/// Solver iterations with the `secondary` layer on (rapier's default of 1 lets the motor chains sag).
/// 8: at 4 the soft contacts let feet / hands sink 2-6 cm into floor and balcony slabs (clearance h48).
const PLAY_SOLVER_ITERS: usize = 8;
/// Contact stiffness (Hz) of the PLAY build; static contacts get twice that (rapier's ratio). Rapier's 30 Hz default
/// leaves a ~1.5 mm mean sink and lets a fast limb sit half through a thin slab, where the normal can flip.
const PLAY_CONTACT_HZ: f32 = 60.0;
/// Error-reduction parameter that turns a `<MaterialInteraction softness>` (read as an ODE CFM) into a contact spring.
const SOFT_ERP: f32 = 0.2;
/// Density for bodies whose mass the file does not give (water-ish, like a body).
const FALLBACK_DENSITY: f32 = 1000.0;
const MIN_MASS: f32 = 0.05;
/// Placeholder half extents of unknown (`File`/`Mesh`) geometry: the 5 cm box `endorphin_geo` draws.
const PLACEHOLDER_HALF: [f32; 3] = [0.025; 3];
/// `.ens` densities are g/cm^3 (Head 4.55 ... limbs 1.5): kg/m^3 = x 1000. Only the `secondary` build uses them scaled;
/// the h44/h46 builds keep the unscaled (clamped, near-uniform) masses so their measured numbers stand.
const DENSITY_SCALE: f32 = 1000.0;
/// Mass compression of the `secondary` build: `m -> MASS_REF (m / MASS_REF)^MASS_EXP` (heavy torso vs light hands servo alike).
const MASS_REF: f32 = 5.0;
const MASS_EXP: f32 = 0.5;
/// Collision group of everything that is not a character (environment, ground).
const WORLD_GROUP: u32 = 1 << 31;

/// World matrices of every joint of one character, in `Character::joints` order (placement included).
#[derive(Clone, Debug, PartialEq)]
pub struct CharPose {
    pub name: String,
    pub world: Vec<Xf>,
}

/// How one joint gets its world matrix.
enum Drive {
    /// Owns a rigid body: world = body pose.
    Body(RigidBodyHandle),
    /// Rides joint `anchor` (index in the same character): world = world[anchor] * offset.
    Follow { anchor: usize, offset: Xf },
    /// Names a body another joint owns (virtual muscle -> hand / heel mass): world = that body's pose, so the shared geometry
    /// is drawn where its collider is (riding the own ancestor chain carried it rigidly through the floor, no collider there).
    /// The controllers (`joint_world`) keep the rigid `Follow` of `anchor` / `offset` (e.g. the head axis to `UpperNeckEndJoint`).
    Shared { body: RigidBodyHandle, anchor: usize, offset: Xf },
    /// No body anywhere above: stays at its start-pose world matrix.
    Fixed(Xf),
}

struct Rig {
    name: String,
    /// joint names in `Character::joints` order.
    names: Vec<String>,
    /// swing gains per joint name from the `behaviour::CALIBRATION` set.
    gains: HashMap<String, Gains>,
    drives: Vec<Drive>,
    /// Per joint: the rigid body it owns.
    handles: Vec<Option<RigidBodyHandle>>,
    /// Per joint: its spherical (limited) impulse joint to the anchor body.
    jhandles: Vec<Option<ImpulseJointHandle>>,
    /// Per joint: nearest ancestor-or-self joint that owns a body.
    anchor: Vec<Option<usize>>,
    /// Per joint: start-pose world matrix (placement included).
    rest: Vec<Xf>,
    /// Simulated character (false = static environment).
    dynamic: bool,
}

/// What `Sim::from_scene_with` adds on top of the passive ragdoll.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimOptions {
    /// Build the joint-drive controllers of the implemented behaviours (`behaviour.rs`).
    pub behaviours: bool,
    /// Apply the scene's `<ForceEvent>` pushes.
    pub forces: bool,
    /// Build the pose-hold / balance / secondary-behaviour layer (`behave.rs`: Stagger, Arms Windmill, Legs Straighten, brace).
    /// Also switches the build to scaled masses, rigid minor joints, flat soles and the shorter substep.
    pub secondary: bool,
}
/// A running simulation of one scene.
pub struct Sim {
    world: PhysicsWorld,
    rigs: Vec<Rig>,
    time: f32,
    catches: Vec<CatchFall>,
    forces: Vec<ForceRt>,
    behave: Option<behave::Behave>,
    max_substep: f32,
    /// Self-collision filter (`SelfPairs`); `enabled = false` keeps the pre-h48 no-self-contact build.
    self_pairs: SelfPairs,
    /// PLAY build: the scene's `Lock` / `JoinBodies` constraint events.
    holds: Vec<HoldRt>,
    /// h54 (default true): a `<ForceEvent>` on a body welded by a `JoinBodies` hold accelerates the whole group
    /// (force = accel x group mass). false = pre-h54 (accel x the one body's mass).
    pub force_welded: bool,
}

/// `<TertiaryBehaviour lockType="Lock|JoinBodies">` (keepOffset): over frames [start, end) the listed bodies are held by
/// fixed joints at their pose of frame `start` — Lock: each to the world, JoinBodies: each to the first listed body.
/// h56 (endorphin.exe, proof/h56): apply 0x4c4e90 -> per-body RPRO magnet setters 0x622060 / 0x6220b0 -> Karma
/// `MdtRPROJoint` linear / angular strength (0x7c3e80 / 0x7c3e50) -> RPRO rows (0x7cde8b): lo = -s, hi = +s, s == 0 ->
/// row zeroed. So per axis (joint frame = body frame at `start`): strength = the rigid row's force bound (N, N m);
/// 0 = free. Rapier: a stiff position motor (rigid at substep resolution) capped by `motor_max_force(s)`.
pub struct HoldRt {
    pub name: String,
    pub lock: bool,
    pub start: i32,
    pub end: i32,
    /// (`PositionStrength` xyz, `AngularStrength` xyz).
    pub strength: ([f32; 3], [f32; 3]),
    /// `character.MetaBodies.Body` -> resolved rigid body (None: unknown / not built).
    pub objects: Vec<(String, Option<RigidBodyHandle>)>,
    joints: Vec<ImpulseJointHandle>,
    /// 0 = waiting, 1 = holding, 2 = released.
    state: u8,
    /// Substep (s) the rows are solved at (set by `Sim::step`): motor gains rigid at this resolution.
    dt: f32,
}

impl HoldRt {
    /// Holding and joining `h` into a group (`JoinBodies` only: a `Lock` ties to the world).
    fn welds(&self, h: RigidBodyHandle) -> bool {
        self.state == 1 && !self.lock && self.objects.iter().any(|o| o.1 == Some(h))
    }
    fn engage(&mut self, w: &mut PhysicsWorld) {
        let bodies: Vec<RigidBodyHandle> = self.objects.iter().filter_map(|o| o.1).collect();
        let dynamic = |w: &PhysicsWorld, h: RigidBodyHandle| w.bodies.get(h).is_some_and(|b| b.is_dynamic());
        let (lin, ang) = self.strength;
        let axes = [
            (JointAxis::LinX, lin[0]),
            (JointAxis::LinY, lin[1]),
            (JointAxis::LinZ, lin[2]),
            (JointAxis::AngX, ang[0]),
            (JointAxis::AngY, ang[1]),
            (JointAxis::AngZ, ang[2]),
        ];
        // Karma RPRO row: rigid (error corrected at substep resolution: k = 1/h^2, c = 2/h, mass-independent), force
        // bounded |f| <= s; s == 0 -> no row (free axis)
        let (k, c) = (1.0 / (self.dt * self.dt), 2.0 / self.dt);
        let joint = |f1: Pose| {
            let mut j = GenericJointBuilder::new(JointAxesMask::empty())
                .local_frame1(f1)
                .local_frame2(Pose::IDENTITY)
                .contacts_enabled(false);
            for &(ax, s) in axes.iter().filter(|a| a.1 > 0.0) {
                j = j
                    .motor_model(ax, MotorModel::AccelerationBased)
                    .motor_position(ax, 0.0, k, c)
                    .motor_max_force(ax, s);
            }
            j.build()
        };
        if self.lock {
            let anchor = w.insert_body(RigidBodyBuilder::fixed());
            let held: Vec<RigidBodyHandle> = bodies.into_iter().filter(|b| dynamic(w, *b)).collect();
            for b in held {
                let Some(p) = w.bodies.get(b).map(|x| *x.position()) else { continue };
                self.joints.push(w.insert_impulse_joint(anchor, b, joint(p)));
            }
        } else if let Some((&a, rest)) = bodies.split_first() {
            for &b in rest {
                if !(dynamic(w, a) || dynamic(w, b)) {
                    continue;
                }
                let (Some(pa), Some(pb)) = (w.bodies.get(a).map(|x| *x.position()), w.bodies.get(b).map(|x| *x.position()))
                else {
                    continue;
                };
                let rel = pa.inverse() * pb;
                self.joints.push(w.insert_impulse_joint(a, b, joint(rel)));
            }
        }
        self.state = 1;
    }
    fn update(&mut self, w: &mut PhysicsWorld, frame: i32) {
        if self.state == 0 && frame >= self.start && frame < self.end {
            self.engage(w);
        } else if self.state == 1 && frame >= self.end {
            for j in self.joints.drain(..) {
                w.impulse_joints.remove(j, true);
            }
            self.state = 2;
        }
    }
    /// Joints built when engaged (0 before).
    pub fn joint_count(&self) -> usize {
        self.joints.len()
    }
}

/// Same-character collider pairs Endorphin collides (`CollisionSurface` `collidesWithSiblings` / `collidesWithAncestors`;
/// cousins always), minus parent/child (joint, `contacts_enabled(false)`) and pairs already overlapping at the start pose.
/// Character colliders carry `user_data = rig index + 1`; any pair of one rig not listed is dropped by the hook.
#[derive(Default)]
struct SelfPairs {
    allowed: HashSet<(ColliderHandle, ColliderHandle)>,
    /// Start-pose overlaps of a dynamic prop collider with another dynamic body's collider (filtered out).
    excluded: HashSet<(ColliderHandle, ColliderHandle)>,
    enabled: bool,
    /// PLAY build: per-surface-pair `<MaterialPair>` interactions (`modify_solver_contacts`).
    mats: Materials,
}

/// One `<MaterialInteraction>`: friction (`.ens` percent / 100), restitution, softness (m of allowed penetration),
/// maxAdhesiveForce (N pulling the pair together while touching).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Interaction {
    friction: f32,
    restitution: f32,
    softness: f32,
    adhesion: f32,
}

/// Scene material table: collider -> `CollisionSurface material=` index; `pair[a * n + b]` = the `<MaterialPair>`'s
/// interaction (symmetric), falling back to the scene's `Default` interaction.
#[derive(Default)]
struct Materials {
    names: Vec<String>,
    of: HashMap<ColliderHandle, usize>,
    pair: Vec<Option<Interaction>>,
    fallback: Option<Interaction>,
}

impl Materials {
    fn parse(root: &hexe_geo::endorphin::Node) -> Materials {
        fn walk<'a>(n: &'a hexe_geo::endorphin::Node, out: &mut Vec<&'a hexe_geo::endorphin::Node>) {
            if matches!(n.tag.as_str(), "MaterialInteraction" | "MaterialPair" | "Material") {
                out.push(n);
            }
            n.children.iter().for_each(|c| walk(c, out));
        }
        let mut nodes = Vec::new();
        walk(root, &mut nodes);
        let inter = |n: &hexe_geo::endorphin::Node| Interaction {
            friction: n.f32("friction").unwrap_or(100.0) / 100.0,
            restitution: n.f32("restitution").unwrap_or(0.0),
            softness: n.f32("softness").unwrap_or(0.0).max(0.0),
            adhesion: n.f32("maxAdhesiveForce").unwrap_or(0.0).max(0.0),
        };
        let named: HashMap<&str, Interaction> = nodes
            .iter()
            .filter(|n| n.tag == "MaterialInteraction")
            .filter_map(|n| Some((n.attr("name")?, inter(n))))
            .collect();
        let mut m = Materials { fallback: named.get("Default").copied(), ..Default::default() };
        for n in nodes.iter().filter(|n| n.tag == "Material") {
            if let Some(s) = n.attr("name") {
                m.index(s);
            }
        }
        let pairs: Vec<(usize, usize, Interaction)> = nodes
            .iter()
            .filter(|n| n.tag == "MaterialPair")
            .filter_map(|n| {
                let i = *named.get(n.attr("interaction")?)?;
                Some((m.index(n.attr("materialA")?), m.index(n.attr("materialB")?), i))
            })
            .collect();
        let k = m.names.len();
        m.pair = vec![None; k * k];
        for (a, b, i) in pairs {
            m.pair[a * k + b] = Some(i);
            m.pair[b * k + a] = Some(i);
        }
        m
    }
    fn index(&mut self, name: &str) -> usize {
        let name = if name.is_empty() { "default" } else { name };
        self.names.iter().position(|n| n == name).unwrap_or_else(|| {
            self.names.push(name.to_string());
            self.names.len() - 1
        })
    }
    fn between(&self, a: ColliderHandle, b: ColliderHandle) -> Option<Interaction> {
        let k = self.names.len();
        match (self.of.get(&a), self.of.get(&b)) {
            (Some(&x), Some(&y)) if x < k && y < k && self.pair.len() == k * k => self.pair[x * k + y].or(self.fallback),
            _ => self.fallback,
        }
    }
}

fn ordered(a: ColliderHandle, b: ColliderHandle) -> (ColliderHandle, ColliderHandle) {
    if a.into_raw_parts() <= b.into_raw_parts() { (a, b) } else { (b, a) }
}

impl PhysicsHooks for SelfPairs {
    fn filter_contact_pair(&self, cx: &PairFilterContext) -> Option<SolverFlags> {
        let (Some(a), Some(b)) = (cx.colliders.get(cx.collider1), cx.colliders.get(cx.collider2)) else {
            return Some(SolverFlags::COMPUTE_IMPULSES);
        };
        // dynamic environment props: pairs overlapping at the start pose (chair pieces, sword in the hand) never collide
        if self.excluded.contains(&ordered(cx.collider1, cx.collider2)) {
            return None;
        }
        if a.user_data == 0 || a.user_data != b.user_data {
            return Some(SolverFlags::COMPUTE_IMPULSES);
        }
        (self.enabled && self.allowed.contains(&ordered(cx.collider1, cx.collider2)))
            .then_some(SolverFlags::COMPUTE_IMPULSES)
    }
    fn modify_solver_contacts(&self, cx: &mut ContactModificationContext) {
        let Some(i) = self.mats.between(cx.collider1, cx.collider2) else { return };
        *cx.friction = i.friction;
        *cx.restitution = i.restitution;
        if i.softness > 0.0 {
            // soft surface = ODE-style constraint force mixing: spring k = ERP / (h * CFM), so a contact carrying the
            // normal force F = impulse / h rests F / k = impulse * CFM / ERP deep (h cancels)
            for c in cx.solver_contacts.iter_mut() {
                let at = (c.contact_id[0] as usize) & 0x7fff_ffff;
                let imp = cx.manifold.points.get(at).map_or(0.0, |p| p.data.impulse.max(0.0));
                c.dist += imp * i.softness / SOFT_ERP;
            }
        }
    }
}

impl Rig {
    fn index_of(&self, joint: &str) -> Option<usize> {
        self.names.iter().position(|n| n == joint)
    }
}

pub(crate) fn v3(a: [f32; 3]) -> Vec3 {
    Vec3::new(a[0], a[1], a[2])
}

fn mat3(x: &Xf) -> Mat3 {
    Mat3::from_cols(v3(x.i), v3(x.j), v3(x.k))
}

fn xf_from(m: &Mat3, p: Vec3) -> Xf {
    Xf { i: m.x_axis.to_array(), j: m.y_axis.to_array(), k: m.z_axis.to_array(), p: p.to_array() }
}

fn finite(x: &Xf) -> bool {
    [x.i, x.j, x.k, x.p].iter().flatten().all(|v| v.is_finite())
}

/// Rotation of the basis (orthonormalised through the quaternion).
pub(crate) fn quat_of(x: &Xf) -> Quat {
    let q = Quat::from_mat3(&mat3(x));
    if q.is_finite() && q.length_squared() > 1e-12 { q.normalize() } else { Quat::IDENTITY }
}

fn pose_of(x: &Xf) -> Pose {
    Pose::from_parts(v3(x.p), quat_of(x))
}

fn xf_of(p: &Pose) -> Xf {
    xf_from(&Mat3::from_quat(p.rotation), p.translation)
}

/// General affine inverse (identity when singular, so hostile matrices never poison the pose).
pub(crate) fn inverse(x: &Xf) -> Xf {
    let m = mat3(x);
    let mi = m.inverse();
    let xi = xf_from(&mi, -(mi * v3(x.p)));
    if finite(&xi) { xi } else { Xf::default() }
}

/// Shape of one collision geometry: builder (in the geometry's own frame), extra local rotation, volume (m^3).
struct Shaped {
    builder: ColliderBuilder,
    rot: Quat,
    volume: f32,
}

/// Radius/half-height clamp: rapier shapes must be strictly positive.
fn pos(v: f32) -> f32 {
    if v.is_finite() { v.abs().max(1e-3) } else { 1e-3 }
}

/// Capsule along Z fitted to a bbox given by half extents `[x, y, z]`: radius = the smaller of x/y, long axis = z.
fn bbox_capsule(half: [f32; 3]) -> Shaped {
    let r = pos(half[0].abs().min(half[1].abs()));
    let h = (pos(half[2]) - r).max(0.0);
    let pi = std::f32::consts::PI;
    Shaped {
        builder: ColliderBuilder::capsule_z(h, r),
        rot: Quat::IDENTITY,
        volume: pi * r * r * 2.0 * h + 4.0 / 3.0 * pi * r * r * r,
    }
}

fn shape_of(g: &Geometry) -> Shaped {
    let [a, b, c, _] = g.params.map(pos);
    let pi = std::f32::consts::PI;
    match g.kind.as_str() {
        "Sphere" => {
            Shaped { builder: ColliderBuilder::ball(a), rot: Quat::IDENTITY, volume: 4.0 / 3.0 * pi * a * a * a }
        }
        "Sphyl" => Shaped {
            builder: ColliderBuilder::capsule_z(b * 0.5, a),
            rot: Quat::IDENTITY,
            volume: pi * a * a * b + 4.0 / 3.0 * pi * a * a * a,
        },
        // rapier cylinders run along Y: turn Y onto Z
        "Cylinder" => Shaped {
            builder: ColliderBuilder::cylinder(b * 0.5, a),
            rot: Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
            volume: pi * a * a * b,
        },
        "Cuboid" => Shaped {
            builder: ColliderBuilder::cuboid(a * 0.5, b * 0.5, c * 0.5),
            rot: Quat::IDENTITY,
            volume: a * b * c,
        },
        _ => bbox_capsule(PLACEHOLDER_HALF),
    }
}

fn half_range(v: f32) -> [f32; 2] {
    let v = if v.is_finite() { v.abs().min(std::f32::consts::PI) } else { 0.0 };
    [-v, v]
}

/// The limits that drive a joint: hard, else soft; `None` = unlimited data -> fixed joint.
fn limits_of(j: &Joint) -> Option<Limits> {
    j.hard.or(j.soft).filter(|l| l.twist > 0.0 || l.swing1 > 0.0 || l.swing2 > 0.0)
}

/// Joints locked rigid in the `secondary` build (trunk, neck, clavicles, wrists, fingers, forearm twist, mid-foot, toes):
/// their tiny segments are too light to servo stably, so the trunk and each hand / foot act as one rigid body.
fn is_minor(name: &str) -> bool {
    ["Spine", "Neck", "Clavicle", "Wrist", "Fingers", "ForearmTwist", "MidFoot", "Toes"]
        .iter()
        .any(|k| name.contains(k))
}

fn owner_joints(c: &Character) -> Vec<Option<usize>> {
    // body name -> first non-virtual-muscle joint that names it (else the first joint at all)
    let mut owner: Vec<Option<usize>> = vec![None; c.joints.len()];
    let mut taken: Vec<(&str, usize, bool)> = Vec::new();
    for (i, j) in c.joints.iter().enumerate() {
        let Some(bn) = j.body.as_deref() else { continue };
        if c.body(bn).is_none() {
            continue;
        }
        let virt = j.raw.attr("isVirtualMuscle") == Some("true");
        match taken.iter_mut().find(|(n, ..)| *n == bn) {
            None => taken.push((bn, i, virt)),
            Some(t) if t.2 && !virt => *t = (bn, i, virt),
            Some(_) => {}
        }
    }
    for (_, i, _) in taken {
        if let Some(slot) = owner.get_mut(i) {
            *slot = Some(i);
        }
    }
    owner
}

impl Sim {
    /// Build the world from `scene` at its t = 0 pose: characters dynamic, environment + ground static.
    pub fn from_scene(scene: &Scene) -> Sim {
        Sim::from_scene_with(scene, SimOptions { behaviours: false, forces: false, secondary: false })
    }
    /// `from_scene` + the CatchFall drives (`behaviour.rs`) + the scene's force events. Same start pose, same determinism.
    pub fn from_scene_active(scene: &Scene) -> Sim {
        Sim::from_scene_with(scene, SimOptions { behaviours: true, forces: true, secondary: false })
    }
    /// PLAY mode (h47): everything the crate simulates — CatchFall drives, force events and the `behave.rs` layer
    /// (pose hold, balance, Stagger / Arms Windmill / Legs Straighten, brace).
    pub fn from_scene_play(scene: &Scene) -> Sim {
        Sim::from_scene_with(scene, SimOptions { behaviours: true, forces: true, secondary: true })
    }
    pub fn from_scene_with(scene: &Scene, opt: SimOptions) -> Sim {
        Sim::from_scene_props(scene, opt, opt.secondary)
    }
    /// `from_scene_with`, choosing whether environment props with an `<InertialBody>` are dynamic (A/B: props fixed).
    pub fn from_scene_props(scene: &Scene, opt: SimOptions, dynamic_props: bool) -> Sim {
        let mut world = PhysicsWorld::new();
        world.gravity = if scene.gravity.iter().all(|v| v.is_finite()) { v3(scene.gravity) } else { Vec3::Y * -9.81 };
        let mut rigs = Vec::new();
        let mut self_pairs =
            SelfPairs { enabled: opt.secondary, mats: Materials::parse(&scene.raw), ..Default::default() };
        let mut mats: Vec<(ColliderHandle, String)> = Vec::new();
        for (i, c) in scene.characters.iter().enumerate() {
            let group = 1u32 << (i % 30);
            let tag = (i + 1) as u128;
            let posed = scene.posed_locals(c, 0.0);
            let selfc = Some((tag, &mut self_pairs.allowed));
            // starts NoPhysics (clip-driven), later full physics: the handover pose is the clip's, not the rest pose
            let handover =
                scene.incarnations().into_iter().find(|i| i.character == c.name).is_some_and(|i| !i.full_physics());
            rigs.push(add_character(&mut world, c, &posed, group, selfc, opt.secondary, handover, false, &mut mats));
        }
        if let Some(e) = &scene.environment {
            rigs.push(add_character(
                &mut world,
                e,
                &scene.posed_locals(e, 0.0),
                WORLD_GROUP,
                None,
                false,
                false,
                // PLAY: props with an `<InertialBody>` are dynamic (density from the file), the rest stays fixed
                dynamic_props,
                &mut mats,
            ));
        }
        // start-pose overlaps between a dynamic prop and any other dynamic body: never collided (as Endorphin starts them)
        let props: Vec<ColliderHandle> = world
            .colliders
            .iter()
            .filter(|(_, c)| c.user_data == 0 && c.parent().and_then(|p| world.bodies.get(p)).is_some_and(|b| b.is_dynamic()))
            .map(|(h, _)| h)
            .collect();
        for &ha in &props {
            for (hb, b) in world.colliders.iter() {
                let Some(a) = world.colliders.get(ha) else { continue };
                if hb == ha || a.parent() == b.parent() {
                    continue;
                }
                if !b.parent().and_then(|p| world.bodies.get(p)).is_some_and(|x| x.is_dynamic()) {
                    continue;
                }
                if rapier3d::parry::query::intersection_test(a.position(), a.shape(), b.position(), b.shape())
                    .unwrap_or(true)
                {
                    self_pairs.excluded.insert(ordered(ha, hb));
                }
            }
        }
        // the built-in ground is the environment's `background` surface (DefaultGroundCollision, tarmac)
        let ground_mat = scene
            .environment
            .iter()
            .flat_map(|e| e.bodies.iter())
            .flat_map(|b| b.collisions.iter())
            .find(|cs| cs.raw.attr("name") == Some("background"))
            .map_or(String::new(), |cs| cs.material.clone());
        if scene.use_default_ground {
            let ground = ColliderBuilder::cuboid(100.0, 0.5, 100.0)
                .position(Pose::from_translation(Vec3::new(0.0, -0.5, 0.0)))
                .friction(0.8)
                .collision_groups(InteractionGroups::new(
                    Group::from_bits_retain(WORLD_GROUP),
                    Group::ALL,
                    InteractionTestMode::And,
                ));
            mats.push((world.insert_collider(ground, None), ground_mat));
        }
        for (h, m) in mats {
            let k = self_pairs.mats.index(&m);
            self_pairs.mats.of.insert(h, k);
        }
        // materials named only by surfaces (no `<Material>` / pair rows) widen the table: re-square it
        let k = self_pairs.mats.names.len();
        if self_pairs.mats.pair.len() != k * k {
            let old = std::mem::take(&mut self_pairs.mats.pair);
            let k0 = (old.len() as f32).sqrt() as usize;
            self_pairs.mats.pair = vec![None; k * k];
            for a in 0..k0 {
                for b in 0..k0 {
                    self_pairs.mats.pair[a * k + b] = old[a * k0 + b];
                }
            }
        }
        let max_substep = if opt.secondary { MAX_SUBSTEP_PLAY } else { MAX_SUBSTEP };
        let mut sim = Sim {
            world,
            rigs,
            time: 0.0,
            catches: Vec::new(),
            forces: Vec::new(),
            behave: None,
            max_substep,
            self_pairs,
            holds: Vec::new(),
            force_welded: true,
        };
        sim.add_behaviours(scene, opt);
        if opt.secondary {
            sim.add_holds(scene);
        }
        if opt.secondary {
            sim.world.integration_parameters.num_solver_iterations = PLAY_SOLVER_ITERS;
            let ip = &mut sim.world.integration_parameters;
            ip.contact_softness.natural_frequency = PLAY_CONTACT_HZ;
            ip.static_contact_softness.natural_frequency = 2.0 * PLAY_CONTACT_HZ;
            // per-surface-pair friction / restitution / softness from the scene's `<MaterialPair>` table (hook)
            if let Some(i) = sim.self_pairs.mats.fallback {
                for (_, c) in sim.world.colliders.iter_mut() {
                    c.set_friction(i.friction);
                    c.set_restitution(i.restitution);
                    c.set_active_hooks(c.active_hooks() | ActiveHooks::MODIFY_SOLVER_CONTACTS);
                }
            }
            let dynamic: Vec<RigidBodyHandle> =
                sim.world.bodies.iter().filter(|(_, b)| b.is_dynamic()).map(|(h, _)| h).collect();
            for h in dynamic {
                if let Some(b) = sim.world.bodies.get_mut(h) {
                    b.enable_ccd(true);
                }
            }
            sim.behave = Some(behave::Behave::build(scene, &sim.rigs, &sim.world));
        }
        sim
    }
    fn add_behaviours(&mut self, scene: &Scene, opt: SimOptions) {
        for inst in scene.behaviour_instances().into_iter().filter(|_| opt.behaviours) {
            if !inst.enabled || !behaviour::CATCH_FALL.contains(&inst.behaviour.as_str()) {
                continue;
            }
            let Some(ri) = self.rigs.iter().position(|r| r.dynamic && r.name == inst.character) else { continue };
            if let Some(c) = CatchFall::new(self, ri, &inst) {
                self.catches.push(c);
            }
        }
        for ev in scene.force_events().into_iter().filter(|_| opt.forces) {
            if ev.end <= ev.start {
                continue;
            }
            let v = behaviour::force_vector(&ev);
            let group: Vec<RigidBodyHandle> = ev.targets.iter().filter_map(|(ch, body)| self.body_of(scene, ch, body)).collect();
            for &h in &group {
                // fixed bodies (static environment props) ignore the push
                if self.world.bodies.get(h).is_some_and(|b| b.is_dynamic()) {
                    let (start, end, rel) = (ev.start, ev.end, ev.relative_acceleration);
                    self.forces.push(ForceRt { start, end, body: h, group: group.clone(), v, rel, fired: false, dv: Vec3::ZERO });
                }
            }
        }
    }
    /// Rigid body of MetaBody `body` of character `ch` (`<Body>Mass` / `PropNN`): the joint naming it owns the body.
    fn body_of(&self, scene: &Scene, ch: &str, body: &str) -> Option<RigidBodyHandle> {
        let rig = self.rigs.iter().find(|r| r.name == ch)?;
        let ji = scene
            .characters
            .iter()
            .chain(scene.environment.iter())
            .find(|c| c.name == ch)
            .and_then(|c| c.joints.iter().position(|j| j.body.as_deref() == Some(body)))?;
        match rig.drives.get(ji) {
            Some(Drive::Body(h) | Drive::Shared { body: h, .. }) => Some(*h),
            _ => None,
        }
    }
    fn add_holds(&mut self, scene: &Scene) {
        for b in scene.behaviours.iter().filter(|b| b.kind == "TertiaryBehaviour") {
            let n = &b.raw;
            let lock = match n.attr("lockType") {
                Some("Lock") => true,
                Some("JoinBodies") => false,
                _ => continue,
            };
            if n.attr("enabled") == Some("false") {
                continue;
            }
            let int = |k: &str| n.attr(k).and_then(|v| v.trim().parse::<i32>().ok()).unwrap_or(0);
            let objects = n
                .child("Objects")
                .map(|o| {
                    o.kids("Object")
                        .map(|ob| {
                            let t = ob.text.trim().to_string();
                            let h = t.split_once(".MetaBodies.").and_then(|(c, m)| self.body_of(scene, c, m));
                            (t, h)
                        })
                        .collect()
                })
                .unwrap_or_default();
            self.holds.push(HoldRt {
                name: b.name.clone(),
                lock,
                start: int("startFrameNumber"),
                end: int("endFrameNumber"),
                strength: (
                    n.child("PositionStrength").map_or([0.0; 3], |s| s.vec3()),
                    n.child("AngularStrength").map_or([0.0; 3], |s| s.vec3()),
                ),
                objects,
                joints: Vec::new(),
                state: 0,
                dt: MAX_SUBSTEP,
            });
        }
    }
    /// PLAY build: the parsed `Lock` / `JoinBodies` constraint events (diagnostics).
    pub fn holds(&self) -> &[HoldRt] {
        &self.holds
    }
    /// World centre of mass of the rigid body of MetaBody `body` of `ch` (prop body frames sit at the world origin) and
    /// whether it is dynamic.
    pub fn body_pos(&self, scene: &Scene, ch: &str, body: &str) -> Option<([f32; 3], bool)> {
        let b = self.world.bodies.get(self.body_of(scene, ch, body)?)?;
        Some((b.center_of_mass().to_array(), b.is_dynamic()))
    }

    /// Advance by `dt` seconds (split into substeps of at most 1/120 s). Non-positive / non-finite `dt` is a no-op.
    pub fn step(&mut self, dt: f32) {
        if !(dt.is_finite() && dt > 0.0) {
            return;
        }
        let n = ((dt / self.max_substep).ceil() as usize).clamp(1, MAX_SUBSTEPS);
        self.world.integration_parameters.dt = dt / n as f32;
        let sub = dt / n as f32;
        for k in 0..n {
            let frame = ((self.time + sub * k as f32) * behaviour::FRAME_RATE + 1e-3).floor() as i32;
            for h in &mut self.holds {
                h.dt = sub;
                h.update(&mut self.world, frame);
            }
            if let Some(b) = self.behave.as_mut() {
                b.apply(&mut self.world, self.time + sub * k as f32, sub);
            }
            if !(self.catches.is_empty() && self.forces.is_empty()) {
                let frame = ((self.time + sub * k as f32) * behaviour::FRAME_RATE + 1e-3).floor() as i32;
                for f in &mut self.forces {
                    if f.due(frame) {
                        f.fire(&mut self.world);
                    }
                }
                for c in &mut self.catches {
                    if c.active_at(frame) {
                        c.tick(&mut self.world, sub);
                    }
                }
            }
            self.adhere(sub);
            self.world.step_with_events(&self.self_pairs, &());
        }
        self.time += dt;
    }
    /// `maxAdhesiveForce`: every touching pair whose interaction has one is pulled together along the contact normal
    /// by that force (the contact solver cancels it while they rest, so it only resists separation).
    fn adhere(&mut self, dt: f32) {
        let m = &self.self_pairs.mats;
        if !m.pair.iter().chain([&m.fallback]).any(|i| i.is_some_and(|i| i.adhesion > 0.0)) {
            return;
        }
        let mut pulls: Vec<(RigidBodyHandle, Vec3)> = Vec::new();
        for p in self.world.narrow_phase.contact_pairs().filter(|p| p.has_any_active_contact()) {
            let Some(i) = m.between(p.collider1, p.collider2).filter(|i| i.adhesion > 0.0) else { continue };
            let Some(k) = p.manifolds.iter().find(|k| !k.data.solver_contacts.is_empty()) else { continue };
            let n = k.data.normal;
            let body = |c| self.world.colliders.get(c).and_then(|c: &Collider| c.parent());
            if let Some(b) = body(p.collider1) {
                pulls.push((b, n * i.adhesion * dt));
            }
            if let Some(b) = body(p.collider2) {
                pulls.push((b, -n * i.adhesion * dt));
            }
        }
        // applied at the centre of mass: the pull's lever arm is ignored
        for (h, j) in pulls {
            if let Some(b) = self.world.bodies.get_mut(h).filter(|b| b.is_dynamic()) {
                b.apply_impulse(j, true);
            }
        }
    }
    /// Same-character contacts (`SelfPairs`) on / off. Default: on in the `secondary` (PLAY) build, off otherwise.
    pub fn set_self_collision(&mut self, on: bool) {
        self.self_pairs.enabled = on;
    }
    /// Diagnostics: per dynamic character, `(deepest overlap m, joint a, joint b)` over the collider pairs Endorphin
    /// self-collides (`SelfPairs`, independent of `set_self_collision`): minus the penetration depth, 0 = none.
    pub fn self_penetration(&self) -> Vec<(String, f32, usize, usize)> {
        let joint_of = |h: ColliderHandle| {
            let b = self.world.colliders.get(h)?.parent()?;
            self.rigs.iter().find_map(|r| r.handles.iter().position(|x| *x == Some(b)))
        };
        let mut out: Vec<(String, f32, usize, usize)> =
            self.rigs.iter().filter(|r| r.dynamic).map(|r| (r.name.clone(), 0.0, 0, 0)).collect();
        for (a, b) in &self.self_pairs.allowed {
            let (Some(ca), Some(cb)) = (self.world.colliders.get(*a), self.world.colliders.get(*b)) else { continue };
            let Ok(Some(k)) =
                rapier3d::parry::query::contact(ca.position(), ca.shape(), cb.position(), cb.shape(), 0.0)
            else {
                continue;
            };
            let ri = ca.user_data as usize - 1;
            let Some(name) = self.rigs.get(ri).map(|r| r.name.as_str()) else { continue };
            if let Some(o) = out.iter_mut().find(|o| o.0 == name)
                && k.dist < o.1
            {
                *o = (o.0.clone(), k.dist, joint_of(*a).unwrap_or(0), joint_of(*b).unwrap_or(0));
            }
        }
        out
    }

    /// Simulated seconds so far.
    pub fn time(&self) -> f32 {
        self.time
    }

    /// Per-character, per-joint world matrices (scene characters in file order, then the environment).
    pub fn pose(&self) -> Vec<CharPose> {
        self.rigs
            .iter()
            .map(|r| {
                let mut world: Vec<Xf> = Vec::with_capacity(r.drives.len());
                for d in &r.drives {
                    let x = match d {
                        Drive::Body(h) | Drive::Shared { body: h, .. } => {
                            self.world.bodies.get(*h).map_or_else(Xf::default, |b| xf_of(b.position()))
                        }
                        Drive::Follow { anchor, offset } => world.get(*anchor).map_or(*offset, |a| a.mul(offset)),
                        Drive::Fixed(x) => *x,
                    };
                    world.push(x);
                }
                CharPose { name: r.name.clone(), world }
            })
            .collect()
    }

    /// World y of the root joint (joint 0) of the named character.
    pub fn root_y(&self, character: &str) -> Option<f32> {
        let r = self.rigs.iter().find(|r| r.name == character)?;
        match r.drives.first()? {
            Drive::Body(h) => self.world.bodies.get(*h).map(|b| b.translation().y),
            Drive::Fixed(x) => Some(x.p[1]),
            Drive::Follow { .. } | Drive::Shared { .. } => {
                self.pose().into_iter().find(|p| p.name == character)?.world.first().map(|x| x.p[1])
            }
        }
    }

    /// World matrix of joint `idx` of `rig` right now.
    fn joint_world(&self, rig: &Rig, idx: usize) -> Option<Xf> {
        match rig.drives.get(idx)? {
            Drive::Body(h) => self.world.bodies.get(*h).map(|b| xf_of(b.position())),
            Drive::Follow { anchor, offset } | Drive::Shared { anchor, offset, .. } => {
                Some(self.joint_world(rig, *anchor)?.mul(offset))
            }
            Drive::Fixed(x) => Some(*x),
        }
    }
    /// World position of `joint` of `character` (None when unknown).
    pub fn joint_pos(&self, character: &str, joint: &str) -> Option<[f32; 3]> {
        let r = self.rigs.iter().find(|r| r.name == character)?;
        Some(self.joint_world(r, r.index_of(joint)?)?.p)
    }
    /// Number of behaviour controllers / force events that were built (0 for a passive sim).
    pub fn behaviour_counts(&self) -> (usize, usize) {
        (self.catches.len(), self.forces.len())
    }
    /// (urgency, predicted contact time s, ground height) of the first CatchFall on `character`.
    pub fn catch_fall_state(&self, character: &str) -> Option<(f32, f32, f32)> {
        let c = self.catches.iter().find(|c| self.rigs.get(c.rig()).is_some_and(|r| r.name == character))?;
        Some((c.last_urgency, c.last_contact_time, c.last_ground))
    }
    /// Live `behave.rs` diagnostics of a character (`None`: no secondary layer / unknown name).
    pub fn behaviour_state(&self, character: &str) -> Option<BehaviourState> {
        self.behave.as_ref()?.state(character)
    }
    /// Mass a `<ForceEvent>` on `h` accelerates: the body's, or (`force_welded`) the sum over the dynamic bodies of the
    /// engaged `JoinBodies` holds welding it (one level: the sample scenes' groups are disjoint).
    fn pushed_mass(&self, h: RigidBodyHandle) -> f32 {
        let own = self.world.bodies.get(h).map_or(0.0, |b| b.mass());
        if !self.force_welded {
            return own;
        }
        // ordered set: a fixed summation order keeps runs bit-identical
        let mut group: Vec<RigidBodyHandle> = vec![h];
        for hold in self.holds.iter().filter(|x| x.welds(h)) {
            group.extend(hold.objects.iter().filter_map(|o| o.1));
        }
        group.sort_by_key(|g| g.into_raw_parts());
        group.dedup();
        group.iter().filter_map(|g| self.world.bodies.get(*g)).filter(|b| b.is_dynamic()).map(|b| b.mass()).sum()
    }
    /// Diagnostics per built `<ForceEvent>` target: (start, end, accel m/s^2, body mass kg, pushed mass kg now, body
    /// linvel, linvel of the pushed group's centre of mass).
    pub fn force_diag(&self) -> Vec<(i32, i32, [f32; 3], f32, f32, [f32; 3], [f32; 3])> {
        self.forces
            .iter()
            .map(|f| {
                let b = self.world.bodies.get(f.body);
                let m = self.pushed_mass(f.body);
                let mut group: HashSet<RigidBodyHandle> = HashSet::from([f.body]);
                for hold in self.holds.iter().filter(|x| x.welds(f.body)) {
                    group.extend(hold.objects.iter().filter_map(|o| o.1));
                }
                let (mut p, mut ms) = (Vec3::ZERO, 0.0);
                for g in group.iter().filter_map(|g| self.world.bodies.get(*g)).filter(|b| b.is_dynamic()) {
                    p += g.linvel() * g.mass();
                    ms += g.mass();
                }
                let gv = if ms > 0.0 { p / ms } else { Vec3::ZERO };
                (
                    f.start,
                    f.end,
                    f.dv.to_array(),
                    b.map_or(0.0, |b| b.mass()),
                    m,
                    b.map_or([0.0; 3], |b| b.linvel().to_array()),
                    gv.to_array(),
                )
            })
            .collect()
    }
    /// Drop every scheduled `<ForceEvent>` push (A/B proofs: with vs without the gun shot).
    pub fn disable_force_events(&mut self) {
        self.forces.clear();
    }
    /// Diagnostics: bodies of `character` touching something else, as `(own joint index, other rig name, other joint index, normal impulse)`.
    pub fn contacts_of(&self, character: &str) -> Vec<(usize, String, Option<usize>, f32)> {
        let owner = |h: RigidBodyHandle| -> Option<(usize, usize)> {
            self.rigs
                .iter()
                .enumerate()
                .find_map(|(ri, r)| r.handles.iter().position(|x| *x == Some(h)).map(|j| (ri, j)))
        };
        let mut out = Vec::new();
        for cp in self.world.narrow_phase.contact_pairs() {
            let imp: f32 = cp.manifolds.iter().flat_map(|m| m.points.iter()).map(|p| p.data.impulse).sum();
            if imp <= 0.0 {
                continue;
            }
            let parent = |c: ColliderHandle| self.world.colliders.get(c).and_then(|c| c.parent());
            let (a, b) = (parent(cp.collider1).and_then(owner), parent(cp.collider2).and_then(owner));
            for (me, other) in [(a, b), (b, a)] {
                if let Some((ri, j)) = me
                    && self.rigs.get(ri).is_some_and(|r| r.name == character)
                {
                    let on = other
                        .and_then(|(oi, _)| self.rigs.get(oi).map(|r| r.name.clone()))
                        .unwrap_or_else(|| "ground".to_string());
                    out.push((j, on, other.map(|(_, oj)| oj), imp));
                }
            }
        }
        out
    }
    /// Diagnostics: per dynamic character, `(deepest penetration m, joint index)` into the static world (environment, ground):
    /// minus the depth inside a static collider of each body's lowest point (its support point along -Y) or of its origin
    /// (tunnelled past the contact). 0 = nothing sunk. Exact point queries (contact manifolds of huge boxes are unreliable).
    pub fn env_penetration(&self) -> Vec<(String, f32, usize)> {
        let statics: Vec<&Collider> = self
            .world
            .colliders
            .iter()
            .map(|(_, c)| c)
            .filter(|c| c.parent().and_then(|p| self.world.bodies.get(p)).is_none_or(|b| b.is_fixed()))
            .collect();
        let depth = |p: Vec3| {
            statics
                .iter()
                .filter(|s| s.shape().contains_point(s.position(), p))
                .map(|s| -s.shape().distance_to_point(s.position(), p, false).abs())
                .fold(0.0f32, f32::min)
        };
        let mut out = Vec::new();
        for r in self.rigs.iter().filter(|r| r.dynamic) {
            let mut best = (0.0f32, 0usize);
            for (j, h) in r.handles.iter().enumerate() {
                let Some(b) = h.and_then(|h| self.world.bodies.get(h)) else { continue };
                let mut pts = vec![b.translation()];
                for c in b.colliders().iter().filter_map(|h| self.world.colliders.get(*h)) {
                    if let Some(m) = c.shape().as_support_map() {
                        pts.push(m.support_point(c.position(), -Vec3::Y));
                    }
                    // shape-shape overlap: catches a limb stuck through a thin slab whose lowest point is already below it
                    for s in &statics {
                        let d = overlap_depth(c, s);
                        if d < best.0 {
                            best = (d, j);
                        }
                    }
                }
                for p in pts {
                    let d = depth(p);
                    if d < best.0 {
                        best = (d, j);
                    }
                }
            }
            out.push((r.name.clone(), best.0, best.1));
        }
        out
    }
    /// Diagnostics: every (collider of `character`, static collider) pair overlapping now, with the solver's view of it:
    /// `owner/joint shape groups depth | narrow-phase pair? active contact? manifold points, solver flags`.
    pub fn env_pairs(&self, character: &str) -> Vec<String> {
        let who = |h: ColliderHandle| -> String {
            let Some(p) = self.world.colliders.get(h).and_then(|c| c.parent()) else { return "ground".into() };
            self.rigs
                .iter()
                .find_map(|r| r.handles.iter().position(|x| *x == Some(p)).map(|j| format!("{}/{}", r.name, r.names[j])))
                .unwrap_or_else(|| "?".into())
        };
        let desc = |h: ColliderHandle, c: &Collider| {
            let g = c.collision_groups();
            format!(
                "{} {:?} {:?} body={:?} mat={:?} m={:x} f={:x} sg={:x}/{:x} sensor={} hooks={:?}",
                who(h),
                h.into_raw_parts(),
                c.shape().shape_type(),
                c.parent().and_then(|p| self.world.bodies.get(p)).map(|b| b.body_type()),
                self.self_pairs.mats.of.get(&h).and_then(|&i| self.self_pairs.mats.names.get(i)),
                g.memberships.bits(),
                g.filter.bits(),
                c.solver_groups().memberships.bits(),
                c.solver_groups().filter.bits(),
                c.is_sensor(),
                c.active_hooks()
            )
        };
        let Some(r) = self.rigs.iter().find(|r| r.name == character) else { return Vec::new() };
        let kin = r.handles.iter().flatten().filter_map(|h| self.world.bodies.get(*h)).filter(|b| b.is_kinematic()).count();
        // worst joint anchor gap (m) of this rig: kinematic clip poses that break the rig's bone lengths snap on handover
        let mine: HashSet<RigidBodyHandle> = r.handles.iter().flatten().copied().collect();
        let gap = self
            .world
            .impulse_joints
            .iter()
            .filter(|(_, j)| mine.contains(&j.body1()) && mine.contains(&j.body2()))
            .filter_map(|(_, j)| {
                let (b1, b2) = (self.world.bodies.get(j.body1())?, self.world.bodies.get(j.body2())?);
                let p1 = b1.position() * j.data.local_frame1;
                let p2 = b2.position() * j.data.local_frame2;
                let ang = if j.data.as_fixed().is_some() { p1.rotation.angle_between(p2.rotation) } else { 0.0 };
                Some(((p1.translation - p2.translation).length() + ang, r.handles.iter().position(|h| *h == Some(j.body2()))?))
            })
            .fold((0.0f32, 0usize), |a, b| if b.0 > a.0 { b } else { a });
        let mut out = vec![format!(
            "bodies={} kinematic={kin} joint_gap={:.4} at {}",
            r.handles.iter().flatten().count(),
            gap.0,
            r.names.get(gap.1).map_or("?", |s| s.as_str())
        )];
        // self pairs (`SelfPairs.allowed`) of this rig overlapping now, with the solver's view of each
        for &(ha, hb) in &self.self_pairs.allowed {
            let (Some(a), Some(b)) = (self.world.colliders.get(ha), self.world.colliders.get(hb)) else { continue };
            if self.rigs.get((a.user_data as usize).wrapping_sub(1)).is_none_or(|x| x.name != character) {
                continue;
            }
            let Ok(Some(k)) = rapier3d::parry::query::contact(a.position(), a.shape(), b.position(), b.shape(), 0.0)
            else {
                continue;
            };
            if k.dist >= 0.0 {
                continue;
            }
            let cp = self.world.narrow_phase.contact_pair(ha, hb);
            let md = cp
                .iter()
                .flat_map(|p| p.manifolds.iter())
                .flat_map(|m| m.points.iter())
                .map(|p| p.dist)
                .fold(f32::INFINITY, f32::min);
            out.push(format!(
                "self epa={:.4} manifold={md:.4} soft={:?} | {} | {} | pair={} active={}",
                k.dist,
                self.self_pairs.mats.between(ha, hb).map(|i| i.softness),
                desc(ha, a),
                desc(hb, b),
                cp.is_some(),
                cp.is_some_and(|p| p.has_any_active_contact())
            ));
        }
        for b in r.handles.iter().flatten().filter_map(|h| self.world.bodies.get(*h)) {
            for &ha in b.colliders() {
                let Some(a) = self.world.colliders.get(ha) else { continue };
                for (hs, s) in self.world.colliders.iter() {
                    // statics + other characters' bodies (cross-character overlaps)
                    let other = s.user_data != 0 && s.user_data != a.user_data;
                    if !other && !s.parent().and_then(|p| self.world.bodies.get(p)).is_none_or(|b| b.is_fixed()) {
                        continue;
                    }
                    let d = overlap_depth(a, s);
                    if d >= 0.0 {
                        continue;
                    }
                    let cp = self.world.narrow_phase.contact_pair(ha, hs);
                    let pts: usize = cp.map_or(0, |p| p.manifolds.iter().map(|m| m.points.len()).sum());
                    let flags = cp.and_then(|p| p.manifolds.first()).map(|m| m.data.solver_flags);
                    // solver's own depth (manifold) vs parry's exact contact (EPA)
                    let md = cp
                        .iter()
                        .flat_map(|p| p.manifolds.iter())
                        .flat_map(|m| m.points.iter())
                        .map(|p| p.dist)
                        .fold(f32::INFINITY, f32::min);
                    let k = rapier3d::parry::query::contact(a.position(), a.shape(), s.position(), s.shape(), 0.0)
                        .ok()
                        .flatten()
                        .map(|k| k.dist);
                    out.push(format!(
                        "d={d:.4} manifold={md:.4} epa={k:?} | {} | {} | pair={} active={} pts={pts} flags={flags:?}",
                        desc(ha, a),
                        desc(hs, s),
                        cp.is_some(),
                        cp.is_some_and(|p| p.has_any_active_contact())
                    ));
                }
            }
        }
        out
    }
    /// Names of the dynamic (simulated) characters.
    pub fn dynamic_characters(&self) -> Vec<&str> {
        self.rigs.iter().filter(|r| r.dynamic).map(|r| r.name.as_str()).collect()
    }

    /// Number of rigid bodies / joints in the world (diagnostics).
    pub fn counts(&self) -> (usize, usize) {
        (self.world.bodies.len(), self.world.impulse_joints.len())
    }
}

/// `-depth` (m) of `a` overlapping the static `b`, 0 if apart. Cuboids: separating axes over `b`'s face normals with `a`'s
/// support map (parry's EPA reports bogus 10-20 cm depths against the 100 m ground box: witness point 50-100 m off),
/// plus the edge x edge axes (`a`'s box axes / capsule-cylinder axis x `b`'s box axes) and EPA's normal, each measured
/// exactly with both support maps: every axis bounds the true depth from above, so the minimum never under-reports a
/// sink. Face axes alone over-report an edge contact (h51: Balcony ch04 shoulder capsule on the chair-seat edge read
/// -1.3..-2.1 cm while the solver manifold and EPA said ~0).
fn overlap_depth(a: &Collider, b: &Collider) -> f32 {
    if let (Some(bx), Some(m), Some(mb)) = (b.shape().as_cuboid(), a.shape().as_support_map(), b.shape().as_support_map())
    {
        let (r, o) = (b.position().rotation, b.position().translation);
        let unit = |k: usize| Vec3::from_array(std::array::from_fn(|i| if i == k { 1.0 } else { 0.0 }));
        let mut pen = f32::INFINITY;
        for (k, h) in bx.half_extents.to_array().into_iter().enumerate() {
            let n = r * unit(k);
            for n in [n, -n] {
                // how far `a` reaches past the face with outward normal `n`, measured inward
                let p = m.support_point(a.position(), -n);
                pen = pen.min(h - (p - o).dot(n));
            }
        }
        // overlap of the two shapes' projections on unit axis `n`
        let along = |n: Vec3| {
            let (amax, amin) = (m.support_point(a.position(), n).dot(n), m.support_point(a.position(), -n).dot(n));
            let (bmax, bmin) = (mb.support_point(b.position(), n).dot(n), mb.support_point(b.position(), -n).dot(n));
            (amax - bmin).min(bmax - amin)
        };
        let ra = a.position().rotation;
        let edges: Vec<Vec3> = if a.shape().as_cuboid().is_some() {
            (0..3).map(|k| ra * unit(k)).collect()
        } else if let Some(c) = a.shape().as_capsule() {
            vec![ra * (c.segment.b - c.segment.a)]
        } else if a.shape().as_cylinder().is_some() {
            vec![ra * Vec3::Y]
        } else {
            Vec::new()
        };
        let mut axes: Vec<Vec3> = edges.iter().flat_map(|e| (0..3).map(move |k| e.cross(r * unit(k)))).collect();
        if let Ok(Some(k)) = rapier3d::parry::query::contact(a.position(), a.shape(), b.position(), b.shape(), 0.0) {
            axes.push(k.normal1);
        }
        for n in axes {
            let l = n.length();
            if l > 1e-6 && l.is_finite() {
                pen = pen.min(along(n / l));
            }
        }
        // face axes alone miss edge/vertex separations: GJK's boolean test settles those
        let hit = pen > 0.0
            && rapier3d::parry::query::intersection_test(a.position(), a.shape(), b.position(), b.shape())
                .unwrap_or(true);
        return if hit { -pen } else { 0.0 };
    }
    match rapier3d::parry::query::contact(a.position(), a.shape(), b.position(), b.shape(), 0.0) {
        Ok(Some(k)) if k.dist < 0.0 => k.dist,
        _ => 0.0,
    }
}

fn add_character(
    world: &mut PhysicsWorld,
    c: &Character,
    locals: &[Xf],
    group: u32,
    selfc: Option<(u128, &mut HashSet<(ColliderHandle, ColliderHandle)>)>,
    rigid_minor: bool,
    handover: bool,
    env_dynamic: bool,
    mats: &mut Vec<(ColliderHandle, String)>,
) -> Rig {
    let dynamic = selfc.is_some();
    let tag = selfc.as_ref().map_or(0, |s| s.0);
    // per collider: (handle, owning joint, collidesWithSiblings, collidesWithAncestors)
    let mut cols: Vec<(ColliderHandle, usize, bool, bool)> = Vec::new();
    // start-pose world matrices incl. the scene placement
    let rest: Vec<Xf> = c.world_of(locals).iter().map(|x| c.xf.mul(x)).collect();
    let owner = owner_joints(c);
    // own group: only the PLAY build hands it to `SelfPairs`; the others keep the broad-phase cut (bit-identical runs)
    let hooked = dynamic && rigid_minor;
    let filter = if dynamic && !hooked { Group::from_bits_retain(!group) } else { Group::ALL };
    let groups = InteractionGroups::new(Group::from_bits_retain(group), filter, InteractionTestMode::And);
    let mut handles: Vec<Option<RigidBodyHandle>> = vec![None; c.joints.len()];
    let mut jhandles: Vec<Option<ImpulseJointHandle>> = vec![None; c.joints.len()];
    for (i, j) in c.joints.iter().enumerate() {
        if owner.get(i).copied().flatten() != Some(i) {
            continue;
        }
        let (Some(body), Some(jw)) = (j.body.as_deref().and_then(|n| c.body(n)), rest.get(i)) else { continue };
        // environment prop with an `<InertialBody>` + density: dynamic in the PLAY build
        let prop = !dynamic && env_dynamic && body.inertial.is_some() && body.density.is_some_and(|d| d > 0.0);
        let builder = if dynamic || prop { RigidBodyBuilder::dynamic() } else { RigidBodyBuilder::fixed() };
        let rb =
            world.insert_body(builder.pose(pose_of(jw)).linear_damping(0.05).angular_damping(0.4).can_sleep(false));
        // collision surfaces with geometry, else the inertial geometry (as endorphin_geo draws it)
        let flag = |cs: &hexe_geo::endorphin::Collision, k: &str| cs.raw.attr(k) != Some("false");
        let mut shapes: Vec<(Shaped, Xf, bool, bool, String)> = body
            .collisions
            .iter()
            .filter_map(|cs| {
                let (sib, anc) = (flag(cs, "collidesWithSiblings"), flag(cs, "collidesWithAncestors"));
                cs.geom.as_ref().map(|g| (shape_of(g), body.xf.mul(&cs.xf), sib, anc, cs.material.clone()))
            })
            .collect();
        // play mode: the foot keeps its inertial box (a flat sole) instead of the collision capsules, so it does not roll
        if rigid_minor
            && ["Heel", "Foot", "Toes"].iter().any(|k| body.name.contains(k))
            && let Some(g) = &body.inertial
        {
            let (sib, anc) = (shapes.iter().all(|s| s.2), shapes.iter().all(|s| s.3));
            let mat = shapes.first().map_or(String::new(), |s| s.4.clone());
            shapes = vec![(shape_of(g), body.xf, sib, anc, mat)];
        }
        if shapes.is_empty()
            && let Some(g) = &body.inertial
        {
            shapes.push((shape_of(g), body.xf, true, true, String::new()));
        }
        let vol: f32 = shapes.iter().map(|(s, ..)| s.volume).sum();
        let mass = body
            .mass()
            .map(|m| {
                if prop {
                    m * DENSITY_SCALE
                } else if rigid_minor {
                    MASS_REF * (m * DENSITY_SCALE / MASS_REF).powf(MASS_EXP)
                } else {
                    m
                }
            })
            .filter(|m| m.is_finite() && *m > 0.0)
            .unwrap_or(vol * FALLBACK_DENSITY)
            .max(MIN_MASS);
        for (s, local, sib, anc, mat) in shapes {
            let share = if vol > 0.0 { s.volume / vol } else { 1.0 };
            let at = Pose::from_parts(v3(local.p), quat_of(&local) * s.rot);
            let mut col =
                s.builder.position(at).mass(mass * share).friction(0.8).restitution(0.0).collision_groups(groups);
            if dynamic {
                col = col.user_data(tag);
            }
            if hooked || prop {
                col = col.active_hooks(ActiveHooks::FILTER_CONTACT_PAIRS);
            }
            let ch = world.insert_collider(col, Some(rb));
            mats.push((ch, mat));
            cols.push((ch, i, sib, anc));
        }
        if let Some(slot) = handles.get_mut(i) {
            *slot = Some(rb);
        }
    }
    // nearest ancestor-or-self that owns a body, per joint
    let mut anchor: Vec<Option<usize>> = vec![None; c.joints.len()];
    for (i, j) in c.joints.iter().enumerate() {
        let a = if handles.get(i).copied().flatten().is_some() {
            Some(i)
        } else {
            j.parent.and_then(|p| anchor.get(p).copied().flatten())
        };
        if let Some(s) = anchor.get_mut(i) {
            *s = a;
        }
    }
    let unlimited = rigid_minor && handover && c.joints.iter().all(|j| limits_of(j).is_none());
    if let Some((_, allowed)) = selfc {
        // body tree over the body-owning joints: parent body of joint i = anchor of its parent joint
        let up = |i: usize| c.joints.get(i).and_then(|j| j.parent).and_then(|p| anchor.get(p).copied().flatten());
        let above = |a: usize, b: usize| {
            // b strictly above a?
            let mut k = up(a);
            for _ in 0..c.joints.len() {
                match k {
                    Some(x) if x == b => return true,
                    Some(x) => k = up(x),
                    None => return false,
                }
            }
            false
        };
        for (x, &(ha, a, sa, aa)) in cols.iter().enumerate() {
            for &(hb, b, sb, ab) in &cols[x + 1..] {
                let ok = if a == b || up(a) == Some(b) || up(b) == Some(a) {
                    false
                } else if above(a, b) {
                    aa
                } else if above(b, a) {
                    ab
                } else if up(a) == up(b) {
                    sa && sb
                } else {
                    true
                };
                let (Some(ca), Some(cb)) = (world.colliders.get(ha), world.colliders.get(hb)) else { continue };
                let overlap =
                    rapier3d::parry::query::intersection_test(ca.position(), ca.shape(), cb.position(), cb.shape())
                        .unwrap_or(true);
                if ok && !overlap {
                    allowed.insert(ordered(ha, hb));
                }
            }
        }
        for (i, j) in c.joints.iter().enumerate() {
            let Some(Some(rb)) = handles.get(i) else { continue };
            let Some(a) = j.parent.and_then(|p| anchor.get(p).copied().flatten()) else { continue };
            let (Some(Some(ra)), Some(wa), Some(wj)) = (handles.get(a), rest.get(a), rest.get(i)) else { continue };
            let rel = inverse(wa).mul(wj);
            let f1 = Pose::from_parts(v3(rel.p), quat_of(&rel));
            let f2 = Pose::IDENTITY;
            let lim = limits_of(j).filter(|_| !(rigid_minor && is_minor(&j.name)));
            let spherical = lim.is_some();
            let joint: GenericJoint = match lim {
                // PLAY: a rig with no joint limits at all (FBX import, effectors DriveNone) = free ball joints in
                // Endorphin; a rest-pose fixed joint would snap the clip pose back at the full-physics handover
                None if unlimited => SphericalJointBuilder::new()
                    .local_frame1(f1)
                    .local_frame2(f2)
                    .contacts_enabled(false)
                    .build()
                    .into(),
                Some(l) => SphericalJointBuilder::new()
                    .local_frame1(f1)
                    .local_frame2(f2)
                    .contacts_enabled(false)
                    .limits(JointAxis::AngX, half_range(l.twist))
                    .limits(JointAxis::AngY, half_range(l.swing1))
                    .limits(JointAxis::AngZ, half_range(l.swing2))
                    .build()
                    .into(),
                None => {
                    FixedJointBuilder::new().local_frame1(f1).local_frame2(f2).contacts_enabled(false).build().into()
                }
            };
            let jh = world.insert_impulse_joint(*ra, *rb, joint);
            if spherical && let Some(slot) = jhandles.get_mut(i) {
                *slot = Some(jh);
            }
        }
    }
    // body name -> rigid body of the joint that owns it
    let owned =
        |name: &str| c.joints.iter().zip(&handles).find_map(|(j, h)| h.filter(|_| j.body.as_deref() == Some(name)));
    let drives = c
        .joints
        .iter()
        .enumerate()
        .map(|(i, j)| {
            let own = rest.get(i).copied().unwrap_or_default();
            let shared = j.body.as_deref().and_then(owned);
            match (handles.get(i).copied().flatten(), anchor.get(i).copied().flatten()) {
                (Some(h), _) => Drive::Body(h),
                (None, Some(a)) => {
                    let offset = inverse(&rest.get(a).copied().unwrap_or_default()).mul(&own);
                    match shared {
                        Some(body) => Drive::Shared { body, anchor: a, offset },
                        None => Drive::Follow { anchor: a, offset },
                    }
                }
                (None, None) => Drive::Fixed(own),
            }
        })
        .collect();
    let gains = behaviour::calibration_gains(c, behaviour::CALIBRATION);
    Rig {
        name: c.name.clone(),
        names: c.joints.iter().map(|j| j.name.clone()).collect(),
        gains,
        drives,
        handles,
        jhandles,
        anchor,
        rest,
        dynamic,
    }
}

#[cfg(test)]
mod tests;
