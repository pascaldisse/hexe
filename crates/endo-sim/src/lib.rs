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
}

/// Same-character collider pairs Endorphin collides (`CollisionSurface` `collidesWithSiblings` / `collidesWithAncestors`;
/// cousins always), minus parent/child (joint, `contacts_enabled(false)`) and pairs already overlapping at the start pose.
/// Character colliders carry `user_data = rig index + 1`; any pair of one rig not listed is dropped by the hook.
#[derive(Default)]
struct SelfPairs {
    allowed: HashSet<(ColliderHandle, ColliderHandle)>,
    enabled: bool,
}

fn ordered(a: ColliderHandle, b: ColliderHandle) -> (ColliderHandle, ColliderHandle) {
    if a.into_raw_parts() <= b.into_raw_parts() { (a, b) } else { (b, a) }
}

impl PhysicsHooks for SelfPairs {
    fn filter_contact_pair(&self, cx: &PairFilterContext) -> Option<SolverFlags> {
        let (Some(a), Some(b)) = (cx.colliders.get(cx.collider1), cx.colliders.get(cx.collider2)) else {
            return Some(SolverFlags::COMPUTE_IMPULSES);
        };
        if a.user_data == 0 || a.user_data != b.user_data {
            return Some(SolverFlags::COMPUTE_IMPULSES);
        }
        (self.enabled && self.allowed.contains(&ordered(cx.collider1, cx.collider2)))
            .then_some(SolverFlags::COMPUTE_IMPULSES)
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
        let mut world = PhysicsWorld::new();
        world.gravity = if scene.gravity.iter().all(|v| v.is_finite()) { v3(scene.gravity) } else { Vec3::Y * -9.81 };
        let mut rigs = Vec::new();
        let mut self_pairs = SelfPairs { allowed: HashSet::new(), enabled: opt.secondary };
        for (i, c) in scene.characters.iter().enumerate() {
            let group = 1u32 << (i % 30);
            let tag = (i + 1) as u128;
            let posed = scene.posed_locals(c, 0.0);
            rigs.push(add_character(&mut world, c, &posed, group, Some((tag, &mut self_pairs.allowed)), opt.secondary));
        }
        if let Some(e) = &scene.environment {
            rigs.push(add_character(&mut world, e, &scene.posed_locals(e, 0.0), WORLD_GROUP, None, false));
        }
        if scene.use_default_ground {
            let ground = ColliderBuilder::cuboid(100.0, 0.5, 100.0)
                .position(Pose::from_translation(Vec3::new(0.0, -0.5, 0.0)))
                .friction(0.8)
                .collision_groups(InteractionGroups::new(
                    Group::from_bits_retain(WORLD_GROUP),
                    Group::ALL,
                    InteractionTestMode::And,
                ));
            world.insert_collider(ground, None);
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
        };
        sim.add_behaviours(scene, opt);
        if opt.secondary {
            sim.world.integration_parameters.num_solver_iterations = PLAY_SOLVER_ITERS;
            let ip = &mut sim.world.integration_parameters;
            ip.contact_softness.natural_frequency = PLAY_CONTACT_HZ;
            ip.static_contact_softness.natural_frequency = 2.0 * PLAY_CONTACT_HZ;
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
        let g = self.world.gravity.length();
        for ev in scene.force_events().into_iter().filter(|_| opt.forces) {
            if ev.end <= ev.start {
                continue;
            }
            let accel = behaviour::force_accel(&ev, g);
            for (ch, body) in &ev.targets {
                let Some(rig) = self.rigs.iter().find(|r| r.name == *ch) else { continue };
                // `<Body>Mass` names the MetaBody; its joint is the one whose body that is
                let Some(ji) = scene
                    .characters
                    .iter()
                    .find(|c| c.name == *ch)
                    .and_then(|c| c.joints.iter().position(|j| j.body.as_deref() == Some(body.as_str())))
                else {
                    continue;
                };
                if let Some(Drive::Body(h)) = rig.drives.get(ji) {
                    self.forces.push(ForceRt { start: ev.start, end: ev.end, body: *h, point: v3(ev.xf.p), accel });
                }
            }
        }
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
            if let Some(b) = self.behave.as_mut() {
                b.apply(&mut self.world, self.time + sub * k as f32, sub);
            }
            if !(self.catches.is_empty() && self.forces.is_empty()) {
                let frame = ((self.time + sub * k as f32) * behaviour::FRAME_RATE + 1e-3).floor() as i32;
                for f in &self.forces {
                    if f.active_at(frame) {
                        f.tick(&mut self.world, sub);
                    }
                }
                for c in &mut self.catches {
                    if c.active_at(frame) {
                        c.tick(&mut self.world, sub);
                    }
                }
            }
            self.world.step_with_events(&self.self_pairs, &());
        }
        self.time += dt;
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
                        if let Ok(Some(k)) =
                            rapier3d::parry::query::contact(c.position(), c.shape(), s.position(), s.shape(), 0.0)
                            && k.dist < best.0
                        {
                            best = (k.dist, j);
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
    /// Names of the dynamic (simulated) characters.
    pub fn dynamic_characters(&self) -> Vec<&str> {
        self.rigs.iter().filter(|r| r.dynamic).map(|r| r.name.as_str()).collect()
    }

    /// Number of rigid bodies / joints in the world (diagnostics).
    pub fn counts(&self) -> (usize, usize) {
        (self.world.bodies.len(), self.world.impulse_joints.len())
    }
}

fn add_character(
    world: &mut PhysicsWorld,
    c: &Character,
    locals: &[Xf],
    group: u32,
    selfc: Option<(u128, &mut HashSet<(ColliderHandle, ColliderHandle)>)>,
    rigid_minor: bool,
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
        let builder = if dynamic { RigidBodyBuilder::dynamic() } else { RigidBodyBuilder::fixed() };
        let rb =
            world.insert_body(builder.pose(pose_of(jw)).linear_damping(0.05).angular_damping(0.4).can_sleep(false));
        // collision surfaces with geometry, else the inertial geometry (as endorphin_geo draws it)
        let flag = |cs: &hexe_geo::endorphin::Collision, k: &str| cs.raw.attr(k) != Some("false");
        let mut shapes: Vec<(Shaped, Xf, bool, bool)> = body
            .collisions
            .iter()
            .filter_map(|cs| {
                let (sib, anc) = (flag(cs, "collidesWithSiblings"), flag(cs, "collidesWithAncestors"));
                cs.geom.as_ref().map(|g| (shape_of(g), body.xf.mul(&cs.xf), sib, anc))
            })
            .collect();
        // play mode: the foot keeps its inertial box (a flat sole) instead of the collision capsules, so it does not roll
        if rigid_minor
            && ["Heel", "Foot", "Toes"].iter().any(|k| body.name.contains(k))
            && let Some(g) = &body.inertial
        {
            let (sib, anc) = (shapes.iter().all(|s| s.2), shapes.iter().all(|s| s.3));
            shapes = vec![(shape_of(g), body.xf, sib, anc)];
        }
        if shapes.is_empty()
            && let Some(g) = &body.inertial
        {
            shapes.push((shape_of(g), body.xf, true, true));
        }
        let vol: f32 = shapes.iter().map(|(s, ..)| s.volume).sum();
        let mass = body
            .mass()
            .map(|m| if rigid_minor { MASS_REF * (m * DENSITY_SCALE / MASS_REF).powf(MASS_EXP) } else { m })
            .filter(|m| m.is_finite() && *m > 0.0)
            .unwrap_or(vol * FALLBACK_DENSITY)
            .max(MIN_MASS);
        for (s, local, sib, anc) in shapes {
            let share = if vol > 0.0 { s.volume / vol } else { 1.0 };
            let at = Pose::from_parts(v3(local.p), quat_of(&local) * s.rot);
            let mut col =
                s.builder.position(at).mass(mass * share).friction(0.8).restitution(0.0).collision_groups(groups);
            if dynamic {
                col = col.user_data(tag);
            }
            if hooked {
                col = col.active_hooks(ActiveHooks::FILTER_CONTACT_PAIRS);
            }
            let ch = world.insert_collider(col, Some(rb));
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
