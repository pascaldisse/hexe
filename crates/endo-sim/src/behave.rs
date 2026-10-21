//! Behaviours for the endo-sim ragdoll (h45). Everything here acts through TORQUES on the rapier bodies (the same actuation a joint motor
//! gives), computed every substep from the live body states; nothing teleports a body or kinematically poses it.
//!
//! Layers, per dynamic character (all params from the `.ens` when present, documented defaults otherwise):
//! 1. **Pose hold** (joint motors): every body-to-body joint is a critically damped PD servo toward its target relative rotation
//!    (default = the scene start pose). Gain = `omega(strength)`, scaled by the subtree inertia so heavy and light limbs behave alike.
//!    Strength = the character's `DesiredPoseEvent` strength while active, else `HOLD_DEFAULT`.
//! 2. **Balance** (standing characters only): ankle strategy. A restoring horizontal force on the centre of mass
//!    `-M (kp e + kd v)` is realised as ankle torques `r x F` (shin +, heel -), clamped to a centre-of-pressure budget inside the
//!    foot, so a hard push really can topple the character. `e` = COM minus foot centre, relative to the start pose.
//! 3. **Scene behaviours** (`SecondaryBehaviour`, frame window x params): `Stagger` (more balance gain + arms windmill with
//!    ArmsSpeed/ArmsStrength; MaxSteps / StaggerSquarely are parsed but there is NO stepping), `Arms Windmill`
//!    (Frequency, ShoulderAmplitude, ElbowAmplitude, SyncOffset, yxPlaneOffset, Strength), `LegsStraighten` (hips + knees blend toward the
//!    rig rest pose by LeftLeg_/RightLeg_ Strength).
//! 4. **Force events** are NOT handled here: `behaviour.rs` (`ForceRt`, h46) owns the one `<ForceEvent>` path of the crate.
//! 5. **Catch-fall**: a smoothed fall measure (torso tilt, COM outside the foot budget) drives the arms to brace toward the fall
//!    direction (upper arms point at the ground ahead, elbows straight) and softens the legs so the body can fold instead of
//!    toppling as a plank.
use super::{Rig, inverse, quat_of, v3};
use hexe_geo::endorphin::{Character, Node, Scene};
use rapier3d::glamx::{Quat, Vec3};
use rapier3d::prelude::*;
use std::collections::HashMap;
use std::f32::consts::{PI, TAU};

/// Timeline rate used for `.ens` frame numbers when the file has no `positionBufferFrameRate`.
const DEFAULT_FRAME_RATE: f32 = 60.0;
/// Servo bandwidth (rad/s) at hold strength 0 / 1.
const W_LO: f32 = 5.0;
const W_HI: f32 = 26.0;
/// Hold strength of a character no `DesiredPoseEvent` covers (or after the event ended).
const HOLD_DEFAULT: f32 = 0.8;
/// Lever arm (m) of the body weight on the leg joints: their servo inertia is at least `M * LOAD_ARM^2`.
const LOAD_ARM: f32 = 0.2;
/// Balance: COM-force gains (1/s^2, 1/s), foot centre-of-pressure budget (m).
const BAL_KP: f32 = 60.0;
const BAL_KD: f32 = 14.0;
const COP_BUDGET: f32 = 0.085;
/// Balance: half width (m) of the foot soles around the foot segment, hip-lean gains (rad per m, rad per m/s), max lean (rad).
const SOLE: f32 = 0.05;
const LEAN_KP: f32 = 2.5;
const LEAN_KD: f32 = 0.5;
const LEAN_MAX: f32 = 0.5;
/// Muscle limit of the arm servos: max joint torque = `ARM_TAU_G` x the static torque of holding the arm subtree out horizontally
/// (`m g d`, d = pivot -> subtree COM). Without it a windmilling arm drives with the full (unbounded) PD stiffness: pressed into the
/// table it shoves the whole seated body up/back off its chair (Balcony c02/c03, T005 late collapse).
const ARM_TAU_G: f32 = 3.0;
/// Fall measure thresholds: torso tilt change (rad) where the fall starts / is certain, COM excursion (m) likewise.
const TILT_LO: f32 = 0.35;
const TILT_HI: f32 = 0.75;
const COM_LO: f32 = 0.10;
const COM_HI: f32 = 0.30;

/// Diagnostics of one character's controller.
#[derive(Clone, Debug, PartialEq)]
pub struct BehaviourState {
    /// Balance controller active (legs support the body).
    pub standing: bool,
    /// Smoothed fall measure 0 (upright) .. 1 (falling: arms brace).
    pub fall: f32,
    /// Horizontal COM excursion from the start-pose support (m).
    pub com_offset: f32,
    /// Torso tilt change from the start pose (rad).
    pub tilt: f32,
    /// Pose-hold strength in force at the last substep.
    pub hold: f32,
    /// Balance couple demanded / COP budget at the last substep (>1: clamp saturated, 0: not balancing).
    pub cop_demand: f32,
    /// Behaviours active now (`.ens` display names).
    pub active: Vec<String>,
    /// Behaviours found in the file but not simulated (name).
    pub ignored: Vec<String>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Part {
    Other,
    Spine,
    Hip,
    Knee,
    Ankle,
    Shoulder,
    Elbow,
}

/// One body of a character (the root body has `parent == None` and no servo).
struct Jc {
    name: String,
    part: Part,
    left: bool,
    body: RigidBodyHandle,
    parent_body: RigidBodyHandle,
    /// Spherical joint to the parent body (motor target is set on it each substep).
    joint: Option<ImpulseJointHandle>,
    /// Index (in `Ctl::jcs`) of the entry owning `parent_body`.
    parent: Option<usize>,
    rest_rel: Quat,
    rig_rel: Quat,
    /// Long axis of the limb in the body frame (towards its farthest child joint), unit or zero.
    axis: Vec3,
    /// Sign of a joint-X rotation that folds the limb forward (arms) / backward (knee).
    flex: f32,
}

#[derive(Clone, Debug)]
enum Kind {
    Stagger { windmill: bool, speed: f32, strength: f32 },
    Windmill { freq: f32, sh: f32, el: f32, sync: f32, yx: f32, strength: f32 },
    Legs { left: f32, right: f32 },
}

#[derive(Clone, Debug)]
struct Beh {
    name: String,
    kind: Kind,
    start: f32,
    end: f32,
}

impl Beh {
    /// Envelope 0..1 over the frame window: ramps in/out over at most 10 frames.
    fn env(&self, fr: f32) -> f32 {
        if fr < self.start || fr > self.end {
            return 0.0;
        }
        let ramp = ((self.end - self.start) * 0.5).clamp(1.0, 10.0);
        ((fr - self.start) / ramp).min((self.end - fr) / ramp).clamp(0.0, 1.0)
    }
}

struct Ctl {
    name: String,
    jcs: Vec<Jc>,
    standing: bool,
    behs: Vec<Beh>,
    ignored: Vec<String>,
    /// `(start, end, strength)` frame windows of `DesiredPoseEvent`s.
    poses: Vec<(f32, f32, f32)>,
    /// Heading frame from the start pose: forward / left (horizontal unit vectors).
    fwd: Vec3,
    left: Vec3,
    ankles: [Option<usize>; 2],
    feet: Vec<usize>,
    head: Option<usize>,
    pelvis: Option<usize>,
    /// COM minus foot centre at the start pose, and the start torso vector.
    e_ref: Vec3,
    torso_ref: Vec3,
    fall: f32,
    com_offset: f32,
    tilt: f32,
    hold: f32,
    cop_demand: f32,
    active: Vec<String>,
    /// Per jc: body hangs below the hips (leg chain) -> takes the balance reaction.
    lower: Vec<bool>,
}

pub(crate) struct Behave {
    fps: f32,
    chars: Vec<Ctl>,
}

/// Goal of one servo.
#[derive(Clone, Copy)]
enum Goal {
    /// Relative rotation (parent -> child) target.
    Rel(Quat, f32),
    /// Limb axis toward a world direction (swing only; twist is damped).
    Dir(Vec3, f32),
}

#[derive(Clone, Copy, Default)]
struct S {
    pos: Vec3,
    rot: Quat,
    com: Vec3,
    m: f32,
    i: f32,
    v: Vec3,
}

fn smooth(x: f32) -> f32 {
    let t = x.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn omega(strength: f32) -> f32 {
    W_LO + (W_HI - W_LO) * strength.clamp(0.0, 1.0)
}

fn horiz(v: Vec3) -> Vec3 {
    Vec3::new(v.x, 0.0, v.z)
}

fn unit_or(v: Vec3, d: Vec3) -> Vec3 {
    let l = v.length();
    if l.is_finite() && l > 1e-6 { v / l } else { d }
}

fn val(node: &Node, key: &str) -> Option<f32> {
    let v = node.kids("Variable").find(|v| v.attr("fk") == Some(key))?.attr("ja")?.trim();
    match v {
        "true" => Some(1.0),
        "false" => Some(0.0),
        _ => v.parse::<f32>().ok().filter(|f| f.is_finite()),
    }
}

fn part_of(name: &str) -> (Part, bool) {
    let left = name.starts_with("Left");
    let p = if name.contains("Virtual") || name.contains("Twist") {
        Part::Other
    } else if name.contains("Shoulder") {
        Part::Shoulder
    } else if name.contains("Elbow") {
        Part::Elbow
    } else if name.contains("Hip") {
        Part::Hip
    } else if name.contains("Knee") {
        Part::Knee
    } else if name.contains("Ankle") {
        Part::Ankle
    } else if name.contains("Spine") {
        Part::Spine
    } else {
        Part::Other
    };
    (p, left)
}

fn rel_quat(a: &crate::Xf, b: &crate::Xf) -> Quat {
    quat_of(&inverse(a).mul(b)).normalize()
}

fn parse_behaviours(scene: &Scene, c: &Character) -> (Vec<Beh>, Vec<String>) {
    let mut out = Vec::new();
    let mut ignored = Vec::new();
    for b in scene
        .behaviours
        .iter()
        .filter(|b| b.kind == "SecondaryBehaviour" && b.raw.attr("character") == Some(c.name.as_str()))
    {
        if b.raw.attr("enabled") == Some("false") {
            continue;
        }
        let label = b.raw.attr("behaviourName").unwrap_or(b.name.as_str()).to_string();
        let f = |k: &str| b.raw.attr(k).and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.0);
        let (start, end) = (f("startFrameNumber"), f("endFrameNumber"));
        let Some(p) = b.raw.child("Parameters") else { continue };
        let low = label.to_ascii_lowercase();
        let kind = if low.contains("stagger") {
            Some(Kind::Stagger {
                windmill: val(p, "ArmsWindmill").unwrap_or(1.0) > 0.5,
                speed: val(p, "ArmsSpeed").unwrap_or(0.8),
                strength: val(p, "ArmsStrength").unwrap_or(0.6),
            })
        } else if low.contains("windmill") {
            Some(Kind::Windmill {
                freq: val(p, "Frequency").unwrap_or(0.85),
                sh: val(p, "ShoulderAmplitude").unwrap_or(0.6),
                el: val(p, "ElbowAmplitude").unwrap_or(0.8),
                sync: val(p, "SyncOffset").unwrap_or(0.5),
                yx: val(p, "yxPlaneOffset").unwrap_or(0.1),
                strength: val(p, "Strength").unwrap_or(0.5),
            })
        } else if low.contains("legsstraighten") || low.contains("legs straighten") {
            let side =
                |on: &str, st: &str| if val(p, on).unwrap_or(0.0) > 0.5 { val(p, st).unwrap_or(0.6) } else { 0.0 };
            Some(Kind::Legs {
                left: side("StraightenLeftLeg", "LeftLeg_Strength"),
                right: side("StraightenRightLeg", "RightLeg_Strength"),
            })
        } else {
            None
        };
        match kind {
            Some(kind) => out.push(Beh { name: label, kind, start, end }),
            None => ignored.push(label),
        }
    }
    (out, ignored)
}

impl Behave {
    pub(crate) fn build(scene: &Scene, rigs: &[Rig], world: &PhysicsWorld) -> Behave {
        let fps = scene
            .raw
            .child("Engine")
            .and_then(|e| e.f32("positionBufferFrameRate"))
            .filter(|f| f.is_finite() && *f > 1.0)
            .unwrap_or(DEFAULT_FRAME_RATE);
        let mut chars = Vec::new();
        for (c, rig) in scene.characters.iter().zip(rigs) {
            if let Some(ctl) = Ctl::build(scene, c, rig, world) {
                chars.push(ctl);
            }
        }
        Behave { fps, chars }
    }
    pub(crate) fn state(&self, name: &str) -> Option<BehaviourState> {
        let c = self.chars.iter().find(|c| c.name == name)?;
        Some(BehaviourState {
            standing: c.standing,
            fall: c.fall,
            com_offset: c.com_offset,
            tilt: c.tilt,
            hold: c.hold,
            cop_demand: c.cop_demand,
            active: c.active.clone(),
            ignored: c.ignored.clone(),
        })
    }

    /// Controller torques for the substep `[t, t + dt)`.
    pub(crate) fn apply(&mut self, w: &mut PhysicsWorld, t: f32, dt: f32) {
        let fr = t * self.fps;
        for c in &mut self.chars {
            c.apply(w, fr, t, dt);
        }
    }
}

impl Ctl {
    fn build(scene: &Scene, c: &Character, rig: &Rig, world: &PhysicsWorld) -> Option<Ctl> {
        let rig_rest: Vec<crate::Xf> = c.world_of(&c.rest_locals()).iter().map(|x| c.xf.mul(x)).collect();
        let mut index: HashMap<usize, usize> = HashMap::new();
        let mut jcs: Vec<Jc> = Vec::new();
        let mut joint_of: Vec<usize> = Vec::new();
        for (i, j) in c.joints.iter().enumerate() {
            let Some(body) = rig.handles.get(i).copied().flatten() else { continue };
            let anchor = j.parent.and_then(|p| rig.anchor.get(p).copied().flatten());
            let (part, left) = part_of(&j.name);
            let mut jc = Jc {
                name: j.name.clone(),
                part,
                left,
                body,
                parent_body: body,
                joint: rig.jhandles.get(i).copied().flatten(),
                parent: None,
                rest_rel: Quat::IDENTITY,
                rig_rel: Quat::IDENTITY,
                axis: Vec3::ZERO,
                flex: 1.0,
            };
            if let Some(a) = anchor {
                let (Some(pb), Some(pk)) = (rig.handles.get(a).copied().flatten(), index.get(&a).copied()) else {
                    continue;
                };
                let (Some(ri), Some(ra), Some(gi), Some(ga)) =
                    (rig.rest.get(i), rig.rest.get(a), rig_rest.get(i), rig_rest.get(a))
                else {
                    continue;
                };
                jc.parent_body = pb;
                jc.parent = Some(pk);
                jc.rest_rel = rel_quat(ra, ri);
                jc.rig_rel = rel_quat(ga, gi);
            }
            index.insert(i, jcs.len());
            joint_of.push(i);
            jcs.push(jc);
        }
        if jcs.len() < 2 {
            return None;
        }
        let pos_of =
            |k: usize| rig.rest.get(joint_of.get(k).copied().unwrap_or(0)).map(|x| v3(x.p)).unwrap_or(Vec3::ZERO);
        let rot_of =
            |k: usize| rig.rest.get(joint_of.get(k).copied().unwrap_or(0)).map(quat_of).unwrap_or(Quat::IDENTITY);
        // limb axes: towards the farthest child joint's body
        let mut axes = vec![Vec3::ZERO; jcs.len()];
        let mut best = vec![0.0f32; jcs.len()];
        for k in 0..jcs.len() {
            if let Some(p) = jcs.get(k).and_then(|j| j.parent) {
                let d = pos_of(k) - pos_of(p);
                let l = d.length();
                if l > best.get(p).copied().unwrap_or(0.0)
                    && let (Some(a), Some(b)) = (axes.get_mut(p), best.get_mut(p))
                {
                    *a = rot_of(p).inverse() * d / l.max(1e-6);
                    *b = l;
                }
            }
        }
        // heading from the root frame (z forward); hinge flexion signs
        let fwd = unit_or(horiz(rot_of(0) * Vec3::Z), Vec3::Z);
        let left = Vec3::Y.cross(fwd);
        for (k, j) in jcs.iter_mut().enumerate() {
            j.axis = axes.get(k).copied().unwrap_or(Vec3::ZERO);
            let hinge = rot_of(k) * Vec3::X;
            let probe = Quat::from_axis_angle(hinge, 0.3) * Vec3::NEG_Y;
            let move_f = (probe - Vec3::NEG_Y).dot(fwd);
            let flex = match j.part {
                Part::Elbow => move_f,
                Part::Knee => -move_f,
                _ => 1.0,
            };
            j.flex = if flex < 0.0 { -1.0 } else { 1.0 };
        }
        let find = |n: &str| jcs.iter().position(|j| j.name == n);
        let (ankles, head, pelvis) = (
            [find("LeftAnkleJoint"), find("RightAnkleJoint")],
            find("UpperNeckJoint").or_else(|| find("LowerNeckJoint")),
            jcs.iter().position(|j| j.parent.is_none()),
        );
        let feet: Vec<usize> =
            jcs.iter().enumerate().filter(|(_, j)| j.name.contains("MidFoot")).map(|(k, _)| k).collect();
        // standing = pelvis well above the ankles (seated: thighs horizontal)
        let standing = match (ankles[0].or(ankles[1]), pelvis) {
            (Some(a), Some(p)) => pos_of(p).y - pos_of(a).y > 0.6,
            _ => false,
        };
        let (behs, ignored) = parse_behaviours(scene, c);
        let mut poses = Vec::new();
        if let Some(ev) = scene.raw.child("Events") {
            for e in ev.kids("DesiredPoseEvent").filter(|e| e.attr("character") == Some(c.name.as_str())) {
                let f = |k: &str| e.attr(k).and_then(|v| v.parse::<f32>().ok());
                poses.push((
                    f("startFrameNumber").unwrap_or(0.0),
                    f("endFrameNumber").unwrap_or(0.0),
                    f("strength").unwrap_or(1.0),
                ));
            }
        }
        let mut ctl = Ctl {
            name: c.name.clone(),
            jcs,
            standing,
            behs,
            ignored,
            poses,
            fwd,
            left,
            ankles,
            feet,
            head,
            pelvis,
            e_ref: Vec3::ZERO,
            torso_ref: Vec3::ZERO,
            fall: 0.0,
            com_offset: 0.0,
            tilt: 0.0,
            hold: 0.0,
            cop_demand: 0.0,
            active: Vec::new(),
            lower: Vec::new(),
        };
        ctl.lower = (0..ctl.jcs.len())
            .map(|k| {
                let mut x = Some(k);
                while let Some(i) = x {
                    if ctl.jcs.get(i).is_some_and(|j| j.part == Part::Hip) {
                        return true;
                    }
                    x = ctl.jcs.get(i).and_then(|j| j.parent);
                }
                false
            })
            .collect();
        let st = ctl.snapshot(world)?;
        let (com, _) = ctl.com(&st);
        let _ = com;
        ctl.e_ref = Vec3::ZERO;
        ctl.torso_ref = ctl.torso(&st);
        Some(ctl)
    }

    fn snapshot(&self, w: &PhysicsWorld) -> Option<Vec<S>> {
        let mut out = Vec::with_capacity(self.jcs.len());
        for j in &self.jcs {
            let b = w.bodies.get(j.body)?;
            let pi = b.mass_properties().local_mprops.principal_inertia();
            out.push(S {
                pos: b.translation(),
                rot: *b.rotation(),
                com: b.center_of_mass(),
                m: b.mass(),
                i: ((pi.x + pi.y + pi.z) / 3.0).max(1e-9),
                v: b.linvel(),
            });
        }
        Some(out)
    }

    fn com(&self, st: &[S]) -> (Vec3, f32) {
        let (mut sum, mut m) = (Vec3::ZERO, 0.0);
        for s in st {
            sum += s.com * s.m;
            m += s.m;
        }
        (if m > 0.0 { sum / m } else { Vec3::ZERO }, m)
    }

    /// Point of the foot segment (feet centres) closest to `from`, horizontal.
    fn support(&self, st: &[S], from: Vec3) -> Vec3 {
        let pts: Vec<Vec3> = if self.feet.is_empty() {
            self.ankles.iter().flatten().filter_map(|k| st.get(*k)).map(|s| s.pos).collect()
        } else {
            self.feet.iter().filter_map(|k| st.get(*k)).map(|s| s.pos).collect()
        };
        match pts.as_slice() {
            [] => Vec3::ZERO,
            [p] => *p,
            [a, b, ..] => {
                let (a, b) = (horiz(*a), horiz(*b));
                let ab = b - a;
                let t = if ab.length_squared() > 1e-8 {
                    ((horiz(from) - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                a + ab * t
            }
        }
    }

    /// Pelvis -> head vector.
    fn torso(&self, st: &[S]) -> Vec3 {
        match (self.pelvis.and_then(|k| st.get(k)), self.head.and_then(|k| st.get(k))) {
            (Some(p), Some(h)) => h.pos - p.pos,
            _ => Vec3::Y,
        }
    }

    fn hold_strength(&self, fr: f32) -> f32 {
        self.poses
            .iter()
            .filter(|(a, b, _)| fr >= *a && fr <= *b)
            .map(|(_, _, s)| *s)
            .reduce(f32::max)
            .unwrap_or(HOLD_DEFAULT)
    }

    fn apply(&mut self, w: &mut PhysicsWorld, fr: f32, t: f32, dt: f32) {
        let Some(st) = self.snapshot(w) else { return };
        let n = st.len();
        let (com, mass) = self.com(&st);
        let g = w.gravity.length().max(1e-3);
        // subtree inertia about each joint pivot (origin shifted to the root body for f32 precision)
        let o = st.first().map_or(Vec3::ZERO, |s| s.pos);
        let mut m_s: Vec<f32> = st.iter().map(|s| s.m).collect();
        let mut mc_s: Vec<Vec3> = st.iter().map(|s| (s.com - o) * s.m).collect();
        let mut mi_s: Vec<f32> = st.iter().map(|s| s.i + s.m * (s.com - o).length_squared()).collect();
        for k in (0..n).rev() {
            if let Some(p) = self.jcs.get(k).and_then(|j| j.parent) {
                let (m, mc, mi) = (m_s[k], mc_s[k], mi_s[k]);
                m_s[p] += m;
                mc_s[p] += mc;
                mi_s[p] += mi;
            }
        }
        let inertia = |k: usize| -> f32 {
            let p = st[k].pos - o;
            (mi_s[k] - 2.0 * p.dot(mc_s[k]) + m_s[k] * p.length_squared()).max(st[k].i)
        };
        // ---- fall measure
        let support = self.support(&st, com);
        let e_raw = horiz(com - support);
        // dead zone: the foot soles carry the last few cm
        let e = e_raw * ((e_raw.length() - SOLE).max(0.0) / e_raw.length().max(1e-6));
        let torso = self.torso(&st);
        let tilt_v = torso - self.torso_ref;
        let tilt = torso.normalize_or_zero().angle_between(self.torso_ref.normalize_or_zero());
        let f_tilt = smooth((tilt - TILT_LO) / (TILT_HI - TILT_LO));
        let f_com = if self.standing { smooth((e.length() - COM_LO) / (COM_HI - COM_LO)) } else { 0.0 };
        self.fall += (f_tilt.max(f_com) - self.fall) * (dt * 10.0).min(1.0);
        self.com_offset = e.length();
        self.tilt = tilt;
        let fall = self.fall;
        let fall_dir = unit_or(horiz(tilt_v) * (f_tilt + 0.01) + e * (f_com + 0.01), self.fwd);
        // ---- active behaviours
        self.active.clear();
        let mut staggering = false;
        let mut windmill: Option<(f32, f32, f32, f32, f32, f32)> = None; // freq, sh, el, sync, yx, strength*env
        let mut legs = [0.0f32; 2]; // left, right blend 0..1
        for b in self.behs.iter() {
            let env = b.env(fr);
            if env <= 0.0 {
                continue;
            }
            self.active.push(b.name.clone());
            match &b.kind {
                Kind::Stagger { windmill: wm, speed, strength } => {
                    staggering = true;
                    if *wm {
                        windmill = Some((0.9 * speed.max(0.2), 0.6, 0.8, 0.8, 0.1, strength * env));
                    }
                }
                Kind::Windmill { freq, sh, el, sync, yx, strength } => {
                    windmill = Some((*freq, *sh, *el, *sync, *yx, strength * env));
                }
                Kind::Legs { left, right } => {
                    legs[0] = legs[0].max(left * env);
                    legs[1] = legs[1].max(right * env);
                }
            }
        }
        let hold = self.hold_strength(fr);
        self.hold = hold;
        self.cop_demand = 0.0;
        // hip strategy: lean the pelvis (hence the trunk) so the COM moves over the foot segment; axis-angle vector, world frame
        let lean_axis = if self.standing {
            let vh = {
                let (mut sv, mut sm) = (Vec3::ZERO, 0.0);
                for x in &st {
                    sv += x.v * x.m;
                    sm += x.m;
                }
                if sm > 0.0 { horiz(sv / sm) } else { Vec3::ZERO }
            };
            let centre = {
 let f: Vec<Vec3> = self.feet.iter().filter_map(|k| st.get(*k)).map(|s| horiz(s.pos)).collect();
 if f.is_empty() { Vec3::ZERO } else { f.iter().fold(Vec3::ZERO, |a, p| a + *p) / f.len() as f32 }
};
let ec = if self.feet.is_empty() { e } else { horiz(com) - centre };
            let w = -(ec * LEAN_KP + vh * LEAN_KD);
            let w = if w.length() > LEAN_MAX { w.normalize_or_zero() * LEAN_MAX } else { w };
            Vec3::Y.cross(w) * (1.0 - fall)
        } else {
            Vec3::ZERO
        };
        // ---- goals
        let mut goals: Vec<Goal> = Vec::with_capacity(n);
        for (k, j) in self.jcs.iter().enumerate() {
            let mut s = hold;
            let mut q = j.rest_rel;
            match j.part {
                Part::Hip | Part::Knee => {
                    s *= 1.0 - 0.55 * fall;
                    if j.part == Part::Hip
                        && self.standing
                        && let Some(c) = st.get(k)
                    {
                        q = q * Quat::from_scaled_axis(c.rot.inverse() * -lean_axis);
                    }
                    let k = if j.left { legs[0] } else { legs[1] };
                    if k > 0.0 {
                        q = j.rest_rel.slerp(j.rig_rel, k.clamp(0.0, 1.0));
                        s = s.max(0.5 + 0.5 * k);
                    }
                }
                Part::Ankle => s = (s + 0.15).min(1.0),
                Part::Spine => s = (s + 0.1).min(1.0),
                _ => {}
            }
            goals.push(Goal::Rel(q, s));
        }
        for (k, j) in self.jcs.iter().enumerate() {
            if !matches!(j.part, Part::Shoulder | Part::Elbow) {
                continue;
            }
            let side = if j.left { 1.0 } else { -1.0 };
            let mut goal = None;
            if let Some((freq, sh, el, sync, yx, str_)) = windmill {
                let phase = TAU * freq * t + if j.left { sync * PI } else { 0.0 };
                let th = 1.2 + 2.0 * sh * phase.sin();
                let dir = (Vec3::NEG_Y * th.cos() + self.fwd * th.sin() + self.left * (side * (0.3 + yx)))
                    .normalize_or_zero();
                let s = (0.35 + 0.65 * str_.clamp(0.0, 1.0)).max(hold * 0.5);
                goal = Some(if j.part == Part::Shoulder {
                    Goal::Dir(dir, s)
                } else {
                    let ang = el * (0.5 + 0.5 * (phase + PI * 0.5).sin());
                    Goal::Rel(j.rig_rel * Quat::from_axis_angle(Vec3::X, j.flex * ang), s)
                });
            }
            if fall > 0.3 {
                let a = smooth((fall - 0.3) / 0.4);
                let dir = (fall_dir * 0.75 + Vec3::NEG_Y * 0.65 + self.left * (side * 0.12)).normalize_or_zero();
                goal = Some(if j.part == Part::Shoulder {
                    Goal::Dir(dir, 0.55 + 0.4 * a)
                } else {
                    Goal::Rel(j.rig_rel, 0.55 + 0.4 * a)
                });
            }
            if let Some(gl) = goal
                && let Some(slot) = goals.get_mut(k)
            {
                *slot = gl;
            }
        }
        // ---- joint motors (implicit in the rapier solver): target = goal relative to the joint's start frame
        for (k, j) in self.jcs.iter().enumerate() {
            let (Some(h), Some(goal), Some(c), Some(pi)) =
                (j.joint, goals.get(k).copied(), st.get(k), j.parent.and_then(|p| st.get(p)))
            else {
                continue;
            };
            let (q_goal, s) = match goal {
                Goal::Rel(q, s) => (q, s),
                Goal::Dir(d, s) if j.axis != Vec3::ZERO => {
                    let swing = Quat::from_rotation_arc(unit_or(c.rot * j.axis, Vec3::Y), unit_or(d, Vec3::NEG_Y));
                    ((pi.rot.inverse() * (swing * c.rot)).normalize(), s)
                }
                Goal::Dir(_, s) => (j.rest_rel, s),
            };
            let mut qt = (j.rest_rel.inverse() * q_goal).normalize();
            if qt.w < 0.0 {
                qt = -qt;
            }
            let wn = omega(s);
            // load-bearing joints carry the whole body: inertia floor = M * (0.2 m)^2
            let floor = if self.lower.get(k).copied().unwrap_or(false) { mass * LOAD_ARM * LOAD_ARM } else { 0.0 };
            let ie = inertia(k).max(floor);
            let (stiff, damp) = (ie * wn * wn, 2.0 * ie * wn);
            let ang = |v: f32| 2.0 * v.clamp(-1.0, 1.0).asin();
            let arm = matches!(j.part, Part::Shoulder | Part::Elbow);
            let tau_max = if arm {
                let reach = (mc_s[k] / m_s[k].max(1e-6) - (c.pos - o)).length().max(0.08);
                ARM_TAU_G * m_s[k] * g * reach
            } else {
                f32::INFINITY
            };
            if let Some(jt) = w.impulse_joints.get_mut(h, false) {
                for (axis, v) in [(JointAxis::AngX, qt.x), (JointAxis::AngY, qt.y), (JointAxis::AngZ, qt.z)] {
                    jt.data.set_motor_model(axis, MotorModel::ForceBased);
                    jt.data.set_motor_position(axis, ang(v), stiff, damp);
jt.data.set_motor_max_force(axis, tau_max);
                }
            }
        }
        // ---- balance (ankle strategy): the COM restoring force becomes a couple r x F about the ankles. The couple
        // acts +tau on the body above the ankles and -tau on the legs below the hips (mass-weighted), like the ground reaction.
        if self.standing && mass > 0.0 {
            let v_com = {
                let (mut s, mut m) = (Vec3::ZERO, 0.0);
                for x in &st {
                    s += x.v * x.m;
                    m += x.m;
                }
                if m > 0.0 { horiz(s / m) } else { Vec3::ZERO }
            };
            let gain = (if staggering { 1.25 } else { 1.0 }) * (1.0 - self.fall.clamp(0.0, 1.0));
            let f = -(e * BAL_KP + v_com * BAL_KD) * mass * gain;
            let mut tau = Vec3::ZERO;
            let n_a = self.ankles.iter().flatten().count().max(1) as f32;
            for a in self.ankles.iter().flatten() {
                if let Some(sa) = st.get(*a) {
                    tau += (com - sa.pos).cross(f / n_a);
                }
            }
            let cap = mass * g * COP_BUDGET;
            self.cop_demand = tau.length() / cap.max(1e-9);
            if tau.length() > cap {
                tau = tau.normalize_or_zero() * cap;
            }
            if tau.is_finite() {
                let (mut m_up, mut m_low) = (0.0, 0.0);
                for (k, j) in self.jcs.iter().enumerate() {
                    if self.lower.get(k).copied().unwrap_or(false) {
                        m_low += st[k].m
                    } else {
                        m_up += st[k].m
                    }
                    let _ = j;
                }
                if m_up > 0.0 && m_low > 0.0 {
                    for (k, j) in self.jcs.iter().enumerate() {
                        let low = self.lower.get(k).copied().unwrap_or(false);
                        let share = if low { -st[k].m / m_low } else { st[k].m / m_up };
                        if let Some(b) = w.bodies.get_mut(j.body) {
                            b.apply_torque_impulse(tau * (share * dt), true);
                        }
                    }
                }
            }
        }
    }
}
