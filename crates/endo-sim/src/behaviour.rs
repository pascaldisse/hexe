//! Active-ragdoll behaviour layer (h46). ONE behaviour end to end: `Bob2: CatchFall 2.0` / `Bob2: CatchFall`.
//!
//! What is data and what is authored (see `oracle/endorphin/NMB.md`):
//! * DATA: which behaviour runs on which character in which frames, and its parameters (`ArmsApart`, `ArmsSpeed`, `ArmsStrength`,
//!   `PredictGroundHeight`, `GroundHeight`) from the scene; the calibration set it drives with (`$character.Calibrations.Normal`,
//!   the `.nmb` default of its string-ref slot) and per-joint swing stiffness / damping / force limit from that set in the character.
//! * AUTHORED (the `.nmb` network is not decoded): the controller below. Arms reach for the predicted hand-ground contact point, the
//!   elbows extend as the reach grows, the head tucks. Every constant that is not data is a named `const` with its reasoning.
//!
//! Motor model: implicit-Euler PD on the swing error between a segment direction and its wanted direction, `tau = (S e - D w) / (1 + (D dt + S dt^2) / I)`,
//! clamped to the file's force limit, applied as an equal and opposite torque impulse on child and parent bodies (stable at any gain).
//! Units of the file's gains (N m / rad assumed) are UNVERIFIED.
use super::*;
use hexe_geo::endorphin_behaviour::BehaviourInstance;
use std::collections::HashMap;

/// Behaviour names this layer implements (`.nmb` `RefNames/Name`).
pub const CATCH_FALL: [&str; 2] = ["Bob2: CatchFall 2.0", "Bob2: CatchFall"];
/// Calibration set the behaviour drives with (default of the string-ref slot `$character.Calibrations.Normal` in M3012 / M2015).
pub const CALIBRATION: &str = "Normal";
/// Timeline rate: key spacing of the corpus (SURVEY.md) = frame numbers of events.
pub const FRAME_RATE: f32 = 30.0;
/// Reaction horizon (s): reach urgency ramps 0 -> 1 over the last `REACT` seconds before the predicted contact.
const REACT: f32 = 0.6;
/// Contact when the shoulder is this fraction of the arm length above the ground (arms land slightly bent).
const CONTACT_REACH: f32 = 0.9;
/// Head tuck at full urgency (rad, chin toward chest).
const TUCK: f32 = 0.35;
/// Reach blend rate (1/s) at ArmsSpeed 0 / 1.
const RATE_SLOW: f32 = 1.5;
const RATE_FAST: f32 = 8.0;
/// Hands never aim farther than this many arm lengths from the shoulder.
const MAX_AIM: f32 = 1.4;
/// ForceEvent push: acceleration = `relativeAcceleration` x |Vk| x |g| of the pushed object. |Vk| = the event's own
/// uniform gizmo scale (`endorphin.exe` ForceEventData getScale = |TransformMatrix row|, setScale clamps <= 50); the
/// "relative" acceleration is mass-independent: the force is that acceleration x the pushed mass (the target body, or
/// the whole welded group while a `JoinBodies` hold joins it). Units x|g| UNVERIFIED (the file carries no unit field).
const FORCE_USES_VK_SCALE: bool = true;

/// Swing gains of one joint (from the character calibration set).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gains {
    pub stiffness: f32,
    pub damping: f32,
    pub force_limit: f32,
}

/// `joint name -> Gains` of calibration set `set` of one character (Effector name -> jointName -> SkeletalEffectorParameters).
pub fn calibration_gains(c: &Character, set: &str) -> HashMap<String, Gains> {
    let mut eff_joint: HashMap<String, String> = HashMap::new();
    if let Some(e) = c.raw.child("Effectors") {
        for ef in e.kids("Effector") {
            if let (Some(n), Some(j)) = (ef.attr("name"), ef.attr("jointName")) {
                eff_joint.insert(n.to_string(), j.to_string());
            }
        }
    }
    let mut out = HashMap::new();
    let Some(cals) = c.raw.child("Calibrations") else { return out };
    let Some(cal) = cals.kids("Calibration").find(|k| k.attr("name") == Some(set)) else { return out };
    for p in cal.kids("SkeletalEffectorParameters") {
        let (Some(n), g) = (p.attr("name"), |k: &str| p.f32(k).filter(|v| v.is_finite() && *v >= 0.0)) else {
            continue;
        };
        let (Some(s), Some(d), Some(f)) = (g("swingStiffness"), g("swingDamping"), g("swingForceLimit")) else {
            continue;
        };
        if let Some(j) = eff_joint.get(n) {
            out.insert(j.clone(), Gains { stiffness: s, damping: d, force_limit: f });
        }
    }
    out
}

fn quat_arc(from: Vec3, to: Vec3) -> Quat {
    if from.length_squared() < 1e-12 || to.length_squared() < 1e-12 {
        return Quat::IDENTITY;
    }
    let q = Quat::from_rotation_arc(from.normalize(), to.normalize());
    if q.is_finite() { q } else { Quat::IDENTITY }
}

/// Direction blended from `a` toward `b` by `t` (both need not be unit).
fn slerp_dir(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    let a = a.normalize_or_zero();
    Quat::IDENTITY.slerp(quat_arc(a, b), t.clamp(0.0, 1.0)) * a
}

/// A body-fixed unit axis: `world = body_rot * local`.
#[derive(Clone, Copy)]
struct Axis(Vec3);

struct ArmRt {
    /// upper arm / forearm / hand bodies and the body the shoulder hangs on.
    upper: RigidBodyHandle,
    fore: RigidBodyHandle,
    hand: RigidBodyHandle,
    chest: RigidBodyHandle,
    /// rest direction of the upper arm in chest-body space; of the forearm in upper-arm-body space.
    rest_upper: Vec3,
    rest_fore: Vec3,
    g_sh: Gains,
    g_el: Gains,
    /// reach progress 0..1
    reach: f32,
    /// +1 left, -1 right (sign along the left-minus-right shoulder axis)
    side: f32,
}

struct HeadRt {
    head: RigidBodyHandle,
    neck: RigidBodyHandle,
    /// head long axis (neck joint -> top of head) in head-body space and its rest direction in chest-body space.
    axis_local: Axis,
    rest_chest: Vec3,
    g: Gains,
}

/// A running CatchFall on one character.
pub(crate) struct CatchFall {
    rig: usize,
    start: i32,
    end: i32,
    arms_apart: f32,
    arms_speed: f32,
    arms_strength: f32,
    predict_ground: bool,
    ground_height: f32,
    arms: Vec<ArmRt>,
    head: Option<HeadRt>,
    chest: RigidBodyHandle,
    /// forward of the standing character in chest space (lateral x up at the start pose).
    fwd_chest: Vec3,
    /// every body of the character with its mass (centre of mass).
    bodies: Vec<RigidBodyHandle>,
    /// diagnostics: last urgency / predicted contact time / ground height used.
    pub(crate) last_urgency: f32,
    pub(crate) last_contact_time: f32,
    pub(crate) last_ground: f32,
}

struct BodyState {
    pos: Vec3,
    rot: Quat,
    vel: Vec3,
    ang: Vec3,
    inertia: f32,
    mass: f32,
}

fn state(w: &PhysicsWorld, h: RigidBodyHandle) -> Option<BodyState> {
    let b = w.bodies.get(h)?;
    let pi = b.mass_properties().local_mprops.principal_inertia();
    let inertia = ((pi.x + pi.y + pi.z) / 3.0).max(1e-4);
    Some(BodyState {
        pos: b.translation(),
        rot: *b.rotation(),
        vel: b.linvel(),
        ang: b.angvel(),
        inertia,
        mass: b.mass(),
    })
}

impl CatchFall {
    /// Build for rig `rig_idx` from the scene instance; `None` when the rig lacks the arm chain.
    pub(crate) fn new(sim: &Sim, rig_idx: usize, inst: &BehaviourInstance) -> Option<CatchFall> {
        let rig = sim.rigs.get(rig_idx)?;
        let body = |name: &str| -> Option<RigidBodyHandle> {
            match rig.drives.get(rig.index_of(name)?)? {
                Drive::Body(h) => Some(*h),
                _ => None,
            }
        };
        let gains = |name: &str| rig.gains.get(name).copied();
        let chest = body("UpperSpineJoint")?;
        let w = &sim.world;
        let cs = state(w, chest)?;
        let mut arms = Vec::new();
        let (lsh, rsh) = (body("LeftShoulderJoint")?, body("RightShoulderJoint")?);
        let lat = (state(w, lsh)?.pos - state(w, rsh)?.pos).normalize_or_zero();
        for (side, sh, el, wr) in [
            (1.0f32, "LeftShoulderJoint", "LeftElbowJoint", "LeftWristJoint"),
            (-1.0, "RightShoulderJoint", "RightElbowJoint", "RightWristJoint"),
        ] {
            let (upper, fore, hand) = (body(sh)?, body(el)?, body(wr)?);
            let (u, f, hd) = (state(w, upper)?, state(w, fore)?, state(w, hand)?);
            let up_dir = (f.pos - u.pos).normalize_or_zero();
            let fo_dir = (hd.pos - f.pos).normalize_or_zero();
            arms.push(ArmRt {
                upper,
                fore,
                hand,
                chest,
                rest_upper: cs.rot.inverse() * up_dir,
                rest_fore: u.rot.inverse() * fo_dir,
                g_sh: gains(sh)?,
                g_el: gains(el)?,
                reach: 0.0,
                side,
            });
        }
        let head = (|| {
            let (head, neck) = (body("UpperNeckJoint")?, body("LowerNeckJoint")?);
            let hs = state(w, head)?;
            let top = rig.index_of("UpperNeckEndJoint")?;
            let top_pos = sim.joint_world(rig, top)?.p;
            let axis = (v3(top_pos) - hs.pos).normalize_or_zero();
            Some(HeadRt {
                head,
                neck,
                axis_local: Axis(hs.rot.inverse() * axis),
                rest_chest: cs.rot.inverse() * axis,
                g: gains("UpperNeckJoint")?,
            })
        })();
        let fwd = lat.cross(Vec3::Y).normalize_or_zero();
        let bodies = rig.drives.iter().filter_map(|d| if let Drive::Body(h) = d { Some(*h) } else { None }).collect();
        Some(CatchFall {
            rig: rig_idx,
            start: inst.start,
            end: inst.end,
            arms_apart: inst.f32("ArmsApart", 0.5).clamp(0.0, 1.0),
            arms_speed: inst.f32("ArmsSpeed", 0.5).clamp(0.0, 1.0),
            arms_strength: inst.f32("ArmsStrength", 0.5).clamp(0.0, 1.0),
            predict_ground: inst.bool("PredictGroundHeight", true),
            ground_height: inst.f32("GroundHeight", 0.0),
            arms,
            head,
            chest,
            fwd_chest: cs.rot.inverse() * fwd,
            bodies,
            last_urgency: 0.0,
            last_contact_time: f32::INFINITY,
            last_ground: 0.0,
        })
    }

    pub(crate) fn rig(&self) -> usize {
        self.rig
    }

    pub(crate) fn active_at(&self, frame: i32) -> bool {
        frame >= self.start && frame < self.end
    }

    /// Ground height under the character: the `GroundHeight` parameter, or (PredictGroundHeight) the highest static surface
    /// straight below the centre of mass found with a ray cast (scene ground / environment), else the parameter.
    fn ground(&self, w: &PhysicsWorld, com: Vec3) -> f32 {
        if !self.predict_ground {
            return self.ground_height;
        }
        let ray = Ray::new(com, -Vec3::Y);
        let filter = QueryFilter::default().groups(InteractionGroups::new(
            Group::from_bits_retain(WORLD_GROUP),
            Group::from_bits_retain(WORLD_GROUP),
            InteractionTestMode::And,
        ));
        match w.cast_ray(&ray, 100.0, false, filter) {
            Some((_, toi)) if toi.is_finite() => com.y - toi,
            _ => self.ground_height,
        }
    }

    /// One controller tick of `dt` seconds (call before the physics step).
    pub(crate) fn tick(&mut self, w: &mut PhysicsWorld, dt: f32) {
        let g = -w.gravity.y;
        // centre of mass + velocity
        let (mut m_sum, mut com, mut vel) = (0.0f32, Vec3::ZERO, Vec3::ZERO);
        for h in &self.bodies {
            if let Some(s) = state(w, *h) {
                m_sum += s.mass;
                com += s.pos * s.mass;
                vel += s.vel * s.mass;
            }
        }
        if m_sum <= 0.0 {
            return;
        }
        com /= m_sum;
        vel /= m_sum;
        let ground = self.ground(w, com);
        self.last_ground = ground;
        let Some(cs) = state(w, self.chest) else { return };
        // shoulder mid / arm length / half span
        let (Some(a0), Some(a1)) = (self.arms.first(), self.arms.get(1)) else { return };
        let (Some(s0), Some(s1)) = (state(w, a0.upper), state(w, a1.upper)) else { return };
        let (Some(e0), Some(h0)) = (state(w, a0.fore), state(w, a0.hand)) else { return };
        let arm_len = (e0.pos - s0.pos).length() + (h0.pos - e0.pos).length() + 0.08;
        let sh_mid = (s0.pos + s1.pos) * 0.5;
        let half_w = (s0.pos - s1.pos).length() * 0.5;
        let lat = {
            let l = s0.pos - s1.pos; // left minus right
            Vec3::new(l.x, 0.0, l.z).normalize_or_zero()
        };
        // time until the shoulder has fallen to CONTACT_REACH x arm length above the ground (ballistic COM motion)
        let drop = sh_mid.y - ground - CONTACT_REACH * arm_len;
        let down = -vel.y;
        let t_contact = if drop <= 0.0 { 0.0 } else { ((down + (down * down + 2.0 * g * drop).sqrt()) / g).max(0.0) };
        self.last_contact_time = t_contact;
        let urgency = (1.0 - t_contact / REACT).clamp(0.0, 1.0);
        self.last_urgency = urgency;
        // contact point: COM carried by its horizontal velocity; when slow, the lean of the centre of mass over the feet
        let mut hv = Vec3::new(vel.x, 0.0, vel.z);
        if hv.length() < 0.3 {
            let f = cs.rot * self.fwd_chest;
            hv = Vec3::new(f.x, 0.0, f.z).normalize_or_zero() * 0.3;
        }
        let reach_pt = Vec3::new(com.x + hv.x * t_contact, ground, com.z + hv.z * t_contact);
        let rate = RATE_SLOW + (RATE_FAST - RATE_SLOW) * self.arms_speed;
        let s = self.arms_strength;
        let tuck = self.head.as_ref().map(|_| urgency);
        for a in self.arms.iter_mut() {
            let (Some(u), Some(f), Some(c)) = (state(w, a.upper), state(w, a.fore), state(w, a.chest)) else {
                continue;
            };
            let step = (urgency - a.reach).clamp(-rate * dt, rate * dt);
            a.reach = (a.reach + step).clamp(0.0, 1.0);
            let side_pt = reach_pt + lat * a.side * half_w * (0.5 + self.arms_apart);
            let mut aim = side_pt - u.pos;
            if aim.length() > MAX_AIM * arm_len {
                aim = aim.normalize_or_zero() * (MAX_AIM * arm_len);
            }
            let rest = c.rot * a.rest_upper;
            let want_up = slerp_dir(rest, aim, a.reach);
            let cur_up = (f.pos - u.pos).normalize_or_zero();
            pd_align(w, a.upper, a.chest, cur_up, want_up, scaled(a.g_sh, s), dt);
            let rest_fore = u.rot * a.rest_fore;
            let want_fore = slerp_dir(rest_fore, cur_up, a.reach);
            let cur_fore = state(w, a.hand).map_or(Vec3::ZERO, |hd| (hd.pos - f.pos).normalize_or_zero());
            pd_align(w, a.fore, a.upper, cur_fore, want_fore, scaled(a.g_el, s), dt);
        }
        if let (Some(hr), Some(t)) = (self.head.as_ref(), tuck)
            && let (Some(hd), Some(c)) = (state(w, hr.head), state(w, self.chest))
        {
            let rest = c.rot * hr.rest_chest;
            let fwd = c.rot * self.fwd_chest;
            let ang = TUCK * t;
            let want = (rest * ang.cos() + fwd * ang.sin()).normalize_or_zero();
            let cur = hd.rot * hr.axis_local.0;
            pd_align(w, hr.head, hr.neck, cur, want, scaled(hr.g, s), dt);
        }
    }
}

fn scaled(g: Gains, s: f32) -> Gains {
    Gains { stiffness: g.stiffness * s, damping: g.damping * s.sqrt(), force_limit: g.force_limit * s }
}

/// Implicit-Euler PD torque rotating `cur` onto `want` (swing only), reaction on the parent.
fn pd_align(
    w: &mut PhysicsWorld,
    child: RigidBodyHandle,
    parent: RigidBodyHandle,
    cur: Vec3,
    want: Vec3,
    g: Gains,
    dt: f32,
) {
    if g.stiffness <= 0.0 && g.damping <= 0.0 {
        return;
    }
    let (Some(c), Some(p)) = (state(w, child), state(w, parent)) else { return };
    let e = quat_arc(cur, want).to_scaled_axis();
    let wrel = c.ang - p.ang;
    let i_red = if p.inertia > 0.0 { c.inertia * p.inertia / (c.inertia + p.inertia) } else { c.inertia };
    let raw = e * g.stiffness - wrel * g.damping;
    let mut tau = raw / (1.0 + (g.damping * dt + g.stiffness * dt * dt) / i_red);
    let m = tau.length();
    if g.force_limit > 0.0 && m > g.force_limit {
        tau *= g.force_limit / m;
    }
    if !tau.is_finite() {
        return;
    }
    if let Some(b) = w.bodies.get_mut(child) {
        b.apply_torque_impulse(tau * dt, true);
    }
    if let Some(b) = w.bodies.get_mut(parent) {
        b.apply_torque_impulse(-tau * dt, true);
    }
}

/// `<ForceEvent>` made runtime: a constant push on one body for frames `[start, end)`.
pub(crate) struct ForceRt {
    pub(crate) start: i32,
    pub(crate) end: i32,
    pub(crate) body: RigidBodyHandle,
    pub(crate) point: Vec3,
    pub(crate) accel: Vec3,
}

impl ForceRt {
    pub(crate) fn active_at(&self, frame: i32) -> bool {
        frame >= self.start && frame < self.end
    }
    /// `mass` = the pushed mass (kg): the body's own, or its welded group's (`Sim::pushed_mass`).
    pub(crate) fn tick(&self, w: &mut PhysicsWorld, dt: f32, mass: f32) {
        let Some(b) = w.bodies.get_mut(self.body) else { return };
        let f = self.accel * mass * dt;
        b.apply_impulse_at_point(f, self.point, true);
    }
}

pub(crate) fn force_accel(ev: &hexe_geo::endorphin_behaviour::ForceEvent, gravity: f32) -> Vec3 {
    let dir = v3(ev.xf.k);
    let scale = if FORCE_USES_VK_SCALE { dir.length() } else { 1.0 };
    dir.normalize_or_zero() * (ev.relative_acceleration * scale * gravity)
}
