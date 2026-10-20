//! `endosim` SOP (hexe-native, no Houdini twin): runs the passive-ragdoll simulation (`endo-sim`, rapier3d) of the Endorphin scene
//! that the upstream `file` SOP opened (found through the `endo_file` detail attr) and outputs the scene geometry posed by the
//! simulation at frame `$F`. Frame 1 = the scene's start pose; frame N = N-1 steps of 1/fps later. The node steps lazily and
//! caches every simulated pose, so scrubbing forwards advances the sim and scrubbing back replays the cache; a changed file / fps
//! restarts. Param `frame`: 0 = follow the timeline `$F` (`ctx.time * ctx.fps + 1`), > 0 = fixed 1-based frame.
//! Param `behaviours` (default 1): 1 = active ragdoll (`endo_sim::behaviour`: the scene's `Bob2: CatchFall` joint drives + its `<ForceEvent>`
//! pushes), 0 = the passive ragdoll of h44 (characters fall and slump). Behaviours without a controller are ignored (NMB.md).
use crate::SopNode;
use endo_sim::{CharPose, Sim, SimOptions};
use hexe_geo::endorphin::{Document, Scene};
use hexe_geo::{AttrClass, AttrData, Detail};
use hexe_graph::{CookCtx, CookError, InputSpec, Node, ParamSpec, ParamValue};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

/// Longest simulated clip (frames): 80 s at 30 fps; bounds both cook time and the pose cache.
pub const MAX_FRAME: i32 = 2400;

pub fn spec() -> Vec<ParamSpec> {
    vec![
        ParamSpec::new("frame", ParamValue::I(0))
            .label("Frame Override (0 = follow timeline $F; else 1-based sim frame)")
            .range(0.0, MAX_FRAME as f32),
        ParamSpec::new("behaviours", ParamValue::I(1))
            .label("Behaviours (1 = active ragdoll: scene behaviours + force events, 0 = passive ragdoll)")
            .range(0.0, 1.0),
    ]
}

pub fn new() -> Box<SopNode> {
    Box::new(EndoSim { specs: spec(), state: Mutex::new(None) })
}

type FileKey = (String, Option<SystemTime>, u64, u32, bool);

/// Everything the node keeps between cooks: the scene, the live sim and the pose of every frame simulated so far.
struct State {
    key: FileKey,
    scene: Arc<Scene>,
    sim: Sim,
    /// `poses[k]` = state after `k` steps (frame `k + 1`).
    poses: Vec<Vec<CharPose>>,
    last: Option<(i32, Arc<Detail>)>,
}

struct EndoSim {
    specs: Vec<ParamSpec>,
    state: Mutex<Option<State>>,
}

/// First value of the string detail attr `name`.
fn detail_str(d: &Detail, name: &str) -> Option<String> {
    match &d.attr(AttrClass::Detail, name)?.data {
        AttrData::Str { idx, table } => table.get(usize::try_from(*idx.first()?).ok()?).cloned(),
        _ => None,
    }
}

fn load_scene(path: &str) -> Result<Scene, CookError> {
    match hexe_geo::read_endorphin_file(path).map_err(|e| CookError::Failed(e.to_string()))? {
        Document::Scene(s) => Ok(*s),
        _ => Err(CookError::Failed(format!("endosim: `{path}` is not an Endorphin scene (.ens)"))),
    }
}

impl Node<Detail> for EndoSim {
    fn type_name(&self) -> &'static str {
        "endosim"
    }
    fn inputs(&self) -> &[InputSpec] {
        &[InputSpec { name: "scene" }]
    }
    fn params(&self) -> &[ParamSpec] {
        &self.specs
    }
    fn time_dependent(&self) -> bool {
        true
    }
    fn cook(&self, ctx: &CookCtx, inputs: &[Arc<Detail>]) -> Result<Arc<Detail>, CookError> {
        let input = inputs
            .first()
            .ok_or_else(|| CookError::Failed("endosim needs a `file` SOP (.ens scene) on its input".into()))?;
        let path = detail_str(input, "endo_file").ok_or_else(|| {
            CookError::Failed(
                "endosim: input is not an Endorphin scene (no `endo_file`); wire a file SOP on a .ens".into(),
            )
        })?;
        let fixed = ctx.params.int("frame").unwrap_or(0);
        if !(0..=MAX_FRAME).contains(&fixed) {
            return Err(CookError::BadParam(format!("frame {fixed} outside 0..={MAX_FRAME}")));
        }
        if !(ctx.fps.is_finite() && ctx.fps > 0.0 && ctx.fps <= 1000.0) {
            return Err(CookError::BadParam(format!("fps {} must be in (0, 1000]", ctx.fps)));
        }
        let time = if ctx.time.is_finite() { ctx.time.max(0.0) } else { 0.0 };
        let want =
            if fixed > 0 { fixed } else { ((time * ctx.fps).round() as i64 + 1).clamp(1, i64::from(MAX_FRAME)) as i32 };
        let meta = std::fs::metadata(&path).map_err(|e| CookError::Failed(format!("{path}: {e}")))?;
        let active = ctx.params.int("behaviours").unwrap_or(1) != 0;
        let key: FileKey = (path.clone(), meta.modified().ok(), meta.len(), ctx.fps.to_bits(), active);
        let mut slot = self.state.lock().map_err(|_| CookError::Failed("endosim state poisoned".into()))?;
        if slot.as_ref().is_none_or(|s| s.key != key) {
            let scene = load_scene(&path)?;
            let sim = Sim::from_scene_with(&scene, SimOptions { behaviours: active, forces: active });
            let poses = vec![sim.pose()];
            *slot = Some(State { key, scene: Arc::new(scene), sim, poses, last: None });
        }
        let Some(st) = slot.as_mut() else { return Err(CookError::Failed("endosim state empty".into())) };
        if let Some((f, d)) = &st.last
            && *f == want
        {
            return Ok(Arc::clone(d));
        }
        let dt = 1.0 / ctx.fps;
        let idx = usize::try_from(want - 1).unwrap_or(0);
        while st.poses.len() <= idx {
            st.sim.step(dt);
            st.poses.push(st.sim.pose());
        }
        let Some(pose) = st.poses.get(idx) else { return Err(CookError::Failed("endosim: pose cache miss".into())) };
        let worlds: Vec<&[hexe_geo::endorphin::Xf]> = pose.iter().map(|p| p.world.as_slice()).collect();
        let d = Arc::new(hexe_geo::endorphin_geo::scene_detail_posed(&st.scene, idx as f32 * dt, &worlds));
        st.last = Some((want, Arc::clone(&d)));
        Ok(d)
    }
}

#[cfg(test)]
mod tests {
    use crate::load_scene;
    use hexe_geo::{AttrClass, AttrData};
    use std::path::PathBuf;

    fn balcony() -> Option<PathBuf> {
        let home = std::env::var_os("HOME")?;
        let root = std::env::var_os("ENDORPHIN_RESOURCES").map(PathBuf::from).unwrap_or_else(|| {
            PathBuf::from(home).join("Library/Application Support/CrossOver/Bottles/endorphin/drive_c/Program Files (x86)/NaturalMotion/endorphin 2.5.2/Resources")
        });
        let f = root.join("Scenes/Sample Scenes/Balcony.ens");
        f.is_file().then_some(f)
    }

    fn graph(file: &str) -> String {
        format!(
            r#"{{"nodes":[{{"name":"f","type":"file","params":{{"file":{file:?}}}}},{{"name":"s","type":"endosim","params":{{}}}}],"edges":[["f",0,"s",0]],"display":"s"}}"#
        )
    }

    fn point(d: &hexe_geo::Detail, joint: &str, character: &str) -> Option<[f32; 3]> {
        let (Some(j), Some(c)) = (d.attr(AttrClass::Point, "joint"), d.attr(AttrClass::Point, "character")) else {
            return None;
        };
        let (AttrData::Str { idx: ji, table: jt }, AttrData::Str { idx: ci, table: ct }) = (&j.data, &c.data) else {
            return None;
        };
        (0..d.point_count as usize).find_map(|p| {
            let (a, b) = (jt.get(*ji.get(p)? as usize)?, ct.get(*ci.get(p)? as usize)?);
            (a == joint && b == character).then(|| d.p().get(p).copied()).flatten()
        })
    }

    /// file -> endosim over the real Balcony: $F scrubs the simulation (forward, then back = cache replay).
    #[test]
    fn frame_scrubs_the_simulation() {
        let Some(f) = balcony() else { return };
        let (mut g, out) = load_scene(&graph(&f.to_string_lossy())).expect("loads");
        g.set_fps(30.0);
        let root_y = |g: &mut hexe_graph::Graph<hexe_geo::Detail>, frame: u32| {
            let d = g.cook(out, (frame as f32 - 1.0) / 30.0).expect("cooks");
            point(&d, "root", "character01").expect("root point")[1]
        };
        let (y1, y30, y90) = (root_y(&mut g, 1), root_y(&mut g, 30), root_y(&mut g, 90));
        assert!((y1 - 3.848).abs() < 5e-3, "frame 1 = start pose {y1}");
        assert!(y1 - y30 > 0.5 && y30.is_finite() && y90.is_finite(), "collapsed by f30: {y1} -> {y30} -> {y90}");
        let y30_again = root_y(&mut g, 30);
        assert_eq!(y30.to_bits(), y30_again.to_bits(), "scrub back replays the cache");
    }

    fn catch_fall() -> Option<PathBuf> {
        let f = balcony()?.parent()?.parent()?.join("Behaviour Scenes/Catch fall2.0.ens");
        f.is_file().then_some(f)
    }

    /// file -> endosim over Catch fall2.0: `behaviours` 1 (default) drives the arms/head, 0 is the passive ragdoll; both deterministic.
    #[test]
    fn behaviours_param_switches_the_active_ragdoll() {
        let Some(f) = catch_fall() else { return };
        let head_y = |param: &str| {
            let g = format!(
                r#"{{"nodes":[{{"name":"f","type":"file","params":{{"file":{:?}}}}},{{"name":"s","type":"endosim","params":{{{param}}}}}],"edges":[["f",0,"s",0]],"display":"s"}}"#,
                f.to_string_lossy()
            );
            let (mut g, out) = load_scene(&g).expect("loads");
            g.set_fps(30.0);
            let d = g.cook(out, 59.0 / 30.0).expect("cooks"); // frame 60
            point(&d, "UpperNeckJoint", "character01").expect("head point")[1]
        };
        let (def, on, off) = (head_y(""), head_y(r#""behaviours":1"#), head_y(r#""behaviours":0"#));
        eprintln!("head y @F60: default {def} on {on} off {off}");
        assert_eq!(def.to_bits(), on.to_bits(), "default = 1");
        assert!(on - off > 0.05, "active ragdoll keeps the head higher: {on} vs {off}");
        assert_eq!(off.to_bits(), head_y(r#""behaviours":0"#).to_bits(), "deterministic");
    }
}
