use super::*;
use behaviour::Gains;
use hexe_geo::endorphin::{Document, parse_xml, read_endorphin};
use std::path::{Path, PathBuf};

fn resources() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let p = std::env::var_os("ENDORPHIN_RESOURCES").map(PathBuf::from).unwrap_or_else(|| {
        Path::new(&home).join("Library/Application Support/CrossOver/Bottles/endorphin/drive_c/Program Files (x86)/NaturalMotion/endorphin 2.5.2/Resources")
    });
    p.is_dir().then_some(p)
}

fn balcony() -> Option<Scene> {
    let bytes = std::fs::read(resources()?.join("Scenes/Sample Scenes/Balcony.ens")).ok()?;
    match read_endorphin(&bytes).ok()? {
        Document::Scene(s) => Some(*s),
        _ => None,
    }
}

fn root_ys(sim: &Sim) -> Vec<(String, f32)> {
    sim.dynamic_characters().into_iter().filter_map(|n| Some((n.to_string(), sim.root_y(n)?))).collect()
}

/// Two-joint character (root + child with limits) 2 m above a default ground.
const MINI: &str = r#"<Scene useDefaultGround="true"><Characters><Character name="c">
<Skeleton name="root" metaBody="pelvis"><TransformMatrix><Vp x="0" y="2" z="0"/></TransformMatrix>
<ChildJoints><SkeletalJoint name="knee" metaBody="shin"><TransformMatrix><Vp x="0" y="0" z="0.4"/></TransformMatrix>
<SoftLimits twistSize="0.5" swing1Size="1.0" swing2Size="1.0"/><HardLimits twistSize="0.6" swing1Size="1.2" swing2Size="1.2"/>
</SkeletalJoint></ChildJoints></Skeleton>
<MetaBodies>
<MetaBody name="pelvis"><CollisionSurface><Geometry type="Sphyl" param0="0.1" param1="0.3"/></CollisionSurface><InertialBody><Geometry type="Sphere" param0="0.1"/><Inertia density="1000"/></InertialBody></MetaBody>
<MetaBody name="shin"><CollisionSurface><Geometry type="Cuboid" param0="0.1" param1="0.1" param2="0.4"/></CollisionSurface></MetaBody>
</MetaBodies></Character></Characters></Scene>"#;

#[test]
fn mini_scene_falls_onto_the_ground_and_stays_finite() {
    let n = parse_xml(MINI).unwrap();
    let s = Scene::from_node(&n);
    let mut sim = Sim::from_scene(&s);
    assert_eq!(sim.counts().0, 2, "two bodies");
    assert_eq!(sim.counts().1, 1, "one joint");
    let y0 = sim.root_y("c").unwrap();
    assert!((y0 - 2.0).abs() < 1e-4, "{y0}");
    for _ in 0..90 {
        sim.step(1.0 / 30.0);
    }
    let y = sim.root_y("c").unwrap();
    assert!(y < 0.5 && y > -0.01, "rests on the ground: {y}");
    for cp in sim.pose() {
        assert!(cp.world.iter().all(|x| [x.i, x.j, x.k, x.p].iter().flatten().all(|v| v.is_finite())));
    }
}

#[test]
fn bbox_fallback_is_a_capsule_for_unknown_geometry() {
    let g = Geometry {
        kind: "File".into(),
        params: [0.0; 4],
        file: Some("x.mesh".into()),
        raw: parse_xml("<Geometry/>").unwrap(),
    };
    assert!(shape_of(&g).volume > 0.0);
    let c = bbox_capsule([0.1, 0.2, 0.5]);
    assert!((c.volume - (std::f32::consts::PI * 0.01 * 0.8 + 4.0 / 3.0 * std::f32::consts::PI * 0.001)).abs() < 1e-5);
}

/// Balcony: character01 stands on the balcony slab (root y 3.85 m, slab top 3.0 m); 02..04 are seated (root ~0.55 m, supported by
/// the set). Passive ragdoll: the standing one collapses (monotone descent to touchdown, then settles within 5 cm), the seated
/// ones stay put-ish; nothing is NaN.
#[test]
fn balcony_standing_characters_fall_monotonically_without_nan() {
    let Some(scene) = balcony() else { return };
    let mut sim = Sim::from_scene(&scene);
    let start = root_ys(&sim);
    assert_eq!(start.len(), 4, "four dynamic characters");
    let mut hist: Vec<Vec<(String, f32)>> = vec![start.clone()];
    for _ in 0..90 {
        sim.step(1.0 / 30.0);
        for cp in &sim.pose() {
            for x in &cp.world {
                assert!([x.i, x.j, x.k, x.p].iter().flatten().all(|v| v.is_finite()), "NaN in {}", cp.name);
            }
        }
        hist.push(root_ys(&sim));
    }
    let mut standing = 0;
    for (i, (name, y0)) in start.iter().enumerate() {
        let ys: Vec<f32> = hist.iter().map(|h| h[i].1).collect();
        if *y0 > 1.0 {
            standing += 1;
            let low = (0..ys.len()).min_by(|&a, &b| ys[a].total_cmp(&ys[b])).unwrap();
            assert!(low > 3 && low < 60, "{name}: touchdown at frame {low}");
            assert!(
                ys.windows(2).take(low).all(|w| w[1] <= w[0]),
                "{name}: y not monotone down to touchdown: {:?}",
                &ys[..=low]
            );
            assert!(ys[0] - ys[low] > 0.5, "{name}: dropped only {}", ys[0] - ys[low]);
            assert!(ys[low..].iter().all(|y| *y - ys[low] < 0.05), "{name}: bounces back up");
        } else {
            assert!(ys.iter().all(|y| (y - y0).abs() < 0.3), "{name}: seated character left its seat: {ys:?}");
        }
    }
    assert_eq!(standing, 1, "character01 only");
}

fn catch_scene() -> Option<Scene> {
    let bytes = std::fs::read(resources()?.join("Scenes/Behaviour Scenes/Catch fall2.0.ens")).ok()?;
    match read_endorphin(&bytes).ok()? {
        Document::Scene(s) => Some(*s),
        _ => None,
    }
}

const PASSIVE: SimOptions = SimOptions { behaviours: false, forces: false, secondary: false };
const FORCES_ONLY: SimOptions = SimOptions { behaviours: false, forces: true, secondary: false };
const ACTIVE: SimOptions = SimOptions { behaviours: true, forces: true, secondary: false };
const PLAY: SimOptions = SimOptions { behaviours: true, forces: true, secondary: true };

fn bits(sim: &Sim) -> Vec<u32> {
    sim.pose()
        .iter()
        .flat_map(|c| c.world.iter().flat_map(|x| [x.i, x.j, x.k, x.p].into_iter().flatten().map(f32::to_bits)))
        .collect()
}

/// Catch fall scene numbers measured by `run`, printed so a failing bound shows the actual values.
#[derive(Debug, Default, Clone)]
struct Run {
    /// first frame (30 fps) a wrist / the head is below 12 cm, per character
    hand_t: [Option<usize>; 2],
    head_t: [Option<usize>; 2],
    /// mean head height (m) over frames 30..=n
    head_mean: [f32; 2],
}

fn run(scene: &Scene, opt: SimOptions, n: usize) -> Run {
    let mut sim = Sim::from_scene_with(scene, opt);
    let mut r = Run::default();
    let mut cnt = 0.0;
    for f in 0..=n {
        for (i, c) in ["character01", "character02"].iter().enumerate() {
            let p = |j: &str| sim.joint_pos(c, j).unwrap();
            let hand = p("LeftWristJoint")[1].min(p("RightWristJoint")[1]);
            let head = p("UpperNeckJoint")[1];
            if hand < 0.12 && r.hand_t[i].is_none() {
                r.hand_t[i] = Some(f);
            }
            if head < 0.12 && r.head_t[i].is_none() {
                r.head_t[i] = Some(f);
            }
            if f >= 30 {
                r.head_mean[i] += head;
            }
        }
        if f >= 30 {
            cnt += 1.0;
        }
        sim.step(1.0 / 30.0);
    }
    r.head_mean = r.head_mean.map(|v| v / cnt);
    r
}

#[test]
fn catch_fall_instances_and_calibration_come_from_the_files() {
    let Some(scene) = catch_scene() else { return };
    let sim = Sim::from_scene_with(&scene, ACTIVE);
    assert_eq!(sim.behaviour_counts(), (2, 2), "two CatchFall controllers, two force events");
    assert_eq!(Sim::from_scene(&scene).behaviour_counts(), (0, 0));
    let inst = scene.behaviour_instances();
    assert_eq!(inst[0].f32("ArmsStrength", 0.0), 0.7);
    // `Normal` set of the character: shoulder 2000 / 500 / 1000, elbow 1000 / 500 / 1000, neck head 1000 / 500 / 1000
    let g = behaviour::calibration_gains(&scene.characters[0], behaviour::CALIBRATION);
    assert_eq!(g.get("LeftShoulderJoint"), Some(&Gains { stiffness: 2000.0, damping: 500.0, force_limit: 1000.0 }));
    assert_eq!(g.get("RightElbowJoint"), Some(&Gains { stiffness: 1000.0, damping: 500.0, force_limit: 1000.0 }));
    assert_eq!(g.get("UpperNeckJoint").map(|g| g.stiffness), Some(1000.0));
    assert!(behaviour::calibration_gains(&scene.characters[0], "NoSuchSet").is_empty());
}

/// The drive (and only the drive) changes the fall: same scene, same pushes, with vs without the CatchFall controllers.
/// Measured (30 fps, frames 0..=120): hands touch ground at f16-17 in every run (they hang at shoulder height, so this is
/// NOT the discriminator); the head touches (< 12 cm) at f25/f33 with only the pushes and at f72/f42 with the drive, and stays
/// 3x higher on average.
#[test]
fn catch_fall_drive_keeps_the_head_off_the_ground_vs_forces_only() {
    let Some(scene) = catch_scene() else { return };
    let (p, f, a) = (run(&scene, PASSIVE, 120), run(&scene, FORCES_ONLY, 120), run(&scene, ACTIVE, 120));
    eprintln!("passive     {p:?}\nforces only {f:?}\nactive      {a:?}");
    for i in 0..2 {
        let (fh, ah) = (f.head_t[i].unwrap(), a.head_t[i].unwrap_or(usize::MAX));
        assert!(ah >= fh + 5, "char {i}: head down at f{ah} (active) vs f{fh} (forces only)");
        assert!(
            a.head_mean[i] > 1.8 * f.head_mean[i],
            "char {i}: mean head height {} vs {}",
            a.head_mean[i],
            f.head_mean[i]
        );
        // hands arrive before the head in every run, and the margin grows with the drive
        for r in [&p, &f, &a] {
            assert!(r.hand_t[i].unwrap() + 8 <= r.head_t[i].unwrap_or(usize::MAX), "char {i}: hand before head: {r:?}");
        }
        assert!(a.head_t[i].unwrap_or(usize::MAX) - a.hand_t[i].unwrap() > f.head_t[i].unwrap() - f.hand_t[i].unwrap());
    }
}

#[test]
fn catch_fall_is_deterministic_and_finite() {
    let Some(scene) = catch_scene() else { return };
    let go = || {
        let mut sim = Sim::from_scene_with(&scene, ACTIVE);
        for _ in 0..90 {
            sim.step(1.0 / 30.0);
        }
        bits(&sim)
    };
    let (a, b) = (go(), go());
    assert_eq!(a, b, "bit-identical across two runs");
    assert!(a.iter().all(|u| f32::from_bits(*u).is_finite()));
}

/// Strength 0 switches the motors off: the sim is bit-identical to the pushes-only sim, so the divergence above is the drive.
#[test]
fn arms_strength_zero_equals_no_drive() {
    let Some(mut scene) = catch_scene() else { return };
    let sb = scene.raw.children.iter_mut().find(|c| c.tag == "SecondaryBehaviours").unwrap();
    for v in sb
        .children
        .iter_mut()
        .flat_map(|b| b.children.iter_mut())
        .filter(|n| n.tag == "Parameters")
        .flat_map(|p| p.children.iter_mut())
    {
        if v.attr("fk") == Some("ArmsStrength") {
            for (k, val) in v.attrs.iter_mut() {
                if k == "ja" {
                    *val = "0".into();
                }
            }
        }
    }
    let go = |s: &Scene, o| {
        let mut sim = Sim::from_scene_with(s, o);
        for _ in 0..60 {
            sim.step(1.0 / 30.0);
        }
        bits(&sim)
    };
    assert_eq!(go(&scene, ACTIVE), go(&scene, FORCES_ONLY));
}

#[test]
fn behaviours_without_the_arm_chain_or_character_are_ignored() {
    // MINI has no arms; a CatchFall on it, on a missing character, and an unimplemented behaviour must build nothing
    let xml = MINI.replace(
        "</Characters>",
        "</Characters><SecondaryBehaviours>\
<SecondaryBehaviour behaviourName=\"Bob2: CatchFall 2.0\" character=\"c\" startFrameNumber=\"0\" endFrameNumber=\"9\"/>\
<SecondaryBehaviour behaviourName=\"Bob2: CatchFall 2.0\" character=\"nobody\" startFrameNumber=\"0\" endFrameNumber=\"9\"/>\
<SecondaryBehaviour behaviourName=\"Bob2: Stagger 2.0\" character=\"c\" startFrameNumber=\"0\" endFrameNumber=\"9\"/>\
</SecondaryBehaviours>",
    );
    let s = Scene::from_node(&parse_xml(&xml).unwrap());
    assert_eq!(s.behaviour_instances().len(), 3);
    let mut sim = Sim::from_scene_active(&s);
    assert_eq!(sim.behaviour_counts(), (0, 0));
    for _ in 0..30 {
        sim.step(1.0 / 30.0);
    }
    assert!(sim.root_y("c").unwrap().is_finite());
}

#[test]
fn force_event_maps_direction_and_scale() {
    use hexe_geo::endorphin_behaviour::ForceEvent;
    let n = parse_xml("<ForceEvent character=\"a\" brelativeAcceleration=\"3e4ccccd\"><TransformMatrix><Vk bx=\"c1400000\" by=\"0\" bz=\"0\"/></TransformMatrix></ForceEvent>").unwrap();
    let a = behaviour::force_accel(&ForceEvent::from_node(&n), 10.0);
    assert!((a.x + 24.0).abs() < 1e-4 && a.y.abs() < 1e-6 && a.z.abs() < 1e-6, "{a:?}");
    let none = parse_xml("<ForceEvent/>").unwrap();
    assert_eq!(behaviour::force_accel(&ForceEvent::from_node(&none), 10.0), Vec3::ZERO);
}
