//! `cargo run --release -p endo-sim --example probe -- scene.ens [passive] [frames] [fps]`: per-second behaviour metrics of a scene.
use endo_sim::Sim;
use hexe_geo::endorphin::{Document, read_endorphin};
fn rot_angle(a: &hexe_geo::endorphin::Xf, b: &hexe_geo::endorphin::Xf) -> f32 {
    // angle of a^T b (rotation parts)
    let dot = |u: [f32; 3], v: [f32; 3]| u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    let tr = dot(a.i, b.i) + dot(a.j, b.j) + dot(a.k, b.k);
    ((tr - 1.0) / 2.0).clamp(-1.0, 1.0).acos()
}
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&a[1]).unwrap();
    let Document::Scene(s) = read_endorphin(&bytes).unwrap() else { return };
    let passive = a.get(2).is_some_and(|v| v == "passive");
    let frames: usize = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(150);
    let fps: f32 = a.get(4).and_then(|v| v.parse().ok()).unwrap_or(30.0);
    let mut sim = if passive { Sim::from_scene(&s) } else { Sim::from_scene_play(&s) };
    let t0 = std::time::Instant::now();
    let start = sim.pose();
    for f in 1..=frames {
        if f > 1 {
            sim.step(1.0 / fps);
        }
        let dense = std::env::var("DENSE").is_ok()
            && f % 3 == 0
            && f <= std::env::var("DENSE").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(24);
        if std::env::var("ERR").ok().and_then(|v| v.parse::<usize>().ok()) == Some(f) {
            let pose = sim.pose();
            let c = &s.characters[0];
            let (p0, p1) = (&start[0], &pose[0]);
            let mut errs: Vec<(f32, String)> = Vec::new();
            for (i, j) in c.joints.iter().enumerate() {
                let Some(pa) = j.parent else { continue };
                let rel = |p: &endo_sim::CharPose| {
                    let a = p.world[pa];
                    let b = p.world[i];
                    let inv = |x: &hexe_geo::endorphin::Xf| hexe_geo::endorphin::Xf {
                        i: [x.i[0], x.j[0], x.k[0]],
                        j: [x.i[1], x.j[1], x.k[1]],
                        k: [x.i[2], x.j[2], x.k[2]],
                        p: [0.0; 3],
                    };
                    inv(&a).mul(&b)
                };
                errs.push((rot_angle(&rel(p0), &rel(p1)), j.name.clone()));
            }
            errs.sort_by(|a, b| b.0.total_cmp(&a.0));
            println!("ERR f{f}: {:?}", errs.iter().take(10).map(|(e, n)| format!("{n}={e:.2}")).collect::<Vec<_>>());
        }
        if dense {
            let pose = sim.pose();
            let c = &s.characters[0];
            let cp = &pose[0];
            let st = sim.behaviour_state(&c.name);
            let g = |n: &str| {
                let i = c.joints.iter().position(|j| j.name == n).unwrap();
                cp.world[i].p
            };
            println!(
                "f{f} pel={:?} lank={:?} rank={:?} lknee={:?} head={:?} {:?}",
                g("root").map(|v| (v * 100.0).round() / 100.0),
                g("LeftAnkleJoint").map(|v| (v * 100.0).round() / 100.0),
                g("RightAnkleJoint").map(|v| (v * 100.0).round() / 100.0),
                g("LeftKneeJoint").map(|v| (v * 100.0).round() / 100.0),
                g("UpperNeckJoint").map(|v| (v * 100.0).round() / 100.0),
                st.map(|s| (s.fall, s.tilt, s.com_offset))
            );
        }
        if f == 1 || f % 15 == 0 || f == frames {
            let pose = sim.pose();
            print!("f{f:<4} t={:.2}s ", (f - 1) as f32 / fps);
            for c in &s.characters {
                let Some(cp) = pose.iter().find(|p| p.name == c.name) else { continue };
                let idx = |n: &str| c.joints.iter().position(|j| j.name == n);
                let y = |n: &str| idx(n).map(|i| cp.world[i].p[1]).unwrap_or(f32::NAN);
                let st = sim.behaviour_state(&c.name);
                let maxabs = cp.world.iter().flat_map(|x| x.p).fold(0.0f32, |m, v| m.max(v.abs()));
                print!(
                    "| {} pelvis={:.2} head={:.2} lhand={:.2} rhand={:.2} {} ",
                    &c.name[c.name.len() - 2..],
                    y("root"),
                    y("UpperNeckJoint"),
                    y("LeftWristJoint"),
                    y("RightWristJoint"),
                    st.map(|s| format!(
                        "fall={:.2} tilt={:.2} com={:.2} act={:?}",
                        s.fall, s.tilt, s.com_offset, s.active
                    ))
                    .unwrap_or_default()
                );
                if maxabs > 1e3 {
                    print!("EXPLODED ");
                }
            }
            println!();
        }
    }
    eprintln!("{} frames in {:?}", frames, t0.elapsed());
}
