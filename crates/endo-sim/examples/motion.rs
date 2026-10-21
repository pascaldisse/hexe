//! `motion scene.ens [frames] [fps]`: per character, how far every joint moved from frame 1 (max |dP| and the joint that moved
//! most) at the sampled frames. Proof that the file PLAYS (rest pose = all zeros).
use endo_sim::Sim;
use hexe_geo::endorphin::{Document, read_endorphin};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let Some(path) = a.get(1) else { return };
    let Ok(bytes) = std::fs::read(path) else { return };
    let Ok(Document::Scene(s)) = read_endorphin(&bytes) else { return };
    let frames: usize = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(300);
    let fps: f32 = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(30.0);
    let passive = a.get(4).is_some_and(|v| v == "passive");
    let mut sim = if passive { Sim::from_scene(&s) } else { Sim::from_scene_play(&s) };
    let start = sim.pose();
    let marks = [1usize, 30, 60, 90, 150, 300];
    for f in 1..=frames {
        if f > 1 {
            sim.step(1.0 / fps);
        }
        if !marks.contains(&f) {
            continue;
        }
        let pose = sim.pose();
        print!("f{f:<4}");
        for (ci, c) in s.characters.iter().enumerate() {
            let (Some(p0), Some(p1)) = (start.get(ci), pose.get(ci)) else { continue };
            let mut best = (0.0f32, "");
            let mut sum = 0.0f32;
            for (ji, j) in c.joints.iter().enumerate() {
                let (Some(a), Some(b)) = (p0.world.get(ji), p1.world.get(ji)) else { continue };
                let d = ((a.p[0] - b.p[0]).powi(2) + (a.p[1] - b.p[1]).powi(2) + (a.p[2] - b.p[2]).powi(2)).sqrt();
                sum += d;
                if d > best.0 {
                    best = (d, j.name.as_str());
                }
            }
            let n = c.joints.len().max(1) as f32;
            print!(" | {} mean={:.3} max={:.3} @{}", c.name, sum / n, best.0, best.1);
        }
        println!();
    }
}
