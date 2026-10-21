//! `table scene.ens [last_frame]`: passive vs behaviours: head height / start, COM-over-feet, and (behaviours) controller diagnostics
//! every 30 sim frames (1/30 s steps; NB the .ens timeline runs at positionBufferFrameRate, 60 fps in Balcony: ens frame = 2 x sim frame).
use endo_sim::Sim;
use hexe_geo::endorphin::{Document, Scene, read_endorphin};
/// Per-character metric row of one mark: `(name, head height m, COM-over-feet m)`.
type Row = Vec<(String, f32, f32)>;
fn metrics(s: &Scene, sim: &Sim) -> Row {
    let pose = sim.pose();
    let mut out = vec![];
    for c in &s.characters {
        let Some(cp) = pose.iter().find(|p| p.name == c.name) else { continue };
        let idx = |n: &str| c.joints.iter().position(|j| j.name == n).unwrap();
        let head = cp.world[idx("UpperNeckJoint")].p[1];
        let (la, ra) = (cp.world[idx("LeftAnkleJoint")].p, cp.world[idx("RightAnkleJoint")].p);
        let n = cp.world.len() as f32;
        let com = cp.world.iter().fold([0.0f32; 3], |a, x| [a[0] + x.p[0] / n, 0.0, a[2] + x.p[2] / n]);
        let (fx, fz) = ((la[0] + ra[0]) / 2.0, (la[2] + ra[2]) / 2.0);
        out.push((c.name.clone(), head, ((com[0] - fx).powi(2) + (com[2] - fz).powi(2)).sqrt()));
    }
    out
}
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let Document::Scene(s) = read_endorphin(&std::fs::read(&a[1]).unwrap()).unwrap() else { return };
    let last: usize = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(300);
    let marks: Vec<usize> = std::iter::once(1).chain((30..=last).step_by(30)).collect();
    let mut rows: Vec<(String, Vec<Row>)> = vec![];
    let mut diag = String::new();
    for (label, passive) in [("passive", true), ("behaviours", false)] {
        let mut sim = if passive { Sim::from_scene(&s) } else { Sim::from_scene_play(&s) };
        let mut v = vec![];
        for f in 1..=last {
            if f > 1 {
                sim.step(1.0 / 30.0);
            }
            if marks.contains(&f) {
                let m = metrics(&s, &sim);
                if !passive {
                    for (n, h, _) in &m {
                        if let Some(b) = sim.behaviour_state(n) {
                            let h0 = v
                                .first()
                                .and_then(|r: &Vec<(String, f32, f32)>| r.iter().find(|x| &x.0 == n))
                                .map_or(*h, |x| x.1);
                            diag.push_str(&format!(
                                "| {n} | {f} | {:.0} | {:.0}% | {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {} |\n",
                                f as f32 * 2.0,
                                h / h0 * 100.0,
                                b.standing,
                                b.hold,
                                b.cop_demand,
                                b.fall,
                                b.com_offset,
                                b.tilt,
                                b.active.join("+")
                            ));
                        }
                    }
                }
                v.push(m);
            }
        }
        rows.push((label.into(), v));
    }
    println!("| char | mode | f | head m | head/start | COM-feet m |");
    for ci in 0..s.characters.len() {
        for (label, v) in &rows {
            let h0 = v[0][ci].1;
            for (mi, f) in marks.iter().enumerate() {
                let (n, h, d) = &v[mi][ci];
                println!("| {n} | {label} | {f} | {h:.2} | {:.0}% | {d:.2} |", h / h0 * 100.0);
            }
        }
    }
    println!(
        "\n| char | f | ens frame | head/start | standing | hold | COP demand/budget | fall | bal err m | tilt rad | active |"
    );
    print!("{diag}");
}
