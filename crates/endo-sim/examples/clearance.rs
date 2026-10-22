//! `clearance scene.ens [mode 0|1|2] [frames]`: per character, deepest penetration into the static world (`Sim::env_penetration`) over F1..=frames
//! at 30 fps, plus the frame and joint where it happened (negative = sunk in).
use endo_sim::{Sim, SimOptions};
use hexe_geo::endorphin::{Document, read_endorphin};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    let path = a.get(1).ok_or("usage: clearance scene.ens [mode] [frames]")?;
    let Document::Scene(s) = read_endorphin(&std::fs::read(path)?)? else { return Err("not a scene".into()) };
    if std::env::var("DUMP").is_ok() {
        dump(&s);
        return Ok(());
    }
    let mode: i32 = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(2);
    let frames: usize = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(300);
    // h54 A/B knobs: PROPS_FIXED=1 (env props fixed), FORCE_WELDED=0 (pre-h54 force mass)
    let on = |k: &str, d: bool| std::env::var(k).map(|v| v != "0").unwrap_or(d);
    let opt = SimOptions { behaviours: mode >= 1, forces: mode >= 1, secondary: mode >= 2 };
    let mut sim = Sim::from_scene_props(&s, opt, opt.secondary && !on("PROPS_FIXED", false));
    sim.force_welded = on("FORCE_WELDED", true);
    // h56 A/B: HOLD_GAMMA=<erp> / HOLD_EPS=<cfm> (Lock / JoinBodies rows)
    let num = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok());
    if let Some(g) = num("HOLD_GAMMA") {
        sim.hold_gamma = g;
    }
    if let Some(e) = num("HOLD_EPS") {
        sim.hold_eps = e;
    }
    // FORCES=1 -> per <ForceEvent> target: accel, masses, impulse N s, measured dv of the body and of its pushed group
    if std::env::var("FORCES").is_ok() {
        let last = sim.force_diag().iter().map(|f| f.1).max().unwrap_or(0);
        let mut at: Vec<Option<([f32; 3], [f32; 3])>> = vec![None; sim.force_diag().len()];
        let sub = |a: [f32; 3], b: [f32; 3]| [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let len = |a: [f32; 3]| (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        for k in 0..=last {
            for (i, f) in sim.force_diag().into_iter().enumerate() {
                if k == f.0 {
                    at[i] = Some((f.5, f.6));
                }
                if k == f.1 {
                    // h55: f.2 = the one-shot dv the exe formula applied at F{start}
                    let (v0, g0) = at[i].unwrap_or((f.5, f.6));
                    println!(
                        "force#{i} F{}..F{} applied dv {:.4} m/s ({:.3} {:.3} {:.3}) body {:.3} kg measured body dv {:.4} group dv {:.4} m/s",
                        f.0, f.1, len(f.2), f.2[0], f.2[1], f.2[2], f.3, len(sub(v0, f.5)), len(sub(g0, f.6))
                    );
                }
            }
            if k < last {
                sim.step(1.0 / 30.0);
            }
        }
        return Ok(());
    }
    if let Ok(w) = std::env::var("WHO") {
        let (who, at) = w.split_once(':').unwrap_or((&w, "1"));
        let at: usize = at.parse().unwrap_or(1);
        for _ in 1..at {
            sim.step(1.0 / 30.0);
        }
        for (n, d, j) in sim.env_penetration() {
            if n == who {
                let c = s.characters.iter().find(|c| c.name == n).ok_or("?")?;
                let jn = &c.joints[j].name;
                println!("{n} F{at} {jn} d={d:.3} pos={:?} root={:?}", sim.joint_pos(&n, jn), sim.root_y(&n));
            }
        }
        return Ok(());
    }
    // PAIRS=name:frame -> the overlapping (collider, static) pairs at that frame and the solver's view of each
    if let Ok(w) = std::env::var("PAIRS") {
        let (who, at) = w.split_once(':').unwrap_or((&w, "1"));
        let at: usize = at.parse().unwrap_or(1);
        for _ in 1..at {
            sim.step(1.0 / 30.0);
        }
        sim.env_pairs(who).iter().for_each(|l| println!("F{at} {l}"));
        return Ok(());
    }
    // PROPS=ch.Body,ch.Body[:stride] -> the parsed Lock/JoinBodies events, then per `stride` frames each body's position
    if let Ok(w) = std::env::var("PROPS") {
        let (list, stride) = w.split_once(':').unwrap_or((&w, "1"));
        let stride: usize = stride.parse().unwrap_or(1).max(1);
        for h in sim.holds() {
            let ok = h.objects.iter().filter(|o| o.1.is_some()).count();
            let objs: Vec<&str> = h.objects.iter().map(|o| o.0.as_str()).collect();
            println!(
                "hold {:?} {} F{}..F{} strength {:?} objects {}/{} {:?}",
                h.name,
                if h.lock { "Lock" } else { "JoinBodies" },
                h.start,
                h.end,
                h.strength,
                ok,
                h.objects.len(),
                objs
            );
        }
        let bodies: Vec<(&str, &str)> = list.split(',').filter_map(|b| b.split_once('.')).collect();
        for f in 1..=frames {
            if f > 1 {
                sim.step(1.0 / 30.0);
            }
            if f % stride == 0 || f == 1 {
                let row: Vec<String> = bodies
                    .iter()
                    .map(|(c, b)| match sim.body_pos(&s, c, b) {
                        Some((p, d)) => format!("{c}.{b} {:.4} {:.4} {:.4} {}", p[0], p[1], p[2], if d { "dyn" } else { "fixed" }),
                        None => format!("{c}.{b} ?"),
                    })
                    .collect();
                println!("F{f}\t{}\tjoints {}", row.join("\t"), sim.holds().iter().map(|h| h.joint_count()).sum::<usize>());
            }
        }
        return Ok(());
    }
    if let Ok(v) = std::env::var("SELF") {
        sim.set_self_collision(v != "0");
    }
    let mut worst: Vec<(String, f32, usize, usize)> = Vec::new();
    // per character: (sum env depth, deepest self overlap, frame, joint a, joint b, sum self depth)
    let mut acc: Vec<(String, f32, f32, usize, usize, usize, f32)> = Vec::new();
    for f in 1..=frames {
        if f > 1 {
            sim.step(1.0 / 30.0);
        }
        let rows = sim.env_penetration();
        for ((name, e, _), (_, d, ja, jb)) in rows.iter().zip(sim.self_penetration()) {
            match acc.iter_mut().find(|a| a.0 == *name) {
                Some(a) => {
                    a.1 += e;
                    a.6 += d;
                    if d < a.2 {
                        (a.2, a.3, a.4, a.5) = (d, f, ja, jb);
                    }
                }
                None => acc.push((name.clone(), *e, d, f, ja, jb, d)),
            }
        }
        for (name, d, j) in rows {
            match worst.iter_mut().find(|w| w.0 == name) {
                Some(w) if d < w.1 => *w = (name, d, f, j),
                Some(_) => {}
                None => worst.push((name, d, f, j)),
            }
        }
    }
    for (name, d, f, j) in worst {
        let jn = s
            .characters
            .iter()
            .find(|c| c.name == name.as_str())
            .and_then(|c| c.joints.get(j))
            .map_or("?", |j| &j.name);
        let jname = |j: usize| {
            s.characters.iter().find(|c| c.name == name).and_then(|c| c.joints.get(j)).map_or("?", |j| &j.name)
        };
        let a = acc.iter().find(|a| a.0 == name).cloned().unwrap_or_default();
        let n = frames as f32;
        println!(
            "{name}\tenv min {d:.3} F{f} {jn} mean {:.4}\tself min {:.3} F{} {}-{} mean {:.4}",
            a.1 / n,
            a.2,
            a.3,
            jname(a.4),
            jname(a.5),
            a.6 / n
        );
    }
    Ok(())
}
#[allow(dead_code)]
fn dump(s: &hexe_geo::endorphin::Scene) {
    for c in &s.characters {
        for j in &c.joints {
            if let Some(b) = &j.body {
                println!("{} {} -> {} virt={:?}", c.name, j.name, b, j.raw.attr("isVirtualMuscle"));
            }
        }
    }
}
