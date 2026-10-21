//! `corpus scene.ens [frames]`: play one scene through the authored-event path (`from_scene_play`)
//! and print ONE tab-separated result row for the h47 corpus table:
//! `name  chars  poseEv  animEv(chan)  incarn  behInst  forceEv  frames  rootEnd  headEnd  contacts  nan  ground  minY  verdict`.
//! Exit code 0 = played clean, 2 = NaN / explosion / sank through the scene ground.
use endo_sim::Sim;
use hexe_geo::endorphin::{Document, read_endorphin};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let Some(path) = a.get(1) else { eprintln!("usage: corpus scene.ens [frames]"); std::process::exit(64) };
    let frames: usize = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(300);
    let name = std::path::Path::new(path).file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => { println!("{name}\t-\t-\t-\t-\t-\t-\t0\t-\t-\t-\t-\tREAD_ERR {e}"); std::process::exit(2) }
    };
    let doc = match read_endorphin(&bytes) {
        Ok(d) => d,
        Err(e) => { println!("{name}\t-\t-\t-\t-\t-\t-\t0\t-\t-\t-\t-\tPARSE_ERR {e}"); std::process::exit(2) }
    };
    let Document::Scene(s) = doc else {
        println!("{name}\t-\t-\t-\t-\t-\t-\t0\t-\t-\t-\t-\tNOT_A_SCENE");
        return;
    };
    let pose_ev = s.pose_events();
    let anim_ev = s.anim_events();
    let chans: usize = anim_ev.iter().map(|e| e.channels.len()).sum();
    let incs = s.incarnations();
    let beh = s.behaviour_instances().len();
    let forces = s.force_events().len();

    // lowest body-carrying joint of the simulated characters (`VirtualMuscle` joints are rig markers inside a body)
    let low = |sim: &Sim| {
        s.characters
            .iter()
            .flat_map(|c| {
                let p = sim.pose().into_iter().find(|p| p.name == c.name);
                c.joints
                    .iter()
                    .enumerate()
                    .filter(|(_, j)| j.body.is_some() && !j.name.contains("VirtualMuscle"))
                    .filter_map(move |(i, _)| p.as_ref().and_then(|p| p.world.get(i)).map(|x| x.p[1]))
                    .collect::<Vec<f32>>()
            })
            .fold(f32::INFINITY, f32::min)
    };
    let mut sim = Sim::from_scene_play(&s);
    let mut nan = 0usize;
    let mut maxabs = 0.0f32;
    let mut prev_low = low(&sim);
    for f in 1..=frames {
        if f > 1 {
            if f == frames {
                prev_low = low(&sim);
            }
            sim.step(1.0 / 30.0);
        }
        for cp in sim.pose() {
            for x in &cp.world {
                for v in x.p.iter().chain(x.i.iter()).chain(x.j.iter()).chain(x.k.iter()) {
                    if !v.is_finite() {
                        nan += 1;
                    } else {
                        maxabs = maxabs.max(v.abs());
                    }
                }
            }
        }
        if nan > 0 {
            println!(
                "{name}\t{}\t{}\t{}({chans})\t{}\t{beh}\t{forces}\t{f}\t-\t-\t-\t{nan}\t{}\t-\tNAN@f{f}",
                s.characters.len(),
                pose_ev.len(),
                anim_ev.len(),
                incs.len(),
                s.use_default_ground
            );
            std::process::exit(2);
        }
    }
    let pose = sim.pose();
    let fmt = |v: [f32; 3]| format!("({:.2},{:.2},{:.2})", v[0], v[1], v[2]);
    let c0 = s.characters.first();
    let root = c0
        .and_then(|c| pose.iter().find(|p| p.name == c.name).and_then(|p| c.joints.iter().position(|j| j.name == "root").map(|i| p.world[i].p)))
        .map(fmt)
        .unwrap_or("-".into());
    let head = c0
        .and_then(|c| {
            pose.iter()
                .find(|p| p.name == c.name)
                .and_then(|p| c.joints.iter().position(|j| j.name == "UpperNeckJoint").map(|i| p.world[i].p[1]))
        })
        .map(|y| format!("{y:.2}"))
        .unwrap_or("-".into());
    let contacts: usize = s.characters.iter().map(|c| sim.contacts_of(&c.name).len()).sum();
    // lowest point of any simulated character at the final frame: with a scene ground it must stay on top of it
    let min_y = low(&sim);
    // sank = below the scene ground AND still sinking (an authored kinematic object may rest below it)
    let sank = s.use_default_ground && min_y < -0.25 && min_y < prev_low - 0.002;
    let verdict = if maxabs > 1.0e3 {
        format!("EXPLODED max={maxabs:.0}")
    } else if sank {
        format!("SANK minY={min_y:.2}")
    } else {
        "OK".to_string()
    };
    println!(
        "{name}\t{}\t{}\t{}({chans})\t{}\t{beh}\t{forces}\t{frames}\t{root}\t{head}\t{contacts}\t0\t{}\t{min_y:.2}\t{verdict}",
        s.characters.len(),
        pose_ev.len(),
        anim_ev.len(),
        incs.len(),
        s.use_default_ground
    );
    if maxabs > 1.0e3 || sank {
        std::process::exit(2);
    }
}
