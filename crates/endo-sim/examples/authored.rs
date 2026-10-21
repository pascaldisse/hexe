//! `authored scene.ens`: what the file authors (what Endorphin plays) vs what this crate simulates.
use hexe_geo::endorphin::{Document, read_endorphin};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let Some(path) = args.get(1) else { return };
    let Ok(bytes) = std::fs::read(path) else { return };
    let Ok(Document::Scene(s)) = read_endorphin(&bytes) else { return };
    println!("# {path}");
    let fps = s.raw.child("Engine").and_then(|e| e.f32("positionBufferFrameRate"));
    println!(
        "gravity {:?} ground {} chars {} env {} rate {:?} duration {:.2}s",
        s.gravity,
        s.use_default_ground,
        s.characters.len(),
        s.environment.is_some(),
        fps,
        s.duration()
    );
    for c in &s.characters {
        println!(
            "char {} joints {} bodies {} anim {} poses {}",
            c.name,
            c.joints.len(),
            c.bodies.len(),
            c.raw.child("Animation").map_or(0, |a| a.count()),
            c.raw.child("Poses").map_or(0, |a| a.count())
        );
    }
    let known = ["catchfall", "stagger", "windmill", "legsstraighten", "legs straighten"];
    for b in s.behaviour_instances() {
        let low = b.behaviour.to_ascii_lowercase();
        let tag = if known.iter().any(|k| low.contains(k)) { "SIM" } else { "IGNORED" };
        let ps: Vec<String> = b
            .params
            .iter()
            .filter(|p| p.value.is_some())
            .map(|p| format!("{}={}", p.key, p.value.clone().unwrap_or_default()))
            .collect();
        println!("beh [{tag}] {} on {} f{}..{} | {}", b.behaviour, b.character, b.start, b.end, ps.join(" "));
    }
    for e in s.force_events() {
        println!(
            "force {:?} f{}..{} relAcc {} Vk {:?} at {:?}",
            e.targets, e.start, e.end, e.relative_acceleration, e.xf.k, e.xf.p
        );
    }
    if let Some(ev) = s.raw.child("Events") {
        for e in &ev.children {
            let attrs: Vec<String> = e.attrs.iter().map(|(k, v)| format!("{k}={v}")).collect();
            let kids: Vec<String> = e.children.iter().map(|k| format!("{}:{}", k.tag, k.children.len())).collect();
            println!("event {} [{}] kids [{}]", e.tag, attrs.join(" "), kids.join(" "));
        }
    }
    // tag census of the whole scene tree (what the file carries that we may not read)
    let mut census: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    fn walk(n: &hexe_geo::endorphin::Node, c: &mut std::collections::BTreeMap<String, usize>) {
        *c.entry(n.tag.clone()).or_default() += 1;
        for k in &n.children {
            walk(k, c);
        }
    }
    walk(&s.raw, &mut census);
    let interesting: Vec<String> = census
        .iter()
        .filter(|(t, _)| {
            let l = t.to_ascii_lowercase();
            l.contains("anim")
                || l.contains("pose")
                || l.contains("channel")
                || l.contains("event")
                || l.contains("key")
                || l.contains("constraint")
                || l.contains("rig")
        })
        .map(|(t, n)| format!("{t}={n}"))
        .collect();
    println!("census {}", interesting.join(" "));
}
