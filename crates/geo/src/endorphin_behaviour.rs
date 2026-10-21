//! Decoded part of Endorphin behaviours (h46, `oracle/endorphin/NMB.md`): the PLAINTEXT layers only.
//!
//! * `.nmb` parameter interface: root `<ry><wx sv kg ja de gu>` (the network blobs `lo lt vw sd sb` are NOT decoded).
//! * scene instances: `<SecondaryBehaviour behaviourName ...><Parameters><Variable fk kg ja de>` and `<ForceEvent>`.
//!
//! Clean-room from data: tag/attribute roles were inferred by correlating the 47 `.nmb` with the 135 scene instances.
use crate::endorphin::{Node, Scene, Xf};

/// Value type of a parameter (`kg` code).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamKind {
    Float,
    Int,
    Bool,
    /// `$character.Calibrations.Normal`, `background.MetaBodies.Prop0.Position.x`
    StringRef,
    /// `Vector(x, y, z)`
    Vector,
    /// `MetaBodyAndCollisionSelection`, `DesiredPoseEventData`, `Randomizer`
    Selection,
    /// encoded blob (opaque)
    Blob,
    /// object slot without value
    Slot,
    Unknown(u8),
}

impl ParamKind {
    pub fn from_code(c: &str) -> ParamKind {
        match c.trim() {
            "0" => ParamKind::Float,
            "1" => ParamKind::Int,
            "2" => ParamKind::Bool,
            "3" => ParamKind::StringRef,
            "4" => ParamKind::Vector,
            "5" => ParamKind::Selection,
            "6" => ParamKind::Blob,
            "7" => ParamKind::Slot,
            o => ParamKind::Unknown(o.parse().unwrap_or(255)),
        }
    }
}

/// One parameter of a behaviour (interface entry or scene instance value).
#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    /// Scene key (`fk`); empty for `.nmb` entries (they carry a numeric id instead).
    pub key: String,
    /// File-local numeric id (`sv`), `.nmb` only.
    pub id: Option<u32>,
    pub kind: ParamKind,
    /// Raw value text (`ja`); `None` = wired by the network / no default.
    pub value: Option<String>,
    /// UI label (`de`); `_undefined` is dropped.
    pub label: Option<String>,
    /// Developer tooltip (`gu`).
    pub tip: Option<String>,
}

impl Param {
    fn from_node(n: &Node) -> Param {
        let s = |k: &str| n.attr(k).map(str::to_string);
        Param {
            key: s("fk").unwrap_or_default(),
            id: n.attr("sv").and_then(|v| v.parse().ok()),
            kind: ParamKind::from_code(n.attr("kg").unwrap_or("")),
            value: s("ja"),
            label: s("de").filter(|l| l != "_undefined"),
            tip: s("gu"),
        }
    }
    /// Float value (ints and `true`/`false` too: 1.0 / 0.0).
    pub fn as_f32(&self) -> Option<f32> {
        let v = self.value.as_deref()?.trim();
        match v {
            "true" => Some(1.0),
            "false" => Some(0.0),
            _ => v.parse::<f32>().ok().filter(|f| f.is_finite()),
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        self.as_f32().map(|f| f != 0.0)
    }
}

/// A `.nmb`: name + root parameter interface. The network itself stays in `raw`.
#[derive(Clone, Debug, PartialEq)]
pub struct NmbInterface {
    pub name: String,
    pub params: Vec<Param>,
    /// `cc` nodes in the file (network size; not interpreted).
    pub nodes: usize,
    /// Plaintext `wx` constants anywhere in the network (`ja` set): (parameter, tooltip) - attached to unknown nodes.
    pub constants: usize,
}

impl NmbInterface {
    pub fn from_node(root: &Node) -> Option<NmbInterface> {
        if root.tag != "cc" {
            return None;
        }
        let name = root.child("RefNames")?.child("Name")?.text.trim().to_string();
        let params = root.kids("ry").flat_map(|ry| ry.kids("wx")).map(Param::from_node).collect();
        fn walk(n: &Node, nodes: &mut usize, consts: &mut usize) {
            if n.tag == "cc" {
                *nodes += 1;
            }
            if n.tag == "wx" && n.attr("ja").is_some() && n.attr("kg") != Some("6") {
                *consts += 1;
            }
            for c in &n.children {
                walk(c, nodes, consts);
            }
        }
        let (mut nodes, mut constants) = (0, 0);
        walk(root, &mut nodes, &mut constants);
        Some(NmbInterface { name, params, nodes, constants })
    }
    /// Parameter by UI label.
    pub fn param(&self, label: &str) -> Option<&Param> {
        self.params.iter().find(|p| p.label.as_deref() == Some(label))
    }
}

/// A behaviour placed in a scene (`SecondaryBehaviour`).
#[derive(Clone, Debug, PartialEq)]
pub struct BehaviourInstance {
    /// `Bob2: CatchFall 2.0` (== `.nmb` `RefNames/Name`).
    pub behaviour: String,
    pub character: String,
    pub start: i32,
    pub end: i32,
    pub priority: i32,
    pub enabled: bool,
    pub params: Vec<Param>,
}

impl BehaviourInstance {
    pub fn from_node(n: &Node) -> Option<BehaviourInstance> {
        let int = |k: &str| n.attr(k).and_then(|v| v.trim().parse::<i32>().ok());
        Some(BehaviourInstance {
            behaviour: n.attr("behaviourName")?.to_string(),
            character: n.attr("character")?.to_string(),
            start: int("startFrameNumber").unwrap_or(0),
            end: int("endFrameNumber").unwrap_or(0),
            priority: int("priority").unwrap_or(0),
            enabled: n.bool("enabled").unwrap_or(true),
            params: n.child("Parameters").map(|p| p.kids("Variable").map(Param::from_node).collect()).unwrap_or_default(),
        })
    }
    /// Parameter by scene key (`fk`), e.g. `ArmsStrength`.
    pub fn param(&self, key: &str) -> Option<&Param> {
        self.params.iter().find(|p| p.key == key)
    }
    /// Float parameter by key, `default` when absent / valueless.
    pub fn f32(&self, key: &str, default: f32) -> f32 {
        self.param(key).and_then(Param::as_f32).unwrap_or(default)
    }
    pub fn bool(&self, key: &str, default: bool) -> bool {
        self.param(key).and_then(Param::as_bool).unwrap_or(default)
    }
    /// Active at 0-based simulation frame `f` (inclusive start, exclusive end).
    pub fn active_at(&self, f: i32) -> bool {
        self.enabled && f >= self.start && f < self.end
    }
}

/// `<ForceEvent>`: a short push on one body. Semantics of `xf` / `relative_acceleration` are UNVERIFIED (see NMB.md / STATUS).
#[derive(Clone, Debug, PartialEq)]
pub struct ForceEvent {
    pub character: String,
    pub start: i32,
    pub end: i32,
    /// `relativeAcceleration`: multiples of the gravity magnitude (measured scale: 0.2).
    pub relative_acceleration: f32,
    /// Where / which way (scene frame): `Vp` = application point, `Vi`/`Vj`/`Vk` = frame (length 12).
    pub xf: Xf,
    /// `character02.MetaBodies.UpperSpineMass` -> `("character02", "UpperSpineMass")`
    pub targets: Vec<(String, String)>,
}

impl ForceEvent {
    pub fn from_node(n: &Node) -> ForceEvent {
        let int = |k: &str| n.attr(k).and_then(|v| v.trim().parse::<i32>().ok());
        ForceEvent {
            character: n.attr("character").unwrap_or("").to_string(),
            start: int("startFrameNumber").unwrap_or(0),
            end: int("endFrameNumber").unwrap_or(0),
            relative_acceleration: n.f32("relativeAcceleration").unwrap_or(0.0),
            xf: Xf::from_node(n).unwrap_or_default(),
            targets: n
                .child("Objects")
                .map(|o| {
                    o.kids("Object")
                        .filter_map(|ob| {
                            let t = ob.text.trim();
                            let (c, rest) = t.split_once(".MetaBodies.")?;
                            Some((c.to_string(), rest.to_string()))
                        })
                        .collect()
                })
                .unwrap_or_default(),
        }
    }
}

impl Scene {
    /// Every `SecondaryBehaviour` of the scene in file order (`behaviourName` carrying elements only).
    pub fn behaviour_instances(&self) -> Vec<BehaviourInstance> {
        self.raw
            .child("SecondaryBehaviours")
            .map(|g| g.children.iter().filter_map(BehaviourInstance::from_node).collect())
            .unwrap_or_default()
    }
    /// Every `<ForceEvent>`.
    pub fn force_events(&self) -> Vec<ForceEvent> {
        self.raw.child("ForceEvents").map(|g| g.kids("ForceEvent").map(ForceEvent::from_node).collect()).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::endorphin::{Document, parse_xml, read_endorphin, read_tree};
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    fn resources() -> Option<PathBuf> {
        let home = std::env::var_os("HOME")?;
        let p = std::env::var_os("ENDORPHIN_RESOURCES").map(PathBuf::from).unwrap_or_else(|| {
            Path::new(&home).join("Library/Application Support/CrossOver/Bottles/endorphin/drive_c/Program Files (x86)/NaturalMotion/endorphin 2.5.2/Resources")
        });
        p.is_dir().then_some(p)
    }

    fn walk(dir: &Path, ext: &str, out: &mut Vec<PathBuf>) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, ext, out);
            } else if p.extension().and_then(|x| x.to_str()) == Some(ext) {
                out.push(p);
            }
        }
    }

    const SYN_NMB: &str = r#"<cc vw="||O" sv="3" hs="1"><RefNames><Name>Bob2: Test</Name></RefNames><hi><cc sv="9" lo="x"><ry><wx sv="1" kg="0" ja="0.25"/><wx sv="2" kg="6" ja="blob" bq="blob"/></ry></cc></hi>
<ry><wx sv="5" kg="0" ja="0.6" de="Strength"/><wx sv="6" kg="2" ja="true" de="UseArms"/><wx sv="7" kg="0" de="_undefined"/><wx sv="8" kg="4" ja="Vector(1.0, 0.0, 0.0)"/></ry></cc>"#;

    #[test]
    fn nmb_interface_reads_root_params_and_counts_the_network() {
        let n = parse_xml(SYN_NMB).unwrap();
        let i = NmbInterface::from_node(&n).unwrap();
        assert_eq!(i.name, "Bob2: Test");
        assert_eq!(i.params.len(), 4);
        assert_eq!(i.nodes, 2);
        assert_eq!(i.constants, 4, "wx with ja (nested 0.25 + 3 root), blobs (kg 6) excluded");
        let s = i.param("Strength").unwrap();
        assert_eq!((s.kind, s.as_f32()), (ParamKind::Float, Some(0.6)));
        assert_eq!(i.param("UseArms").unwrap().as_bool(), Some(true));
        assert!(i.params[2].label.is_none(), "_undefined label dropped");
        assert_eq!(i.params[3].kind, ParamKind::Vector);
        assert_eq!(i.params[3].as_f32(), None);
        assert!(NmbInterface::from_node(&parse_xml("<Scene/>").unwrap()).is_none());
    }

    #[test]
    fn instance_params_and_force_events_decode() {
        let xml = r#"<Scene><SecondaryBehaviours><SecondaryBehaviour behaviourName="Bob2: CatchFall 2.0" character="character02" startFrameNumber="0" endFrameNumber="300" priority="0" enabled="true"><Parameters>
<Variable fk="ArmsApart" kg="0" ja="0.5" de="ArmsApart"/><Variable fk="ArmsStrength" kg="0" ja="0.7"/><Variable fk="PredictGroundHeight" kg="2" ja="true"/><Variable fk="GroundHeight" kg="0"/></Parameters></SecondaryBehaviour>
<SecondaryBehaviour name="no behaviourName"/></SecondaryBehaviours>
<ForceEvents><ForceEvent character="character02" startFrameNumber="0" endFrameNumber="1" brelativeAcceleration="3e4ccccd"><TransformMatrix><Vk bx="41400000" by="0" bz="0"/><Vp bx="3f99999a" by="3fb0a3d7" bz="0"/></TransformMatrix><Objects><Object>character02.MetaBodies.UpperSpineMass</Object><Object>junk</Object></Objects></ForceEvent></ForceEvents></Scene>"#;
        let s = Scene::from_node(&parse_xml(xml).unwrap());
        let b = s.behaviour_instances();
        assert_eq!(b.len(), 1);
        let b = &b[0];
        assert_eq!((b.behaviour.as_str(), b.character.as_str(), b.start, b.end), ("Bob2: CatchFall 2.0", "character02", 0, 300));
        assert_eq!(b.f32("ArmsStrength", -1.0), 0.7);
        assert_eq!(b.f32("GroundHeight", 9.0), 9.0, "valueless -> default");
        assert_eq!(b.f32("Missing", 4.0), 4.0);
        assert!(b.bool("PredictGroundHeight", false));
        assert!(b.active_at(0) && b.active_at(299) && !b.active_at(300) && !b.active_at(-1));
        let f = s.force_events();
        assert_eq!(f.len(), 1);
        assert!((f[0].relative_acceleration - 0.2).abs() < 1e-6);
        assert_eq!(f[0].targets, vec![("character02".to_string(), "UpperSpineMass".to_string())]);
        assert_eq!(f[0].xf.k, [12.0, 0.0, 0.0]);
    }

    #[test]
    fn hostile_input_returns_none_or_defaults() {
        for x in ["<cc/>", "<cc><RefNames/></cc>", "<cc><RefNames><Name/></RefNames><ry><wx kg=\"x\" sv=\"-1\"/></ry></cc>"] {
            let _ = NmbInterface::from_node(&parse_xml(x).unwrap());
        }
        let n = parse_xml("<SecondaryBehaviour behaviourName=\"a\" character=\"c\" startFrameNumber=\"zz\"/>").unwrap();
        assert_eq!(BehaviourInstance::from_node(&n).unwrap().start, 0);
        assert!(BehaviourInstance::from_node(&parse_xml("<SecondaryBehaviour/>").unwrap()).is_none());
    }

    /// NMB.md section 2, measured on the corpus: 47 `.nmb`, 135 scene instances, every instance names a `.nmb`, every labelled
    /// parameter exists in that `.nmb`'s root interface with the same type code.
    #[test]
    fn corpus_instances_match_nmb_interfaces() {
        let Some(root) = resources() else { return };
        let mut nmbs = Vec::new();
        walk(&root, "nmb", &mut nmbs);
        let mut by_name: BTreeMap<String, NmbInterface> = BTreeMap::new();
        for p in &nmbs {
            let n = read_tree(&std::fs::read(p).unwrap()).unwrap();
            let i = NmbInterface::from_node(&n).unwrap_or_else(|| panic!("{}", p.display()));
            by_name.insert(i.name.clone(), i);
        }
        assert_eq!(nmbs.len(), 47);
        assert_eq!(by_name.len(), 47, "47 distinct behaviour names");
        let mut ens = Vec::new();
        walk(&root, "ens", &mut ens);
        let (mut inst, mut labelled, mut found, mut same_kind, mut same_default) = (0, 0, 0, 0, 0);
        for p in &ens {
            let Document::Scene(s) = read_endorphin(&std::fs::read(p).unwrap()).unwrap() else { continue };
            for b in s.behaviour_instances() {
                inst += 1;
                let i = by_name.get(&b.behaviour).unwrap_or_else(|| panic!("{}: unknown behaviour {}", p.display(), b.behaviour));
                for v in &b.params {
                    let Some(l) = v.label.as_deref() else { continue };
                    labelled += 1;
                    if let Some(q) = i.param(l) {
                        found += 1;
                        same_kind += usize::from(q.kind == v.kind);
                        same_default += usize::from(q.value == v.value);
                    }
                }
            }
        }
        assert_eq!(inst, 135);
        assert_eq!((labelled, found, same_kind), (1457, 1457, 1457));
        assert_eq!(same_default, 961, "the rest are user overrides");
    }

    #[test]
    fn corpus_catchfall_scene_carries_decoded_params() {
        let Some(root) = resources() else { return };
        let Ok(bytes) = std::fs::read(root.join("Scenes/Behaviour Scenes/Catch fall2.0.ens")) else { return };
        let Document::Scene(s) = read_endorphin(&bytes).unwrap() else { panic!("scene") };
        let b = s.behaviour_instances();
        assert_eq!(b.len(), 2);
        assert!(b.iter().all(|x| x.behaviour == "Bob2: CatchFall 2.0" && x.start == 0 && x.end == 300));
        assert_eq!(b[0].f32("ArmsStrength", 0.0), 0.7);
        assert_eq!(b[0].f32("ArmsApart", 0.0), 0.5);
        assert!(b[0].bool("PredictGroundHeight", false));
        assert_eq!(s.force_events().len(), 2);
    }
}
