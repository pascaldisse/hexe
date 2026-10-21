//! Open reader for NaturalMotion Endorphin 2.x files (`.ens` scene, `.nmc` character, `.nmb` behaviour, `.nma` pose).
//! CLEAN-ROOM FROM DATA: every fact below was read off the sample corpus (see `oracle/endorphin/SURVEY.md`); no vendor
//! code, binary or documentation was consulted. Container = zlib stream (`78 01`) -> UTF-8 XML. Floats are written twice
//! per attribute: decimal `x="0.5"` and exact IEEE-754 hex `bx="3f000000"` (`bfloat`); the hex form wins.
//! Unknown elements/attributes are never dropped: every typed struct keeps its source subtree in `raw` (`Node`).
//! Hostile input is the default (AGENTS.md section 4): inflate, depth and node counts are capped.
use crate::GeoError;
use serde_json::{Map, Value};

/// Cap on inflated bytes.
pub const MAX_INFLATED: usize = 256 << 20;
/// Cap on element nesting.
pub const MAX_DEPTH: usize = 256;
/// Cap on elements per document.
pub const MAX_NODES: usize = 4_000_000;

fn bad(m: impl Into<String>) -> GeoError {
    GeoError::Syntax(format!("endorphin: {}", m.into()))
}

/// `78 xx` zlib stream -> bytes (capped).
pub fn inflate(bytes: &[u8]) -> Result<Vec<u8>, GeoError> {
    miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(bytes, MAX_INFLATED)
        .map_err(|e| bad(format!("inflate: {:?}", e.status)))
}

/// Hex IEEE-754 single (`"c11cf5c3"` = -9.81); 1..=8 hex digits, no sign/prefix.
pub fn bfloat(s: &str) -> Option<f32> {
    if s.is_empty() || s.len() > 8 || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    u32::from_str_radix(s, 16).ok().map(f32::from_bits)
}

// ---------------------------------------------------------------- raw XML tree

/// One XML element, lossless for our purposes (attribute order kept, text concatenated).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Node {
    pub tag: String,
    pub attrs: Vec<(String, String)>,
    pub children: Vec<Node>,
    pub text: String,
}

impl Node {
    pub fn attr(&self, name: &str) -> Option<&str> {
        self.attrs.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }
    /// Float attribute: the exact hex twin `b<name>` first, then the decimal `<name>`.
    pub fn f32(&self, name: &str) -> Option<f32> {
        let hex = format!("b{name}");
        if let Some(v) = self.attr(&hex).and_then(bfloat) {
            return Some(v);
        }
        self.attr(name).and_then(|v| v.trim().parse::<f32>().ok()).filter(|v| !v.is_nan())
    }
    pub fn bool(&self, name: &str) -> Option<bool> {
        match self.attr(name)? {
            "true" => Some(true),
            "false" => Some(false),
            _ => None,
        }
    }
    pub fn child(&self, tag: &str) -> Option<&Node> {
        self.children.iter().find(|c| c.tag == tag)
    }
    pub fn kids<'a>(&'a self, tag: &'a str) -> impl Iterator<Item = &'a Node> + 'a {
        self.children.iter().filter(move |c| c.tag == tag)
    }
    /// `x,y,z` (hex-first) of this element, e.g. `<Gravity bx= by= bz=/>`.
    pub fn vec3(&self) -> [f32; 3] {
        [self.f32("x").unwrap_or(0.0), self.f32("y").unwrap_or(0.0), self.f32("z").unwrap_or(0.0)]
    }
    /// Elements in the subtree (self included).
    pub fn count(&self) -> usize {
        1 + self.children.iter().map(Node::count).sum::<usize>()
    }
    /// Lossless JSON image `{tag, attrs{}, text?, children[]}` (attribute order is not kept in JSON objects).
    pub fn to_json(&self) -> Value {
        let mut m = Map::new();
        m.insert("tag".into(), Value::String(self.tag.clone()));
        if !self.attrs.is_empty() {
            let a: Map<String, Value> = self.attrs.iter().map(|(k, v)| (k.clone(), Value::String(v.clone()))).collect();
            m.insert("attrs".into(), Value::Object(a));
        }
        if !self.text.trim().is_empty() {
            m.insert("text".into(), Value::String(self.text.clone()));
        }
        if !self.children.is_empty() {
            m.insert("children".into(), Value::Array(self.children.iter().map(Node::to_json).collect()));
        }
        Value::Object(m)
    }
}

struct P<'a> {
    s: &'a [u8],
    i: usize,
    nodes: usize,
}

fn is_name(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'_' | b':' | b'-' | b'.')
}

fn unescape(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(p) = rest.find('&') {
        out.push_str(&rest[..p]);
        rest = &rest[p..];
        let Some(semi) = rest.find(';').filter(|&e| e <= 12) else {
            out.push('&');
            rest = &rest[1..];
            continue;
        };
        let ent = &rest[1..semi];
        let ch = match ent {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            _ => ent
                .strip_prefix("#x")
                .and_then(|h| u32::from_str_radix(h, 16).ok())
                .or_else(|| ent.strip_prefix('#').and_then(|d| d.parse::<u32>().ok()))
                .and_then(char::from_u32),
        };
        match ch {
            Some(c) => {
                out.push(c);
                rest = &rest[semi + 1..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

impl P<'_> {
    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }
    fn starts(&self, pat: &[u8]) -> bool {
        self.s.get(self.i..).is_some_and(|r| r.starts_with(pat))
    }
    fn ws(&mut self) {
        while self.peek().is_some_and(|b| b.is_ascii_whitespace()) {
            self.i += 1;
        }
    }
    fn skip_to(&mut self, pat: &[u8]) -> Result<(), GeoError> {
        let rest = self.s.get(self.i..).unwrap_or(&[]);
        let at = rest.windows(pat.len()).position(|w| w == pat).ok_or_else(|| bad("unterminated markup"))?;
        self.i += at + pat.len();
        Ok(())
    }
    fn name(&mut self) -> Result<String, GeoError> {
        let st = self.i;
        while self.peek().is_some_and(is_name) {
            self.i += 1;
        }
        if st == self.i {
            return Err(bad(format!("expected a name at byte {st}")));
        }
        Ok(String::from_utf8_lossy(&self.s[st..self.i]).into_owned())
    }
    fn text(&self, a: usize, b: usize) -> String {
        unescape(&String::from_utf8_lossy(&self.s[a..b]))
    }
    fn element(&mut self, depth: usize) -> Result<Node, GeoError> {
        if depth > MAX_DEPTH {
            return Err(bad("element nesting too deep"));
        }
        self.nodes += 1;
        if self.nodes > MAX_NODES {
            return Err(bad("too many elements"));
        }
        self.i += 1; // '<'
        let mut n = Node { tag: self.name()?, ..Node::default() };
        loop {
            self.ws();
            match self.peek() {
                Some(b'/') => {
                    self.i += 1;
                    if self.peek() != Some(b'>') {
                        return Err(bad("expected '>' after '/'"));
                    }
                    self.i += 1;
                    return Ok(n);
                }
                Some(b'>') => {
                    self.i += 1;
                    break;
                }
                Some(_) => {
                    let k = self.name()?;
                    self.ws();
                    if self.peek() != Some(b'=') {
                        return Err(bad(format!("attribute `{k}` without value")));
                    }
                    self.i += 1;
                    self.ws();
                    let q =
                        self.peek().filter(|q| matches!(q, b'"' | b'\'')).ok_or_else(|| bad("unquoted attribute"))?;
                    self.i += 1;
                    let st = self.i;
                    while self.peek().is_some_and(|b| b != q) {
                        self.i += 1;
                    }
                    if self.peek().is_none() {
                        return Err(bad("unterminated attribute"));
                    }
                    let v = self.text(st, self.i);
                    self.i += 1;
                    n.attrs.push((k, v));
                }
                None => return Err(bad("eof inside a tag")),
            }
        }
        // content
        loop {
            let st = self.i;
            while self.peek().is_some_and(|b| b != b'<') {
                self.i += 1;
            }
            if self.i > st {
                n.text.push_str(&self.text(st, self.i));
            }
            if self.peek().is_none() {
                return Err(bad(format!("eof inside <{}>", n.tag)));
            }
            if self.starts(b"</") {
                self.i += 2;
                let close = self.name()?;
                self.ws();
                if self.peek() != Some(b'>') || close != n.tag {
                    return Err(bad(format!("</{close}> does not close <{}>", n.tag)));
                }
                self.i += 1;
                return Ok(n);
            } else if self.starts(b"<!--") {
                self.skip_to(b"-->")?;
            } else if self.starts(b"<![CDATA[") {
                self.i += 9;
                let st = self.i;
                self.skip_to(b"]]>")?;
                n.text.push_str(&String::from_utf8_lossy(&self.s[st..self.i - 3]));
            } else if self.starts(b"<?") {
                self.skip_to(b"?>")?;
            } else {
                let c = self.element(depth + 1)?;
                n.children.push(c);
            }
        }
    }
}

/// XML text -> root element (prolog, comments, PIs skipped). Attribute order and every element are kept.
pub fn parse_xml(xml: &str) -> Result<Node, GeoError> {
    let mut p = P { s: xml.as_bytes(), i: 0, nodes: 0 };
    loop {
        p.ws();
        if p.starts(b"\xEF\xBB\xBF") {
            p.i += 3;
        } else if p.starts(b"<?") {
            p.skip_to(b"?>")?;
        } else if p.starts(b"<!--") {
            p.skip_to(b"-->")?;
        } else if p.starts(b"<!") {
            p.skip_to(b">")?;
        } else if p.peek() == Some(b'<') {
            return p.element(0);
        } else {
            return Err(bad("no root element"));
        }
    }
}

/// Container bytes (zlib, or bare XML) -> root element.
pub fn read_tree(bytes: &[u8]) -> Result<Node, GeoError> {
    let owned;
    let xml: &[u8] = if bytes.first() == Some(&0x78) {
        owned = inflate(bytes)?;
        &owned
    } else {
        bytes
    };
    parse_xml(std::str::from_utf8(xml).map_err(|_| bad("payload is not UTF-8"))?)
}

// ---------------------------------------------------------------- typed layer

/// `<TransformMatrix><Vi/><Vj/><Vk/><Vp/></TransformMatrix>`: basis columns `i j k` + translation `p`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Xf {
    pub i: [f32; 3],
    pub j: [f32; 3],
    pub k: [f32; 3],
    pub p: [f32; 3],
}

impl Default for Xf {
    fn default() -> Self {
        Xf { i: [1.0, 0.0, 0.0], j: [0.0, 1.0, 0.0], k: [0.0, 0.0, 1.0], p: [0.0; 3] }
    }
}

impl Xf {
    pub fn from_node(parent: &Node) -> Option<Xf> {
        let t = parent.child("TransformMatrix")?;
        let g = |n: &str, d: [f32; 3]| t.child(n).map_or(d, Node::vec3);
        let id = Xf::default();
        Some(Xf { i: g("Vi", id.i), j: g("Vj", id.j), k: g("Vk", id.k), p: g("Vp", id.p) })
    }
    /// Rotate (no translation).
    pub fn rot(&self, v: [f32; 3]) -> [f32; 3] {
        [0, 1, 2].map(|c| self.i[c] * v[0] + self.j[c] * v[1] + self.k[c] * v[2])
    }
    pub fn apply(&self, v: [f32; 3]) -> [f32; 3] {
        let r = self.rot(v);
        [r[0] + self.p[0], r[1] + self.p[1], r[2] + self.p[2]]
    }
    /// `self * o` (apply `o` first).
    pub fn mul(&self, o: &Xf) -> Xf {
        Xf { i: self.rot(o.i), j: self.rot(o.j), k: self.rot(o.k), p: self.apply(o.p) }
    }
}

impl Xf {
    /// Rotation matrix of the unit quaternion `(w, x, y, z)` as basis columns (column-vector convention, like `rot`).
    pub fn from_quat(q: [f32; 4]) -> Xf {
        let n = q.iter().map(|v| v * v).sum::<f32>().sqrt();
        if !n.is_finite() || n < 1e-12 {
            return Xf::default();
        }
        let [w, x, y, z] = q.map(|v| v / n);
        Xf {
            i: [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)],
            j: [2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)],
            k: [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)],
            p: [0.0; 3],
        }
    }
}
/// `<Geometry type="Sphyl|Sphere|Cylinder|Cuboid|File|Mesh" param0..>`.
#[derive(Clone, Debug, PartialEq)]
pub struct Geometry {
    pub kind: String,
    /// `param0..param3` (missing = 0).
    pub params: [f32; 4],
    pub file: Option<String>,
    pub raw: Node,
}

impl Geometry {
    pub fn from_node(n: &Node) -> Geometry {
        let p = |k: &str| n.f32(k).unwrap_or(0.0);
        Geometry {
            kind: n.attr("type").unwrap_or("").to_string(),
            params: [p("param0"), p("param1"), p("param2"), p("param3")],
            file: n.attr("file").map(str::to_string),
            raw: n.clone(),
        }
    }
    /// Analytic volume (m^3) of the primitive kinds; `None` for File/Mesh/unknown.
    /// Sphyl = capsule (radius p0, cylinder length p1 along local Z); Cylinder = radius p0, height p1 (Z); Cuboid = full extents p0,p1,p2.
    pub fn volume(&self) -> Option<f32> {
        let [a, b, c, _] = self.params;
        let pi = std::f32::consts::PI;
        match self.kind.as_str() {
            "Sphere" => Some(4.0 / 3.0 * pi * a * a * a),
            "Sphyl" => Some(pi * a * a * b + 4.0 / 3.0 * pi * a * a * a),
            "Cylinder" => Some(pi * a * a * b),
            "Cuboid" => Some(a * b * c),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Collision {
    pub name: String,
    pub material: String,
    pub xf: Xf,
    pub geom: Option<Geometry>,
    pub raw: Node,
}

/// `<MetaBody>`: a rigid body = collision surfaces + one inertial body (density x volume = mass) + optional graphics.
#[derive(Clone, Debug, PartialEq)]
pub struct Body {
    pub name: String,
    /// Relative to the joint that names this body (`metaBody=` attr), world-relative for prop roots.
    pub xf: Xf,
    pub collisions: Vec<Collision>,
    pub inertial: Option<Geometry>,
    pub density: Option<f32>,
    pub raw: Node,
}

impl Body {
    /// density x inertial-geometry volume; `None` when either is unknown (File/Mesh inertial).
    pub fn mass(&self) -> Option<f32> {
        Some(self.density? * self.inertial.as_ref()?.volume()?)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Limits {
    pub twist: f32,
    pub swing1: f32,
    pub swing2: f32,
    pub radius: f32,
    pub twist_stiffness: f32,
    pub swing_stiffness: f32,
    pub linear_stiffness: f32,
}

impl Limits {
    fn from_node(n: &Node) -> Limits {
        let g = |k: &str| n.f32(k).unwrap_or(0.0);
        Limits {
            twist: g("twistSize"),
            swing1: g("swing1Size"),
            swing2: g("swing2Size"),
            radius: g("radius"),
            twist_stiffness: g("twistStiffness"),
            swing_stiffness: g("swingStiffness"),
            linear_stiffness: g("linearStiffness"),
        }
    }
}

/// `<Skeleton>` (root), `<SkeletalJoint>`, `<Null>` (prop joint), or any other element found under `<ChildJoints>`.
#[derive(Clone, Debug, PartialEq)]
pub struct Joint {
    pub name: String,
    pub kind: String,
    pub parent: Option<usize>,
    /// Local to the parent joint.
    pub xf: Xf,
    pub body: Option<String>,
    pub fixed_coords: Option<String>,
    pub dof_order: Option<String>,
    pub soft: Option<Limits>,
    pub hard: Option<Limits>,
    /// The element without its `ChildJoints` (children are separate `Joint`s).
    pub raw: Node,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Character {
    pub name: String,
    pub path_name: Option<String>,
    pub type_name: Option<String>,
    pub joints: Vec<Joint>,
    pub bodies: Vec<Body>,
    /// Scene placement: the `<Character>`'s own `TransformMatrix` (world = placement x joint chain). Identity for a bare `<Skeleton>`.
    pub xf: Xf,
    /// The whole `<Character>` (or bare `<Skeleton>`) subtree, unabridged.
    pub raw: Node,
}

impl Character {
    pub fn from_node(n: &Node) -> Character {
        let mut c = Character {
            name: n.attr("name").unwrap_or("").to_string(),
            path_name: n.attr("pathName").map(str::to_string),
            type_name: n.attr("typeName").map(str::to_string),
            joints: Vec::new(),
            bodies: Vec::new(),
            xf: if n.tag == "Character" { Xf::from_node(n).unwrap_or_default() } else { Xf::default() },
            raw: n.clone(),
        };
        let skel = if n.tag == "Skeleton" { Some(n) } else { n.child("Skeleton") };
        if let Some(s) = skel {
            collect_joints(s, None, &mut c.joints);
        }
        if let Some(mbs) = n.child("MetaBodies") {
            c.bodies = mbs.kids("MetaBody").map(body_from).collect();
        }
        c
    }
    pub fn body(&self, name: &str) -> Option<&Body> {
        self.bodies.iter().find(|b| b.name == name)
    }
    /// Rest joint-local transforms (`<Joint>`'s own `TransformMatrix`), in joint order.
    pub fn rest_locals(&self) -> Vec<Xf> {
        self.joints.iter().map(|j| j.xf).collect()
    }
    /// `base` overlaid with the animation channels of `chans` sampled at `t` seconds (parent-local, joint order).
    /// Channel naming rule (SURVEY.md "Scene pose"): `<char>.Joints.<joint>.{quat,tx,ty,tz}` or `<char>.<joint>.{...}`.
    pub fn locals_at(&self, base: &[Xf], chans: &[Channel], t: f32) -> Vec<Xf> {
        let by_name: std::collections::HashMap<&str, &Channel> =
            chans.iter().map(|ch| (ch.name.as_str(), ch)).collect();
        self.joints
            .iter()
            .enumerate()
            .map(|(idx, j)| {
                let mut x = base.get(idx).copied().unwrap_or(j.xf);
                for pre in [format!("{}.Joints.{}.", self.name, j.name), format!("{}.{}.", self.name, j.name)] {
                    let get = |leaf: &str| by_name.get(format!("{pre}{leaf}").as_str()).and_then(|ch| ch.sample(t));
                    if let Some(q) = get("quat").filter(|q| q.len() == 4) {
                        let r = Xf::from_quat([q[0], q[1], q[2], q[3]]);
                        (x.i, x.j, x.k) = (r.i, r.j, r.k);
                    }
                    for (axis, leaf) in ["tx", "ty", "tz"].into_iter().enumerate() {
                        if let (Some(v), Some(slot)) = (get(leaf).and_then(|v| v.first().copied()), x.p.get_mut(axis))
                        {
                            *slot = v;
                        }
                    }
                }
                x
            })
            .collect()
    }
    /// Joint world transforms from parent-local `locals` (joint order): chain product, no placement.
    pub fn world_of(&self, locals: &[Xf]) -> Vec<Xf> {
        let mut w: Vec<Xf> = Vec::with_capacity(self.joints.len());
        for (idx, j) in self.joints.iter().enumerate() {
            let local = locals.get(idx).copied().unwrap_or(j.xf);
            let x = match j.parent.and_then(|p| w.get(p)) {
                Some(pw) => pw.mul(&local),
                None => local,
            };
            w.push(x);
        }
        w
    }
    /// Joint world transforms (rest pose; `root_p` overrides the root joint's translation).
    pub fn world(&self, root_p: Option<[f32; 3]>) -> Vec<Xf> {
        let mut w: Vec<Xf> = Vec::with_capacity(self.joints.len());
        for (idx, j) in self.joints.iter().enumerate() {
            let mut local = j.xf;
            if let (0, Some(p)) = (idx, root_p) {
                local.p = p;
            }
            let x = match j.parent.and_then(|p| w.get(p)) {
                Some(pw) => pw.mul(&local),
                None => local,
            };
            w.push(x);
        }
        w
    }
}

fn collect_joints(n: &Node, parent: Option<usize>, out: &mut Vec<Joint>) {
    let idx = out.len();
    let mut raw = n.clone();
    raw.children.retain(|c| c.tag != "ChildJoints");
    out.push(Joint {
        name: n.attr("name").unwrap_or("").to_string(),
        kind: n.tag.clone(),
        parent,
        xf: Xf::from_node(n).unwrap_or_default(),
        body: n.attr("metaBody").map(str::to_string),
        fixed_coords: n.attr("fixedCoords").map(str::to_string),
        dof_order: n.attr("dofOrder").map(str::to_string),
        soft: n.child("SoftLimits").map(Limits::from_node),
        hard: n.child("HardLimits").map(Limits::from_node),
        raw,
    });
    if let Some(cj) = n.child("ChildJoints") {
        for c in &cj.children {
            collect_joints(c, Some(idx), out);
        }
    }
}

fn body_from(n: &Node) -> Body {
    let ib = n.child("InertialBody");
    Body {
        name: n.attr("name").unwrap_or("").to_string(),
        xf: Xf::from_node(n).unwrap_or_default(),
        collisions: n
            .kids("CollisionSurface")
            .map(|c| Collision {
                name: c.attr("name").unwrap_or("").to_string(),
                material: c.attr("material").unwrap_or("").to_string(),
                xf: Xf::from_node(c).unwrap_or_default(),
                geom: c.child("Geometry").map(Geometry::from_node),
                raw: c.clone(),
            })
            .collect(),
        inertial: ib.and_then(|i| i.child("Geometry")).map(Geometry::from_node),
        density: ib.and_then(|i| i.child("Inertia")).and_then(|i| i.f32("density")),
        raw: n.clone(),
    }
}

/// A scene-level behaviour element (`Primary|Secondary|TertiaryBehaviour`) or a `.nmb` file (root `<cc>`).
#[derive(Clone, Debug, PartialEq)]
pub struct Behaviour {
    /// Element tag (`TertiaryBehaviour`, `SecondaryBehaviour`, `cc`, ...).
    pub kind: String,
    pub name: String,
    /// `<RefNames><Name>` entries (.nmb).
    pub ref_names: Vec<String>,
    pub raw: Node,
}

impl Behaviour {
    pub fn from_node(n: &Node) -> Behaviour {
        Behaviour {
            kind: n.tag.clone(),
            name: n
                .attr("name")
                .map(str::to_string)
                .or_else(|| n.child("RefNames").and_then(|r| r.child("Name")).map(|m| m.text.clone()))
                .unwrap_or_default(),
            ref_names: n
                .child("RefNames")
                .map(|r| r.kids("Name").map(|m| m.text.clone()).collect())
                .unwrap_or_default(),
            raw: n.clone(),
        }
    }
}

/// `.nma` / scene `<Pose>`: per-relationship transforms (+ metabody overrides, kept raw).
#[derive(Clone, Debug, PartialEq)]
pub struct Pose {
    pub character: String,
    pub ref_frame: String,
    pub relationships: Vec<(String, Option<Xf>)>,
    pub raw: Node,
}

impl Pose {
    pub fn from_node(n: &Node) -> Pose {
        Pose {
            character: n.attr("character").unwrap_or("").to_string(),
            ref_frame: n.attr("refFrame").unwrap_or("").to_string(),
            relationships: n
                .child("Relationships")
                .map(|r| {
                    r.kids("Relationship")
                        .map(|x| (x.attr("name").unwrap_or("").to_string(), Xf::from_node(x)))
                        .collect()
                })
                .unwrap_or_default(),
            raw: n.clone(),
        }
    }
}

/// Scene `<Animation>` channel: `Channel` keys = `[x(time), y(value)]`, `QuatChannel` keys = `[t, x, y, z, w]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Channel {
    /// `character01.Joints.LeftKneeJoint.quat`, `character01.root.ty`, ...
    pub name: String,
    pub quat: bool,
    pub interp: String,
    pub keys: Vec<Vec<f32>>,
}

impl Channel {
    /// Linear sample at `t` seconds (clamped to the first/last key). Scalar -> 1 value; quat -> `(w, x, y, z)`, shortest-arc nlerp.
    /// Key `t` is in seconds (corpus: 1/30 s spacing). NB the file names quat attrs `bx by bz bw` but the order on disk is `w x y z`
    /// (worked example in SURVEY.md "Scene pose").
    pub fn sample(&self, t: f32) -> Option<Vec<f32>> {
        let n = self.keys.len();
        let need = if self.quat { 5 } else { 2 };
        if n == 0 || self.keys.iter().any(|k| k.len() < need) {
            return None;
        }
        let vals = |k: &Vec<f32>| -> Vec<f32> { k[1..need].to_vec() };
        let (first, last) = (&self.keys[0], &self.keys[n - 1]);
        if !t.is_finite() || t <= first[0] || n == 1 {
            return Some(vals(first));
        }
        if t >= last[0] {
            return Some(vals(last));
        }
        let hi = self.keys.iter().position(|k| k[0] > t)?;
        let (a, b) = (&self.keys[hi - 1], &self.keys[hi]);
        let span = b[0] - a[0];
        let u = if span > 1e-9 { (t - a[0]) / span } else { 1.0 };
        let (va, mut vb) = (vals(a), vals(b));
        if self.quat && va.iter().zip(&vb).map(|(x, y)| x * y).sum::<f32>() < 0.0 {
            vb.iter_mut().for_each(|v| *v = -*v);
        }
        Some(va.iter().zip(&vb).map(|(x, y)| x + (y - x) * u).collect())
    }
    /// Value at the first key (a scalar channel's `y`).
    pub fn first_value(&self) -> Option<f32> {
        let k = self.keys.first()?;
        if self.quat { None } else { k.get(1).copied() }
    }
}

/// Every `<Channel>` / `<QuatChannel>` directly under `anim` (a scene `<Animation>` or an `<AnimationEvent><Animation>`).
pub fn channels_of(anim: &Node) -> Vec<Channel> {
    anim.children
        .iter()
        .filter(|c| c.tag == "Channel" || c.tag == "QuatChannel")
        .map(|c| Channel {
            name: c.attr("name").unwrap_or("").to_string(),
            quat: c.tag == "QuatChannel",
            interp: c.attr("type").unwrap_or("").to_string(),
            keys: c
                .kids("Key")
                .map(|k| {
                    if c.tag == "QuatChannel" {
                        ["t", "x", "y", "z", "w"].iter().map(|a| k.f32(a).unwrap_or(0.0)).collect()
                    } else {
                        ["x", "y"].iter().map(|a| k.f32(a).unwrap_or(0.0)).collect()
                    }
                })
                .collect(),
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq)]
pub struct Scene {
    pub gravity: [f32; 3],
    pub unit_length: Option<f32>,
    pub unit_mass: Option<f32>,
    pub use_default_ground: bool,
    pub characters: Vec<Character>,
    /// `<Environment>` (the `background` prop character).
    pub environment: Option<Character>,
    pub behaviours: Vec<Behaviour>,
    pub poses: Vec<Pose>,
    pub channels: Vec<Channel>,
    pub raw: Node,
}

impl Scene {
    pub fn from_node(n: &Node) -> Scene {
        let mut behaviours = Vec::new();
        for grp in ["PrimaryBehaviours", "SecondaryBehaviours", "TertiaryBehaviours"] {
            if let Some(g) = n.child(grp) {
                behaviours.extend(g.children.iter().map(Behaviour::from_node));
            }
        }
        let channels = n.child("Animation").map(channels_of).unwrap_or_default();
        let eng = n.child("Engine");
        Scene {
            gravity: eng.and_then(|e| e.child("Gravity")).map_or([0.0, -9.81, 0.0], Node::vec3),
            unit_length: eng.and_then(|e| e.f32("unitLength")),
            unit_mass: eng.and_then(|e| e.f32("unitMass")),
            use_default_ground: n.bool("useDefaultGround").unwrap_or(false),
            characters: n
                .child("Characters")
                .map(|c| c.kids("Character").map(Character::from_node).collect())
                .unwrap_or_default(),
            environment: n.child("Environment").map(Character::from_node),
            behaviours,
            poses: n.child("Poses").map(|p| p.kids("Pose").map(Pose::from_node).collect()).unwrap_or_default(),
            channels,
            raw: n.clone(),
        }
    }
    /// Parent-local joint transforms of `c` at `t` seconds. Rule (SURVEY.md "Scene pose"), per joint, later wins:
    /// rig rest `TransformMatrix` < scene `Poses/Pose[character=c]/Relationship[name=joint]` (full parent-local matrix)
    /// < timeline channels `c.Joints.<joint>.{quat,tx,ty,tz}` or `c.<joint>.{...}` (root = `c.root.*`) sampled at `t`.
    pub fn posed_locals(&self, c: &Character, t: f32) -> Vec<Xf> {
        let pose = self
            .poses
            .iter()
            .find(|p| p.character == c.name && (p.ref_frame.is_empty() || p.ref_frame.eq_ignore_ascii_case("parent")));
        let by_name: std::collections::HashMap<&str, &Channel> = self
            .channels
            .iter()
            .filter(|ch| ch.name.starts_with(c.name.as_str()))
            .map(|ch| (ch.name.as_str(), ch))
            .collect();
        c.joints
            .iter()
            .map(|j| {
                let mut x = j.xf;
                if let Some(Some(px)) =
                    pose.and_then(|p| p.relationships.iter().find(|(n, _)| *n == j.name)).map(|(_, x)| x)
                {
                    x = *px;
                }
                for pre in [format!("{}.Joints.{}.", c.name, j.name), format!("{}.{}.", c.name, j.name)] {
                    let get = |leaf: &str| by_name.get(format!("{pre}{leaf}").as_str()).and_then(|ch| ch.sample(t));
                    if let Some(q) = get("quat").filter(|q| q.len() == 4) {
                        let r = Xf::from_quat([q[0], q[1], q[2], q[3]]);
                        (x.i, x.j, x.k) = (r.i, r.j, r.k);
                    }
                    for (axis, leaf) in ["tx", "ty", "tz"].into_iter().enumerate() {
                        if let (Some(v), Some(slot)) = (get(leaf).and_then(|v| v.first().copied()), x.p.get_mut(axis)) {
                            *slot = v;
                        }
                    }
                }
                x
            })
            .collect()
    }
    /// Last key time (seconds) over every channel; 0 when nothing is keyed beyond one key.
    pub fn duration(&self) -> f32 {
        self.channels.iter().filter_map(|c| c.keys.last().and_then(|k| k.first().copied())).fold(0.0f32, f32::max)
    }
    /// First-key scalar of `<character>.<path>` (e.g. `character01`, `root.ty`).
    pub fn channel0(&self, character: &str, path: &str) -> Option<f32> {
        let want = format!("{character}.{path}");
        self.channels.iter().find(|c| c.name == want)?.first_value()
    }
}

/// One opened file.
#[derive(Clone, Debug, PartialEq)]
pub enum Document {
    Scene(Box<Scene>),
    Character(Character),
    Behaviour(Behaviour),
    Pose(Pose),
    /// Root element we have no typed form for (kept raw).
    Unknown(Node),
}

impl Document {
    pub fn from_node(n: Node) -> Document {
        match n.tag.as_str() {
            "Scene" => Document::Scene(Box::new(Scene::from_node(&n))),
            "Character" | "Skeleton" => Document::Character(Character::from_node(&n)),
            "Pose" => Document::Pose(Pose::from_node(&n)),
            "cc" => Document::Behaviour(Behaviour::from_node(&n)),
            _ => Document::Unknown(n),
        }
    }
    pub fn raw(&self) -> &Node {
        match self {
            Document::Scene(s) => &s.raw,
            Document::Character(c) => &c.raw,
            Document::Behaviour(b) => &b.raw,
            Document::Pose(p) => &p.raw,
            Document::Unknown(n) => n,
        }
    }
}

/// Bytes of any `.ens/.nmc/.nmb/.nma` -> typed document.
pub fn read_endorphin(bytes: &[u8]) -> Result<Document, GeoError> {
    Ok(Document::from_node(read_tree(bytes)?))
}

// ---------------------------------------------------------------- vocabulary survey

/// Per-element-tag vocabulary: how many elements, in how many files, which attributes (and how often).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TagStat {
    pub elements: u64,
    pub files: u64,
    pub attrs: std::collections::BTreeMap<String, u64>,
    pub child_tags: std::collections::BTreeMap<String, u64>,
    pub with_text: u64,
}

/// Tag -> stat, merged over many files.
#[derive(Clone, Debug, Default)]
pub struct Vocab {
    pub tags: std::collections::BTreeMap<String, TagStat>,
    pub files: u64,
}

impl Vocab {
    /// Add one document (counts `files` once per tag per call).
    pub fn add(&mut self, root: &Node) {
        self.files += 1;
        let mut seen = std::collections::BTreeSet::new();
        fn walk(n: &Node, v: &mut Vocab, seen: &mut std::collections::BTreeSet<String>) {
            let t = v.tags.entry(n.tag.clone()).or_default();
            t.elements += 1;
            if seen.insert(n.tag.clone()) {
                t.files += 1;
            }
            for (k, _) in &n.attrs {
                *t.attrs.entry(k.clone()).or_insert(0) += 1;
            }
            for c in &n.children {
                *t.child_tags.entry(c.tag.clone()).or_insert(0) += 1;
            }
            if !n.text.trim().is_empty() {
                t.with_text += 1;
            }
            for c in &n.children {
                walk(c, v, seen);
            }
        }
        walk(root, self, &mut seen);
    }
}
