//! PolyReduce SOP (token `polyreduce`): reduce polygon count by Garland-Heckbert quadric error metric.
//!
//! # Semantics (DOC-VERIFIED vs Houdini 20.5 help `sop/polyreduce.txt`, tests/polyreduce_semantics.rs: topology invariants; no recorded row.
//! Boundary is HARD here, a weight in Houdini, so a target can hit a boundary floor)
//! * `target` menu percentage(0)|finalcount(1); `percentage` F 50 (0..=100) OR `finalcount` I (>=0): target final prim count.
//!   100% = input Arc shared. `preserveboundary` B on (default): boundary edges never collapsed.
//! * Quadric error metric (Garland-Heckbert): per-vertex 4×4 Q from face planes sum_{a,b,c,d} K K^T where plane ax+by+cz+d=0,
//!   edge cost = v^T(Q1+Q2)v at optimal v = (Q1+Q2)^{-1} [0,0,0,1]^T, fallback midpoint if singular. Min-heap of edges with
//!   lazy invalidation (version stamps). Normal-flip rejection: skip collapse if any incident face normal flips > 90°.
//!   Degenerate faces (<3 distinct corners) removed. Boundary quadric penalty: boundary edges contribute weighted Q.
//! * Attributes: merged points blend (`meshutil::rebuild`); surviving vertices/prims copy theirs; groups follow.
//!   Non-poly prims pass through. NOT modelled (DEFERRED): attribute-seam/UV preservation, symmetry.
use super::meshutil::{Remap, V3, bad, failed, positions, rebuild, to_f32, toggle};
use crate::SopNode;
use hexe_geo::{Detail, Prim, PrimKind};
use hexe_graph::{CookCtx, CookError, InputSpec, Node, ParamSpec, ParamValue};
use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap};
use std::sync::Arc;

/// Houdini's input labels (PolyReduce 1.0, help `sop/polyreduce-.txt`): the mesh, feature polygons (their edges, picked by
/// the `creases` group, resist collapse by `creaseweight` "Stiffen Features"), a reference mesh; 2 and 3 match input 1 by
/// point number. Inputs 2 / 3 wired with geometry = explicit DEFERRED error (feature edges / reference not modelled yet).
const INPUTS: [InputSpec; 3] = [
    InputSpec { name: "Triangular Mesh" },
    InputSpec { name: "Feature Polygons" },
    InputSpec { name: "Reference Mesh" },
];
const OP: &str = "polyreduce";

/// 4×4 symmetric matrix for quadric error metric
#[derive(Clone, Copy, Default)]
struct Quadric {
    /// Upper triangle: [0,0] [0,1] [0,2] [0,3] [1,1] [1,2] [1,3] [2,2] [2,3] [3,3]
    m: [f64; 10],
}

impl Quadric {
    fn from_plane(a: f64, b: f64, c: f64, d: f64) -> Self {
        Self { m: [a * a, a * b, a * c, a * d, b * b, b * c, b * d, c * c, c * d, d * d] }
    }
    fn add(&self, other: &Self) -> Self {
        let mut m = [0.0; 10];
        for (i, item) in m.iter_mut().enumerate() {
            *item = self.m[i] + other.m[i];
        }
        Self { m }
    }
    fn scale(&self, s: f64) -> Self {
        let mut m = [0.0; 10];
        for (i, item) in m.iter_mut().enumerate() {
            *item = self.m[i] * s;
        }
        Self { m }
    }
    /// Error at point (x, y, z)
    fn error(&self, x: f64, y: f64, z: f64) -> f64 {
        let [m00, m01, m02, m03, m11, m12, m13, m22, m23, m33] = self.m;
        let v0 = m00 * x + m01 * y + m02 * z + m03;
        let v1 = m01 * x + m11 * y + m12 * z + m13;
        let v2 = m02 * x + m12 * y + m22 * z + m23;
        let v3 = m03 * x + m13 * y + m23 * z + m33;
        x * v0 + y * v1 + z * v2 + v3
    }
    /// Try to find optimal position minimizing error; return None if singular
    fn optimal(&self) -> Option<[f64; 3]> {
        let [m00, m01, m02, m03, m11, m12, m13, m22, m23, _m33] = self.m;
        // Solve [Q_3x3] v = -[m03, m13, m23]^T via Cramer's rule
        let det = m00 * (m11 * m22 - m12 * m12) - m01 * (m01 * m22 - m12 * m02) + m02 * (m01 * m12 - m11 * m02);
        if det.abs() < 1e-12 {
            return None;
        }
        let inv_det = 1.0 / det;
        let a00 = (m11 * m22 - m12 * m12) * inv_det;
        let a01 = (m02 * m12 - m01 * m22) * inv_det;
        let a02 = (m01 * m12 - m02 * m11) * inv_det;
        let a11 = (m00 * m22 - m02 * m02) * inv_det;
        let a12 = (m02 * m01 - m00 * m12) * inv_det;
        let a22 = (m00 * m11 - m01 * m01) * inv_det;
        let x = -(a00 * m03 + a01 * m13 + a02 * m23);
        let y = -(a01 * m03 + a11 * m13 + a12 * m23);
        let z = -(a02 * m03 + a12 * m13 + a22 * m23);
        if x.is_finite() && y.is_finite() && z.is_finite() { Some([x, y, z]) } else { None }
    }
}

/// Feature edges from the Feature Polygons input (PolyReduce 1.0 help): its polygons' edges; `creases` narrows them -
/// whitespace tokens `*` (all), `pA-B-C...` (an edge path), an edge group, or a primitive group (its polygons' edges).
/// Empty = every polygon edge of the input. Point numbers carry over to input 1 as-is.
fn feature_edges(f: &Detail, creases: &str) -> Result<Vec<[u32; 2]>, CookError> {
    let poly_edges = |pick: &dyn Fn(usize) -> bool| -> Vec<[u32; 2]> {
        let mut out = Vec::new();
        for (q, pr) in f.prims.iter().enumerate() {
            let PrimKind::Poly { closed } = pr.kind else { continue };
            if !pick(q) {
                continue;
            }
            let pts: Vec<u32> = pr.verts.clone().filter_map(|v| f.vertex_point.get(v as usize).copied()).collect();
            let n = pts.len();
            let segs = if closed && n > 2 { n } else { n.saturating_sub(1) };
            out.extend((0..segs).map(|k| [pts[k], pts[(k + 1) % n]]));
        }
        out
    };
    if creases.trim().is_empty() {
        return Ok(poly_edges(&|_| true));
    }
    let mut out = Vec::new();
    for tok in creases.split_whitespace() {
        if tok == "*" {
            out.extend(poly_edges(&|_| true));
        } else if let Some(path) =
            tok.strip_prefix('p').filter(|s| s.chars().next().is_some_and(|c| c.is_ascii_digit()))
        {
            let ids: Result<Vec<u32>, _> = path.split('-').map(str::parse::<u32>).collect();
            let ids = ids.map_err(|_| bad(format!("polyreduce: bad feature edge path `{tok}`")))?;
            out.extend(ids.windows(2).map(|w| [w[0], w[1]]));
        } else if let Some(g) = f.edge_group(tok) {
            out.extend(g.edges.iter().copied());
        } else if let Some(g) = f.group(hexe_geo::AttrClass::Prim, tok) {
            let mask = g.mask.clone();
            out.extend(poly_edges(&|q| mask.get(q).copied().unwrap_or(false)));
        } else {
            return Err(failed(
                OP,
                &format!("Features: no edge or primitive group `{tok}` on the Feature Polygons input"),
            ));
        }
    }
    Ok(out)
}

pub fn spec() -> Vec<ParamSpec> {
    vec![
        ParamSpec::new("target", ParamValue::Menu(0))
            .label("Target")
            .menu(&[("percentage", "Percentage"), ("finalcount", "Final Count")]),
        ParamSpec::new("percentage", ParamValue::F(50.0)).label("Percentage").range(0.0, 100.0),
        ParamSpec::new("finalcount", ParamValue::I(100)).label("Final Count"),
        ParamSpec::new("preserveboundary", ParamValue::B(true)).label("Preserve Boundary"),
        // PolyReduce 1.0 feature edges (Houdini parm names): `creases` picks them from input 2, `creaseweight` stiffens
        ParamSpec::new("creases", ParamValue::S(String::new())).label("Features"),
        ParamSpec::new("creaseweight", ParamValue::F(1.0)).label("Stiffen Features").range(0.0, 100.0),
    ]
}

pub fn new() -> Box<SopNode> {
    Box::new(PolyReduce { specs: spec() })
}

struct PolyReduce {
    specs: Vec<ParamSpec>,
}

struct Mesh {
    faces: Vec<Option<(Vec<u32>, Vec<usize>)>>,
    of_point: Vec<BTreeSet<usize>>,
}

impl Mesh {
    fn edge_faces(&self, a: u32, b: u32) -> Vec<usize> {
        self.of_point[a as usize]
            .iter()
            .copied()
            .filter(|&f| {
                self.faces[f].as_ref().is_some_and(|(c, _)| {
                    (0..c.len())
                        .any(|k| (c[k] == a && c[(k + 1) % c.len()] == b) || (c[k] == b && c[(k + 1) % c.len()] == a))
                })
            })
            .collect()
    }
    fn neighbours(&self, p: u32) -> BTreeSet<u32> {
        let mut n = BTreeSet::new();
        for &f in &self.of_point[p as usize] {
            if let Some((c, _)) = &self.faces[f] {
                for k in 0..c.len() {
                    if c[k] == p {
                        n.insert(c[(k + 1) % c.len()]);
                        n.insert(c[(k + c.len() - 1) % c.len()]);
                    }
                }
            }
        }
        n
    }
    fn on_boundary(&self, p: u32) -> bool {
        self.neighbours(p).into_iter().any(|q| self.edge_faces(p, q).len() == 1)
    }
    fn is_boundary_edge(&self, a: u32, b: u32) -> bool {
        self.edge_faces(a, b).len() == 1
    }
}

fn face_normal(pos: &[V3], corners: &[u32]) -> Option<[f64; 3]> {
    if corners.len() < 3 {
        return None;
    }
    let mut nx = 0.0;
    let mut ny = 0.0;
    let mut nz = 0.0;
    for k in 0..corners.len() {
        let i = corners[k] as usize;
        let j = corners[(k + 1) % corners.len()] as usize;
        let (pi, pj) = (pos[i], pos[j]);
        nx += (pi[1] - pj[1]) * (pi[2] + pj[2]);
        ny += (pi[2] - pj[2]) * (pi[0] + pj[0]);
        nz += (pi[0] - pj[0]) * (pi[1] + pj[1]);
    }
    let len = (nx * nx + ny * ny + nz * nz).sqrt();
    if len > 1e-12 { Some([nx / len, ny / len, nz / len]) } else { None }
}

type EdgeHeap = BinaryHeap<Reverse<(u64, u32, u32, u32, u32)>>;

impl Node<Detail> for PolyReduce {
    fn type_name(&self) -> &'static str {
        OP
    }
    /// Houdini: only the first input is required.
    fn min_inputs(&self) -> usize {
        1
    }
    fn inputs(&self) -> &[InputSpec] {
        &INPUTS
    }
    fn params(&self) -> &[ParamSpec] {
        &self.specs
    }
    fn cook(&self, ctx: &CookCtx, inputs: &[Arc<Detail>]) -> Result<Arc<Detail>, CookError> {
        let p = ctx.params;
        let target_mode = match p.menu("target") {
            Some(v) if (0..=1).contains(&v) => v,
            _ => return Err(bad("polyreduce: `target` must be 0 or 1")),
        };
        let pct = match p.float("percentage") {
            Some(v) if v.is_finite() && (0.0..=100.0).contains(&v) => f64::from(v),
            _ => return Err(bad("polyreduce: `percentage` must be finite in 0..=100")),
        };
        let final_count = match p.int("finalcount") {
            Some(v) if v >= 0 => v as usize,
            _ => return Err(bad("polyreduce: `finalcount` must be >= 0")),
        };
        let preserve_boundary = toggle(p, "preserveboundary")?;

        let Some(src) = inputs.first() else { return Ok(Arc::new(Detail::default())) };
        // feature edges (input 2), matched to input 1 by point number; penalised in the quadrics below
        let crease_w = match p.float("creaseweight") {
            Some(v) if v.is_finite() && v >= 0.0 => f64::from(v),
            _ => return Err(bad("polyreduce: `creaseweight` must be finite and >= 0")),
        };
        let features = match inputs.get(1).filter(|f| f.point_count > 0) {
            Some(f) => feature_edges(f, p.string("creases").unwrap_or(""))?,
            None => Vec::new(),
        };
        // wired = it changes Houdini's result: say so instead of silently reducing without it
        if inputs.get(2).is_some_and(|r| r.point_count > 0) {
            return Err(bad("polyreduce: Reference Mesh (input 3) is not modelled yet"));
        }
        let np = src.point_count as usize;
        let mut m = Mesh { faces: Vec::new(), of_point: vec![BTreeSet::new(); np] };
        let mut locked = vec![false; np];
        let mut face_of_prim: Vec<Option<usize>> = Vec::with_capacity(src.prims.len());

        for pr in &src.prims {
            let vs = (usize::try_from(pr.verts.start).ok(), usize::try_from(pr.verts.end).ok());
            let corners = match vs {
                (Some(lo), Some(hi)) => src.vertex_point.get(lo..hi).filter(|c| c.iter().all(|&i| (i as usize) < np)),
                _ => None,
            };
            match (&pr.kind, corners) {
                (PrimKind::Poly { closed: true }, Some(c)) if c.len() >= 3 => {
                    let f = m.faces.len();
                    for &i in c {
                        m.of_point[i as usize].insert(f);
                    }
                    m.faces.push(Some((c.to_vec(), (pr.verts.start as usize..pr.verts.end as usize).collect())));
                    face_of_prim.push(Some(f));
                }
                (_, _) => {
                    let (lo, hi) = (pr.verts.start as usize, (pr.verts.end as usize).min(src.vertex_point.len()));
                    for &i in src.vertex_point.get(lo..hi.max(lo)).unwrap_or(&[]) {
                        if (i as usize) < np {
                            locked[i as usize] = true;
                        }
                    }
                    face_of_prim.push(None);
                }
            }
        }

        let total = m.faces.len();
        let target =
            if target_mode == 0 { (total as f64 * pct / 100.0).round() as usize } else { final_count.min(total) };

        if total == 0 || target >= total {
            return Ok(Arc::clone(src));
        }

        let mut pos = positions(src);
        let mut alive = vec![true; np];
        let mut ver = vec![0u32; np];
        let mut merged: Vec<Vec<usize>> = (0..np).map(|i| vec![i]).collect();

        // Compute quadric matrices
        let mut q: Vec<Quadric> = vec![Quadric::default(); np];
        for (corners, _) in m.faces.iter().flatten() {
            if let Some(n) = face_normal(&pos, corners) {
                // Plane equation: n·(p - p0) = 0 → n·p - n·p0 = 0 → ax + by + cz + d = 0
                let p0 = pos[corners[0] as usize];
                let d = -(n[0] * p0[0] + n[1] * p0[1] + n[2] * p0[2]);
                let plane_q = Quadric::from_plane(n[0], n[1], n[2], d);
                for &ci in corners {
                    q[ci as usize] = q[ci as usize].add(&plane_q);
                }
            }
            // Boundary penalty
            for k in 0..corners.len() {
                let a = corners[k];
                let b = corners[(k + 1) % corners.len()];
                if m.is_boundary_edge(a, b) {
                    // Add boundary quadric (perpendicular to edge, passing through edge midpoint)
                    let pa = pos[a as usize];
                    let pb = pos[b as usize];
                    let mid = [(pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0, (pa[2] + pb[2]) / 2.0];
                    let edge_vec = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
                    let edge_len_sq = edge_vec[0] * edge_vec[0] + edge_vec[1] * edge_vec[1] + edge_vec[2] * edge_vec[2];
                    if edge_len_sq > 1e-12 {
                        // Boundary plane perpendicular to edge
                        let inv_len = 1.0 / edge_len_sq.sqrt();
                        let n = [edge_vec[0] * inv_len, edge_vec[1] * inv_len, edge_vec[2] * inv_len];
                        let d = -(n[0] * mid[0] + n[1] * mid[1] + n[2] * mid[2]);
                        let boundary_q = Quadric::from_plane(n[0], n[1], n[2], d).scale(1000.0);
                        q[a as usize] = q[a as usize].add(&boundary_q);
                        q[b as usize] = q[b as usize].add(&boundary_q);
                    }
                }
            }
        }

        // Feature edges (Stiffen Features): Garland-Heckbert constraint planes - for each face beside the edge, the plane
        // through the edge perpendicular to that face, weighted `creaseweight` x squared edge length, so collapses that
        // move the edge off its line cost more (weight 0 = no effect). Edges not in input 1's mesh are ignored.
        if crease_w > 0.0 {
            for &[a, b] in &features {
                let (Some(pa), Some(pb)) = (pos.get(a as usize).copied(), pos.get(b as usize).copied()) else {
                    continue;
                };
                let e = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
                let len2 = e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
                if len2 <= 1e-12 {
                    continue;
                }
                let shared: Vec<usize> = m
                    .of_point
                    .get(a as usize)
                    .into_iter()
                    .flatten()
                    .filter(|f| m.of_point.get(b as usize).is_some_and(|s| s.contains(f)))
                    .copied()
                    .collect();
                for f in shared {
                    let Some(Some((corners, _))) = m.faces.get(f) else { continue };
                    let Some(fnrm) = face_normal(&pos, corners) else { continue };
                    let c = [
                        e[1] * fnrm[2] - e[2] * fnrm[1],
                        e[2] * fnrm[0] - e[0] * fnrm[2],
                        e[0] * fnrm[1] - e[1] * fnrm[0],
                    ];
                    let cl = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
                    if cl <= 1e-12 {
                        continue;
                    }
                    let n = [c[0] / cl, c[1] / cl, c[2] / cl];
                    let d = -(n[0] * pa[0] + n[1] * pa[1] + n[2] * pa[2]);
                    let fq = Quadric::from_plane(n[0], n[1], n[2], d).scale(crease_w * len2);
                    q[a as usize] = q[a as usize].add(&fq);
                    q[b as usize] = q[b as usize].add(&fq);
                }
            }
        }

        let mut heap: EdgeHeap = BinaryHeap::new();
        let push_edges = |heap: &mut EdgeHeap, m: &Mesh, q: &[Quadric], pos: &[V3], ver: &[u32], pt: u32| {
            for nb in m.neighbours(pt) {
                let (a, b) = (pt.min(nb), pt.max(nb));
                let q_sum = q[a as usize].add(&q[b as usize]);
                let collapse_pos = q_sum.optimal().unwrap_or([
                    (pos[a as usize][0] + pos[b as usize][0]) / 2.0,
                    (pos[a as usize][1] + pos[b as usize][1]) / 2.0,
                    (pos[a as usize][2] + pos[b as usize][2]) / 2.0,
                ]);
                let cost = q_sum.error(collapse_pos[0], collapse_pos[1], collapse_pos[2]);
                let cost_bits = if cost.is_finite() && cost >= 0.0 { cost.to_bits() } else { u64::MAX };
                heap.push(Reverse((cost_bits, a, b, ver[a as usize], ver[b as usize])));
            }
        };

        for pt in 0..np as u32 {
            for nb in m.neighbours(pt).into_iter().filter(|&q| q > pt) {
                let q_sum = q[pt as usize].add(&q[nb as usize]);
                let collapse_pos = q_sum.optimal().unwrap_or([
                    (pos[pt as usize][0] + pos[nb as usize][0]) / 2.0,
                    (pos[pt as usize][1] + pos[nb as usize][1]) / 2.0,
                    (pos[pt as usize][2] + pos[nb as usize][2]) / 2.0,
                ]);
                let cost = q_sum.error(collapse_pos[0], collapse_pos[1], collapse_pos[2]);
                let cost_bits = if cost.is_finite() && cost >= 0.0 { cost.to_bits() } else { u64::MAX };
                heap.push(Reverse((cost_bits, pt, nb, 0, 0)));
            }
        }

        let mut live = total;
        while live > target {
            let Some(Reverse((_, a, b, va, vb))) = heap.pop() else { break };
            let (ai, bi) = (a as usize, b as usize);
            if !alive[ai] || !alive[bi] || ver[ai] != va || ver[bi] != vb || locked[ai] || locked[bi] {
                continue;
            }

            if preserve_boundary && (m.on_boundary(a) || m.on_boundary(b)) {
                continue;
            }

            let shared = m.edge_faces(a, b);
            if shared.is_empty() {
                continue;
            }

            // Link condition check
            let mut allowed: BTreeSet<u32> = BTreeSet::new();
            for &f in &shared {
                if let Some((c, _)) = &m.faces[f] {
                    allowed.extend(c.iter().copied());
                }
            }
            let (na, nb) = (m.neighbours(a), m.neighbours(b));
            if na.intersection(&nb).any(|c| !allowed.contains(c)) {
                continue;
            }

            // Compute collapse position
            let q_sum = q[ai].add(&q[bi]);
            let new_pos = q_sum.optimal().unwrap_or([
                (pos[ai][0] + pos[bi][0]) / 2.0,
                (pos[ai][1] + pos[bi][1]) / 2.0,
                (pos[ai][2] + pos[bi][2]) / 2.0,
            ]);

            // Normal flip test
            let mut flip_detected = false;
            for &f in &m.of_point[ai].union(&m.of_point[bi]).copied().collect::<Vec<_>>() {
                if let Some((c, _)) = &m.faces[f]
                    && (c.contains(&a) || c.contains(&b))
                {
                    // Compute original normal
                    if let Some(orig_n) = face_normal(&pos, c) {
                        // Compute new normal after collapse
                        let new_corners: Vec<u32> = c.iter().map(|&x| if x == b { a } else { x }).collect();
                        let mut temp_pos = pos.clone();
                        temp_pos[ai] = [new_pos[0], new_pos[1], new_pos[2]];
                        if let Some(new_n) = face_normal(&temp_pos, &new_corners) {
                            let dot = orig_n[0] * new_n[0] + orig_n[1] * new_n[1] + orig_n[2] * new_n[2];
                            if dot < 0.0 {
                                flip_detected = true;
                                break;
                            }
                        }
                    }
                }
            }
            if flip_detected {
                continue;
            }

            // Perform collapse
            pos[ai] = [new_pos[0], new_pos[1], new_pos[2]];
            q[ai] = q_sum;
            alive[bi] = false;
            ver[ai] += 1;
            let moved = std::mem::take(&mut merged[bi]);
            merged[ai].extend(moved);

            let touched: Vec<usize> = m.of_point[bi].iter().copied().collect();
            for f in touched {
                let Some((c, v)) = m.faces[f].take() else { continue };
                let (mut nc, mut nv): (Vec<u32>, Vec<usize>) = (Vec::new(), Vec::new());
                for (id, vx) in c.iter().zip(v.iter()) {
                    let id = if *id == b { a } else { *id };
                    if nc.last() != Some(&id) {
                        nc.push(id);
                        nv.push(*vx);
                    }
                }
                while nc.len() > 1 && nc.first() == nc.last() {
                    nc.pop();
                    nv.pop();
                }
                if nc.len() >= 3 {
                    m.of_point[ai].insert(f);
                    m.faces[f] = Some((nc, nv));
                } else {
                    live -= 1;
                    for id in c {
                        m.of_point[id as usize].remove(&f);
                    }
                }
            }
            m.of_point[bi].clear();
            push_edges(&mut heap, &m, &q, &pos, &ver, a);
        }

        // Output
        let mut new_id = vec![u32::MAX; np];
        let mut rm = Remap::default();
        let mut p_out: Vec<V3> = Vec::new();
        for i in 0..np {
            if alive[i] {
                new_id[i] = rm.point_src.len() as u32;
                rm.point_src.push(std::mem::take(&mut merged[i]));
                p_out.push(pos[i]);
            }
        }
        let mut vertex_point: Vec<u32> = Vec::new();
        let mut prims: Vec<Prim> = Vec::new();
        for (pi, pr) in src.prims.iter().enumerate() {
            let start = vertex_point.len() as u32;
            match face_of_prim[pi] {
                Some(f) => {
                    let Some((c, v)) = &m.faces[f] else { continue };
                    vertex_point.extend(c.iter().map(|&i| new_id[i as usize]));
                    rm.vert_src.extend(v.iter().map(|&x| Some(x)));
                }
                None => {
                    let (lo, hi) = (pr.verts.start as usize, pr.verts.end as usize);
                    for k in lo..hi.min(src.vertex_point.len()) {
                        let id = src.vertex_point[k];
                        vertex_point.push(new_id.get(id as usize).copied().unwrap_or(id));
                        rm.vert_src.push(Some(k));
                    }
                }
            }
            prims.push(Prim { kind: pr.kind.clone(), verts: start..vertex_point.len() as u32 });
            rm.prim_src.push(Some(pi));
        }
        let p32 = to_f32(OP, &p_out)?;
        if vertex_point.contains(&u32::MAX) {
            return Err(failed(OP, "internal: dangling point after collapse"));
        }
        Ok(Arc::new(rebuild(OP, src, vertex_point, prims, &rm, p32)?))
    }
}

#[cfg(test)]
mod tests {
    use super::super::meshutil::{testkit, tri_volume};
    use super::*;
    use crate::load_scene;

    use hexe_graph::ParamSet;

    fn run(d: Detail, p: ParamSet) -> Result<Arc<Detail>, CookError> {
        let n = new();
        let ctx = CookCtx { time: 0.0, fps: 24.0, params: &p, log: &|_| {} };
        n.cook(&ctx, &[Arc::new(d)])
    }
    fn defaults() -> ParamSet {
        testkit::defaults(&*new())
    }

    fn sphere_grid(rows: u32, cols: u32) -> Detail {
        let mut d = Detail::default();
        d.add_points((rows - 2) * cols + 2);
        let mut p = Vec::new();
        // Top pole
        p.push([0.0, 1.0, 0.0]);
        // Latitude rings
        for r in 1..rows - 1 {
            let theta = std::f32::consts::PI * r as f32 / (rows - 1) as f32;
            let y = theta.cos();
            let ring_r = theta.sin();
            for c in 0..cols {
                let phi = 2.0 * std::f32::consts::PI * c as f32 / cols as f32;
                let x = ring_r * phi.cos();
                let z = ring_r * phi.sin();
                p.push([x, y, z]);
            }
        }
        // Bottom pole
        p.push([0.0, -1.0, 0.0]);
        *d.p_mut() = p;

        // Top cap triangles (CCW from top)
        for c in 0..cols {
            let a = 0;
            let b = 1 + c;
            let c_next = 1 + (c + 1) % cols;
            d.add_poly(&[a, c_next, b], true);
        }
        // Middle quads (outward winding)
        for r in 0..rows - 3 {
            for c in 0..cols {
                let base = 1 + r * cols;
                let a = base + c;
                let b = base + (c + 1) % cols;
                let c_pt = base + cols + c;
                let d_pt = base + cols + (c + 1) % cols;
                d.add_poly(&[a, c_pt, d_pt, b], true);
            }
        }
        // Bottom cap triangles
        let bottom_pole = (rows - 2) * cols + 1;
        let last_ring = 1 + (rows - 3) * cols;
        for c in 0..cols {
            let a = last_ring + c;
            let b = last_ring + (c + 1) % cols;
            let c_pt = bottom_pole;
            d.add_poly(&[a, c_pt, b], true);
        }
        d
    }

    fn volume(d: &Detail) -> f64 {
        let pos = positions(d);
        d.prims
            .iter()
            .filter_map(|pr| {
                if pr.verts.len() < 3 {
                    return None;
                }
                let corners: Vec<usize> = d.vertex_point[pr.verts.start as usize..pr.verts.end as usize]
                    .iter()
                    .map(|&x| x as usize)
                    .collect();
                // Fan triangulation
                let mut v = 0.0;
                for k in 1..corners.len() - 1 {
                    v += tri_volume(pos[corners[0]], pos[corners[k]], pos[corners[k + 1]]);
                }
                Some(v)
            })
            .sum()
    }

    #[test]
    fn sphere_20x20_reduces_to_about_20_percent_within_one_prim() {
        let s = sphere_grid(20, 20);
        let orig_count = s.prim_count();
        let o = run(s, defaults().with("percentage", ParamValue::F(20.0))).expect("cooks");
        let target = (orig_count as f64 * 0.2).round() as usize;
        assert!(o.prim_count().abs_diff(target as u32) <= 1, "expected {} ±1, got {}", target, o.prim_count());
    }

    #[test]
    fn closed_mesh_euler_characteristic_preserved() {
        let s = sphere_grid(12, 12);
        let orig_v = s.point_count;
        let orig_f = s.prim_count();
        let orig_e = s.vertex_point.len() / 2; // Rough edge count for closed mesh
        let orig_chi = orig_v as i32 - orig_e as i32 + orig_f as i32;

        let o = run(s, defaults().with("percentage", ParamValue::F(50.0))).expect("cooks");
        let new_v = o.point_count;
        let new_f = o.prim_count();
        let new_e = o.vertex_point.len() / 2;
        let new_chi = new_v as i32 - new_e as i32 + new_f as i32;

        // For a closed sphere, χ = 2
        assert_eq!(orig_chi, 2, "original should be sphere topology");
        assert_eq!(new_chi, 2, "reduced should preserve topology");
    }

    #[test]
    fn open_grid_boundary_vertices_preserved() {
        let mut d = Detail::default();
        d.add_points(25);
        *d.p_mut() = (0..25).map(|i| [(i % 5) as f32, (i / 5) as f32, 0.0]).collect();
        for r in 0..4 {
            for c in 0..4 {
                let a = r * 5 + c;
                d.add_poly(&[a, a + 5, a + 6, a + 1], true);
            }
        }
        let orig_boundary = 16; // 5+5+5+5-4 corners

        let o =
            run(d, defaults().with("preserveboundary", ParamValue::B(true)).with("percentage", ParamValue::F(30.0)))
                .expect("cooks");
        let boundary_count = o.p().iter().filter(|p| p[0] == 0.0 || p[0] == 4.0 || p[1] == 0.0 || p[1] == 4.0).count();
        assert_eq!(boundary_count, orig_boundary, "boundary vertices should be preserved");
    }

    #[test]
    fn no_flipped_normals() {
        let s = sphere_grid(10, 10);
        let orig_vol = volume(&s);
        assert!(orig_vol > 0.0, "original should have positive volume");

        let o = run(s, defaults().with("percentage", ParamValue::F(25.0))).expect("cooks");
        let new_vol = volume(&o);
        assert!(new_vol > 0.0, "reduced should still have positive volume (no flips)");
    }

    #[test]
    fn finalcount_target_mode() {
        let s = sphere_grid(12, 12);
        let target = 50;
        let o = run(s, defaults().with("target", ParamValue::Menu(1)).with("finalcount", ParamValue::I(target)))
            .expect("cooks");
        assert!(o.prim_count().abs_diff(target as u32) <= 1, "expected {} ±1, got {}", target, o.prim_count());
    }

    #[test]
    fn percentage_100_shares_arc() {
        let s = sphere_grid(8, 8);
        let a = Arc::new(s);
        let n = new();
        let p = defaults().with("percentage", ParamValue::F(100.0));
        let ctx = CookCtx { time: 0.0, fps: 24.0, params: &p, log: &|_| {} };
        assert!(Arc::ptr_eq(&n.cook(&ctx, std::slice::from_ref(&a)).expect("cooks"), &a));
    }

    /// Stiffen Features keeps a feature line: a gently waved 9x9 grid with the middle row's 8 edges as the feature
    /// group; at weight 100 the points near that row drift off it >4x less than at weight 0 (= features off; input 2
    /// stays wired either way, so only `creaseweight` differs).
    #[test]
    fn feature_edges_hold_their_line() {
        let mut g = Detail::default();
        g.add_points(81);
        let pts: Vec<[f32; 3]> = (0..81)
            .map(|i| {
                let (x, z) = ((i % 9) as f32, (i / 9) as f32);
                [x, 0.6 * (x * 0.7).sin() * (z * 0.9).cos(), z]
            })
            .collect();
        *g.p_mut() = pts;
        for r in 0..8u32 {
            for c in 0..8u32 {
                let i = r * 9 + c;
                g.add_poly(&[i, i + 1, i + 10, i + 9], true);
            }
        }
        let row: Vec<u32> = (36..45).collect(); // z = 4
        let mut f = g.clone();
        f.set_edge_group("crease", row.windows(2).map(|w| [w[0], w[1]]));
        let (g, f) = (Arc::new(g), Arc::new(f));
        let n = new();
        let on_row = |d: &Detail| d.p().iter().filter(|q| (q[2] - 4.0).abs() < 1e-4).count();
        let off_row_near =
            |d: &Detail| d.p().iter().filter(|q| (q[2] - 4.0).abs() > 1e-4 && (q[2] - 4.0).abs() < 0.75).count();
        let base = defaults()
            .with("percentage", ParamValue::F(40.0))
            .with("preserveboundary", ParamValue::B(true))
            .with("creases", ParamValue::S("crease".into()))
            .with("creaseweight", ParamValue::F(0.0));
        let pf = base.clone().with("creaseweight", ParamValue::F(100.0));
        let ctx = CookCtx { time: 0.0, fps: 24.0, params: &pf, log: &|_| {} };
        let with = n.cook(&ctx, &[g.clone(), f.clone()]).expect("reduces with features");
        assert!(with.prims.len() < g.prims.len(), "still reduces the whole mesh");
        let bctx = CookCtx { time: 0.0, fps: 24.0, params: &base, log: &|_| {} };
        let without = n.cook(&bctx, &[g.clone(), f.clone()]).expect("reduces");
        assert_eq!(with.prims.len(), without.prims.len(), "same reduction target either way");
        // drift = how far the points near the line sit from it; features keep it far smaller
        let drift = |d: &Detail| d.p().iter().map(|q| (q[2] - 4.0).abs()).filter(|dz| *dz < 0.75).sum::<f32>();
        assert!(drift(&with) * 4.0 < drift(&without), "drift with {} vs without {}", drift(&with), drift(&without));
        assert!(on_row(&with) + off_row_near(&with) >= 2, "the line survives");
    }

    #[test]
    fn three_inputs_like_houdini_and_unmodelled_ones_say_so() {
        let names: Vec<_> = new().inputs().iter().map(|i| i.name).collect();
        assert_eq!(names, ["Triangular Mesh", "Feature Polygons", "Reference Mesh"]);
        let n = new();
        let p = defaults();
        let ctx = CookCtx { time: 0.0, fps: 24.0, params: &p, log: &|_| {} };
        let mesh = Arc::new(sphere_grid(8, 8));
        let empty = Arc::new(Detail::default());
        // unwired inputs 2 / 3 (fed empty geometry) reduce exactly as with one input
        let one = n.cook(&ctx, std::slice::from_ref(&mesh)).expect("one input");
        let three = n.cook(&ctx, &[mesh.clone(), empty.clone(), empty.clone()]).expect("empty extras");
        assert_eq!((one.point_count, one.prims.len()), (three.point_count, three.prims.len()));
        // an EMPTY feature group (your file: group1 has no edges) = plain reduction, same result as no features
        let mut tagged = (*mesh).clone();
        tagged.set_edge_group("group1", []);
        let tagged = Arc::new(tagged);
        let pf = p.clone().with("creases", ParamValue::S("group1".into())).with("creaseweight", ParamValue::F(10.0));
        let fctx = CookCtx { time: 0.0, fps: 24.0, params: &pf, log: &|_| {} };
        let none = n.cook(&fctx, &[tagged.clone(), tagged.clone(), empty.clone()]).expect("empty features");
        assert_eq!((none.p(), &none.prims), (one.p(), &one.prims));
        // unknown feature group = an error naming it
        let pbad = p.clone().with("creases", ParamValue::S("nope".into()));
        let bctx = CookCtx { time: 0.0, fps: 24.0, params: &pbad, log: &|_| {} };
        assert!(format!("{:?}", n.cook(&bctx, &[mesh.clone(), mesh.clone()]).expect_err("unknown")).contains("nope"));
        let refm = n.cook(&ctx, &[mesh.clone(), empty, mesh]).expect_err("reference not modelled");
        assert!(format!("{refm:?}").contains("Reference Mesh"), "{refm:?}");
    }

    #[test]
    fn hostile_params_and_geometry() {
        let s = sphere_grid(5, 5);
        for (k, v) in [
            ("percentage", ParamValue::F(f32::NAN)),
            ("percentage", ParamValue::F(-1.0)),
            ("percentage", ParamValue::F(101.0)),
            ("finalcount", ParamValue::I(-1)),
            ("target", ParamValue::Menu(2)),
        ] {
            assert!(matches!(run(s.clone(), defaults().with(k, v)), Err(CookError::BadParam(_))), "{}", k);
        }
        assert_eq!(run(Detail::default(), defaults()).expect("cooks").prim_count(), 0);
    }

    #[test]
    fn example_scene_cooks_and_registry_has_slot() {
        let ex = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/polyreduce.json"))
            .expect("example present");
        let (mut g, id) = load_scene(&ex).expect("example loads");
        let out = g.cook(id, 0.0).expect("example cooks");
        assert!(out.prim_count() > 0 && out.prim_count() < 200, "{}", out.prim_count());
        assert_eq!(crate::registry().create("polyreduce").map(|n| n.type_name()), Some("polyreduce"));
    }
}
