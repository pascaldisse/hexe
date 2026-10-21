//! Orient Along Curve SOP (token `orientalongcurve`): per-vertex frames along each polygon curve, written as point or
//! vertex attributes (axes / translation / quaternion / 3x3 / 4x4 transform) (Houdini 20.5 `orientalongcurve`).
//!
//! # Semantics (Houdini 20.5 HDK sample `SOP_OrientAlongCurve` / `GU_CurveFrame`, checked against hython recordings of
//! synthetic curves: `oracle/houdini/orientalongcurve/`, `fixtures/sop-orientalongcurve`)
//! Per selected prim (`group`, prim pattern): an open poly whose last vertex is its first point is "unrolled" (treated as
//! closed, its last vertex copies the first vertex's frame); one-vertex prims are open with no edges. Edges shorter than
//! `sqrt(1e-37)` count as zero length. Tangents (`tangenttype`): average of the unit edge directions (`avgdir`), central
//! difference (`diff`), previous / next edge, or no curve (`none`: identity frame). Open ends copy the end edge unless
//! `extrapolateendtangents`; runs of zero-length edges interpolate the bounding directions. Up vectors: the start up is
//! the target up (`upvectortype`: curve normal (largest-area fallback, else +Y), X / Y / Z axis, prim / detail
//! `upvectorattrib`, `upvector`) made perpendicular to the first tangent (parallel: an arbitrary perpendicular), then
//! carried by minimal rotations. `continuousclosed` spreads the closing twist of closed curves over the curve BY DISTANCE
//! (whatever `rollper` is: the shipped node differs from the HDK sample here, pinned by the recorded knot cases);
//! `useendupvector` does the same towards the end target up (open curves); `upvectoratstart` off rolls the whole curve so
//! the AVERAGE up best matches the target. Frame matrix rows = (up x tangent, up, tangent), pre-multiplied by the Euler
//! rotation of `pitch` / `yaw` / `roll` (+ increments per edge, per distance, per full curve, or times `rollattrib` / ...;
//! closed continuous curves round the total to whole turns, `fulltwists` adds turns of roll), post-multiplied by the
//! stretch-around-turns matrix and `scale`, translated to `P`. Houdini quirk kept: Euler step k rotates about axis
//! `rOrd[k]` by angle k (pitch, yaw, roll), so a non-`xyz` order also re-assigns angles to axes. Another kept quirk:
//! "per full curve by edges" on a one-point curve divides 0 by 0 and Houdini writes NaN frames.
//! Point outputs average the vertices of the point, walked newest first (unit directions with sign alignment, mean
//! length; quaternions with sign alignment; matrices by polar decomposition). Quaternions: trace >= 0 takes the w branch.
//! `upvectoratstart` off when the target up lies in a planar curve's plane is a sign knife-edge (the projection is 0
//! in exact arithmetic, +-1e-16 here): values inside `SIGN_EPS_REL` of zero take Houdini's `>= 0` branch, so the
//! recorded `y_avg` case matches on any FPU instead of depending on the last roundoff bit.
//!
//! Not modelled (-> `Unsupported`): `adjustupcurvature` (curvature attribute features), `transformbyattribs` when the input
//! carries instance attributes (N, up, pscale, scale, rot, orient, pivot, trans, transform), non-polygon prims (NURBS /
//! Bezier / quadrics), and an existing output attribute of a different tuple shape. The point-vertex walk order used for
//! averaging is vertex index order (Houdini's linked list order is only visible when the averaged values disagree in sign).
use super::group_sop::parse_pattern;
use super::meshutil::{V3, add, bad, cross, dot, failed, menu, scale, sub, toggle};
use crate::SopNode;
use hexe_geo::{AttrClass, AttrData, Attribute, Detail, PrimKind, TypeInfo};
use hexe_graph::{CookCtx, CookError, InputSpec, Node, ParamSet, ParamSpec, ParamValue};
use std::sync::Arc;

const OP: &str = "orientalongcurve";
const INPUTS: [InputSpec; 1] = [InputSpec { name: "input0" }];
const EXTREMELY_SMALL_L2: f64 = 1e-37;
const QUITE_SMALL_REL: f64 = 1e-5;
const QUITE_SMALL_REL2: f64 = 1e-10;
/// Relative zero for the average-up sign solve (see the module note): |projection| below this counts as 0.
const SIGN_EPS_REL: f64 = 1e-12;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Tangent {
    Circular,
    Subd,
    Prev,
    Next,
    None,
}
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Per {
    Edge,
    Distance,
    Attrib,
    FullEdges,
    FullDistance,
}
const PER_MENU: [(&str, &str); 5] = [
    ("edge", "Per Edge"),
    ("distance", "Per Unit Distance"),
    ("attrib", "Scale by Attribute"),
    ("fulledges", "Per Full Curve by Edges"),
    ("fulldistance", "Per Full Curve by Distance"),
];

pub fn spec() -> Vec<ParamSpec> {
    let s = |n: &str, v: &str| ParamSpec::new(n, ParamValue::S(v.into()));
    let b = |n: &str, v: bool| ParamSpec::new(n, ParamValue::B(v));
    let f = |n: &str, v: f32| ParamSpec::new(n, ParamValue::F(v));
    let per = |n: &str| ParamSpec::new(n, ParamValue::Menu(4)).menu(&PER_MENU);
    vec![
        s("group", "").label("Group"),
        ParamSpec::new("tangenttype", ParamValue::Menu(0)).label("Tangent Type").menu(&[
            ("avgdir", "Average of Edge Directions"),
            ("diff", "Central Difference"),
            ("prev", "Previous Edge"),
            ("next", "Next Edge"),
            ("none", "Z Axis (Ignore Curve)"),
        ]),
        b("continuousclosed", true).label("Make Closed Curve Orientations Continuous"),
        b("extrapolateendtangents", false).label("Extrapolate End Tangents"),
        b("transformbyattribs", false).label("Transform Using Curve Point Attributes"),
        ParamSpec::new("upvectortype", ParamValue::Menu(0)).label("Target Up Vector").menu(&[
            ("normal", "Curve Normal"),
            ("x", "X Axis"),
            ("y", "Y Axis"),
            ("z", "Z Axis"),
            ("attrib", "Attribute"),
            ("custom", "Custom"),
        ]),
        b("upvectoratstart", true).label("Target Up Vector at Start (else Average)"),
        b("useendupvector", false).label("Use Target End Up Vector"),
        s("upvectorattrib", "target_up").label("Start Up Attribute"),
        s("endupvectorattrib", "target_up_end").label("End Up Attribute"),
        ParamSpec::new("upvector", ParamValue::V3([0.0, 1.0, 0.0])).label("Start Up Vector"),
        ParamSpec::new("endupvector", ParamValue::V3([0.0, 1.0, 0.0])).label("End Up Vector"),
        b("adjustupcurvature", false).label("Adjust Up Vector Using Curvature"),
        f("curvaturescale", 1.0).label("Curvature Scale"),
        ParamSpec::new("enablecurvaturescaleattrib", ParamValue::Menu(0))
            .label("Curvature Scale Attribute")
            .menu(&[("noscaling", "No Scaling"), ("byattrib", "Scale by Attrib")]),
        s("curvaturescaleattrib", "speed").label("Curvature Scale Attrib"),
        b("enablecurvatureattrib", false).label("Output Curvature"),
        s("curvatureattrib", "curvature").label("Curvature Attribute"),
        ParamSpec::new("rOrd", ParamValue::Menu(0)).label("Rotate Order").menu(&[
            ("xyz", "Pitch, Yaw, Roll"),
            ("xzy", "Pitch, Roll, Yaw"),
            ("yxz", "Yaw, Pitch, Roll"),
            ("yzx", "Yaw, Roll, Pitch"),
            ("zxy", "Roll, Pitch, Yaw"),
            ("zyx", "Roll, Yaw, Pitch"),
        ]),
        b("applyroll", false).label("Apply Roll or Twist"),
        f("roll", 0.0).label("Roll"),
        per("rollper").label("Twist Per"),
        ParamSpec::new("fulltwists", ParamValue::I(0)).label("Full Twists"),
        f("incroll", 0.0).label("Partial Twist"),
        s("rollattrib", "roll").label("Twist Ramp Attribute"),
        b("applyyaw", false).label("Apply Yaw"),
        f("yaw", 0.0).label("Yaw"),
        per("yawper").label("Yaw Per"),
        f("incyaw", 0.0).label("Incremental Yaw"),
        s("yawattrib", "yaw").label("Yaw Ramp Attribute"),
        b("applypitch", false).label("Apply Pitch"),
        f("pitch", 0.0).label("Pitch"),
        per("pitchper").label("Pitch Per"),
        f("incpitch", 0.0).label("Incremental Pitch"),
        s("pitchattrib", "pitch").label("Pitch Ramp Attribute"),
        b("normalize", true).label("Normalize Scales"),
        f("scale", 1.0).label("Uniform Scale"),
        b("stretcharoundturns", false).label("Stretch Around Turns"),
        f("maxstretcharoundturns", 10.0).label("Max Stretch"),
        ParamSpec::new("class", ParamValue::Menu(0)).label("Class").menu(&[("point", "Point"), ("vertex", "Vertex")]),
        b("outputxaxis", false).label("Output X Axis"),
        s("xaxisname", "out").label("X Axis"),
        b("outputyaxis", true).label("Output Up Vector"),
        s("yaxisname", "up").label("Up Vector"),
        b("outputzaxis", true).label("Output Tangent (Z Axis)"),
        s("zaxisname", "N").label("Tangent (Z Axis)"),
        b("outputtranslation", false).label("Output Translation"),
        s("translationname", "P").label("Translation"),
        b("outputquaternion", false).label("Output Quaternion"),
        s("quaternionname", "orient").label("Quaternion"),
        b("outputtransform3", false).label("Output 3x3 Transform"),
        s("transform3name", "transform").label("3x3 Transform"),
        b("outputtransform4", false).label("Output 4x4 Transform"),
        s("transform4name", "transform").label("4x4 Transform"),
    ]
}

pub fn new() -> Box<SopNode> {
    Box::new(OrientAlongCurve { specs: spec() })
}

struct OrientAlongCurve {
    specs: Vec<ParamSpec>,
}

type M3 = [[f64; 3]; 3];
const IDENT: M3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

fn len2(v: V3) -> f64 {
    dot(v, v)
}
/// UT `normalize()`: unit vector (zero / non-finite length: unchanged) and the old length.
fn normalized(v: V3) -> (V3, f64) {
    let l = len2(v).sqrt();
    if l > 0.0 && l.is_finite() { (scale(v, 1.0 / l), l) } else { (v, l) }
}
fn neg(v: V3) -> V3 {
    scale(v, -1.0)
}
fn mul(a: &M3, b: &M3) -> M3 {
    let mut o = [[0.0; 3]; 3];
    for (i, row) in o.iter_mut().enumerate() {
        for (j, c) in row.iter_mut().enumerate() {
            *c = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    o
}
fn transpose(a: &M3) -> M3 {
    [[a[0][0], a[1][0], a[2][0]], [a[0][1], a[1][1], a[2][1]], [a[0][2], a[1][2], a[2][2]]]
}
fn det(m: &M3) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// UT `arbitraryPerp`: the axis of (ties-resolved) smallest |component| of `v`, made perpendicular to unit `v`.
fn arbitrary_perp(v: V3) -> V3 {
    let a = [v[0].abs(), v[1].abs(), v[2].abs()];
    let axis = if a[1] <= a[0] && a[1] <= a[2] {
        [0.0, 1.0, 0.0]
    } else if a[0] <= a[2] {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    normalized(sub(axis, scale(v, dot(axis, v)))).0
}
/// `other` made perpendicular to unit `dir` and normalised (too parallel: an arbitrary perpendicular).
fn perp_to(other: V3, dir: V3) -> V3 {
    let o = sub(other, scale(dir, dot(other, dir)));
    let l2 = len2(o);
    if l2 >= QUITE_SMALL_REL2 { scale(o, 1.0 / l2.sqrt()) } else { arbitrary_perp(dir) }
}
/// UT `dihedral(a, b)` applied to `v` (row vector): the minimal rotation taking unit `a` to unit `b`; `None` = opposed.
fn dihedral_apply(a: V3, b: V3, v: V3) -> Option<V3> {
    let (c, l) = normalized(add(a, b));
    if l.is_nan() || l <= 1e-12 {
        return None;
    }
    let v1 = sub(v, scale(a, 2.0 * dot(v, a)));
    Some(sub(v1, scale(c, 2.0 * dot(v1, c))))
}

struct Stretch {
    dir: V3,
    scale: f64,
}

#[allow(clippy::too_many_arguments)]
fn interpolate_tangent(
    tt: Tangent,
    stretch: bool,
    max_scale: f64,
    thr: f64,
    prev_edge: V3,
    next_edge: V3,
    prev_l2: f64,
    next_l2: f64,
    is_last: bool,
) -> (V3, Option<Stretch>) {
    let next_dir = if next_l2 != 1.0 { scale(next_edge, 1.0 / next_l2.sqrt()) } else { next_edge };
    let prev_dir = if prev_l2 != 1.0 { scale(prev_edge, 1.0 / prev_l2.sqrt()) } else { prev_edge };
    let mut mid_dir = [0.0; 3];
    let mut mid_len = 0.0;
    if tt == Tangent::Circular || stretch {
        mid_dir = add(prev_dir, next_dir);
        mid_len = len2(mid_dir).sqrt();
        if mid_len < QUITE_SMALL_REL {
            mid_dir = if is_last { prev_dir } else { next_dir };
        } else {
            mid_dir = scale(mid_dir, 1.0 / mid_len);
        }
    }
    let st = stretch.then(|| {
        if mid_len < thr {
            let dir = if mid_len < QUITE_SMALL_REL {
                if is_last { prev_dir } else { next_dir }
            } else {
                normalized(sub(next_dir, prev_dir)).0
            };
            Stretch { dir, scale: max_scale }
        } else {
            Stretch { dir: normalized(sub(next_dir, prev_dir)).0, scale: (2.0 / mid_len).min(max_scale) }
        }
    });
    let t = match tt {
        Tangent::Circular => mid_dir,
        Tangent::Prev => prev_dir,
        Tangent::Next => next_dir,
        Tangent::Subd => {
            let m = add(prev_edge, next_edge);
            let m2 = len2(m);
            if m2 < QUITE_SMALL_REL2 * next_l2 {
                if is_last { prev_dir } else { next_dir }
            } else {
                scale(m, 1.0 / m2.sqrt())
            }
        }
        Tangent::None => [0.0, 0.0, 1.0],
    };
    (t, st)
}

/// `(tangent, stretch)` at an open end; the tangent is `outer_dir` when the inner edge has zero length (no stretch then).
#[allow(clippy::too_many_arguments)]
fn extrapolate_end_tangent(
    tt: Tangent,
    stretch: bool,
    max_scale: f64,
    thr: f64,
    outer_edge: V3,
    inner_edge: V3,
    outer_dir: V3,
    is_last: bool,
) -> (V3, Option<Stretch>) {
    let inner_l2 = len2(inner_edge);
    if inner_l2 == 0.0 {
        return (outer_dir, None);
    }
    let inner_dir = scale(inner_edge, 1.0 / inner_l2.sqrt());
    let (ext, ext_l2, end, end_l2) = if tt != Tangent::Subd {
        (sub(scale(outer_dir, 2.0 * dot(inner_dir, outer_dir)), inner_dir), 1.0, outer_dir, 1.0)
    } else {
        let e = add(outer_edge, sub(outer_edge, inner_edge));
        (e, len2(e), outer_edge, len2(outer_edge))
    };
    if is_last {
        interpolate_tangent(tt, stretch, max_scale, thr, end, ext, end_l2, ext_l2, true)
    } else {
        interpolate_tangent(tt, stretch, max_scale, thr, ext, end, ext_l2, end_l2, false)
    }
}

/// Per-curve backbone result (`computeSingleBackboneFrames`).
struct Frames {
    rotate: bool,
    stretch: bool,
    dirs: Vec<V3>,
    tangents: Vec<V3>,
    ups: Vec<V3>,
    sdirs: Vec<V3>,
    sscales: Vec<f64>,
    twist_loop: f64,
}

/// Settings shared by every curve.
struct Cfg {
    tangent: Tangent,
    extrapolate: bool,
    use_normal: bool,
    at_start: bool,
    continuous: bool,
    use_end: bool,
    max_stretch: f64,
}

/// `pos`: the curve's vertex positions (`nverts`).
#[allow(clippy::too_many_arguments)]
fn backbone(
    cfg: &Cfg,
    pos: &[V3],
    nedges: usize,
    npoints: usize,
    closed: bool,
    rotate_in: bool,
    stretch_in: bool,
    need_dirs: bool,
    mut target_up: V3,
    end_up: V3,
) -> Frames {
    let nverts = pos.len();
    let mut f = Frames {
        rotate: rotate_in && nedges >= 1,
        stretch: stretch_in && npoints >= 3,
        dirs: Vec::new(),
        tangents: Vec::new(),
        ups: Vec::new(),
        sdirs: Vec::new(),
        sscales: Vec::new(),
        twist_loop: 0.0,
    };
    if !f.rotate && !f.stretch && !need_dirs {
        return f;
    }
    let tt = cfg.tangent;
    let thr = 2.0 / cfg.max_stretch;
    let max_s = cfg.max_stretch;
    let mut dirs = vec![[0.0; 3]; nedges];
    let mut max_l2: f64 = 0.0;
    let clean = |d: V3| if len2(d) < EXTREMELY_SMALL_L2 { [0.0; 3] } else { d };
    for i in 1..nverts {
        let d = clean(sub(pos[i], pos[i - 1]));
        if let Some(s) = dirs.get_mut(i - 1) {
            *s = d;
        }
        max_l2 = max_l2.max(len2(d));
    }
    if closed && nverts == nedges && nverts >= 1 {
        let d = clean(sub(pos[0], pos[nverts - 1]));
        if let Some(s) = dirs.get_mut(nverts - 1) {
            *s = d;
        }
        max_l2 = max_l2.max(len2(d));
    }
    f.dirs = dirs;
    if !f.rotate && !f.stretch {
        return f;
    }
    let dirs = f.dirs.clone();
    if max_l2 == 0.0 || dirs.is_empty() || !max_l2.is_finite() {
        f.rotate = false;
        f.stretch = false;
        return f;
    }
    let ne = dirs.len();
    let (mut first_nz, mut last_nz) = (0usize, ne - 1);
    while first_nz < last_nz && len2(dirs[first_nz]) == 0.0 {
        first_nz += 1;
    }
    while first_nz < last_nz && len2(dirs[last_nz]) == 0.0 {
        last_nz -= 1;
    }
    f.tangents = vec![[0.0; 3]; npoints];
    f.ups = vec![[0.0; 3]; npoints];
    if first_nz == last_nz {
        let dir = scale(dirs[first_nz], 1.0 / max_l2.sqrt());
        f.tangents.fill(dir);
        f.ups.fill(perp_to(target_up, dir));
        f.stretch = false;
        return f;
    }
    if f.stretch {
        f.sdirs = vec![[0.0; 3]; npoints];
        f.sscales = vec![1.0; npoints];
    }
    let stretch = f.stretch;
    let put_stretch = |f: &mut Frames, i: usize, s: Option<Stretch>| {
        let Some(s) = s else { return };
        if let (Some(d), Some(k)) = (f.sdirs.get_mut(i), f.sscales.get_mut(i)) {
            *d = s.dir;
            *k = s.scale;
        }
    };
    let first_dir = normalized(dirs[first_nz]).0;
    let last_dir = normalized(dirs[last_nz]).0;
    if !closed {
        if first_nz > 0 {
            for i in 0..=first_nz.min(npoints - 1) {
                f.tangents[i] = first_dir;
            }
        } else if cfg.extrapolate {
            let (t, s) = extrapolate_end_tangent(tt, stretch, max_s, thr, dirs[0], dirs[1], first_dir, false);
            f.tangents[0] = t;
            put_stretch(&mut f, 0, s);
        } else {
            f.tangents[0] = first_dir;
        }
        if last_nz < ne - 1 {
            for i in last_nz + 1..npoints {
                f.tangents[i] = last_dir;
            }
        } else if cfg.extrapolate {
            let (t, s) = extrapolate_end_tangent(tt, stretch, max_s, thr, dirs[ne - 1], dirs[ne - 2], last_dir, true);
            f.tangents[npoints - 1] = t;
            put_stretch(&mut f, npoints - 1, s);
        } else {
            f.tangents[npoints - 1] = last_dir;
        }
    } else if first_nz > 0 || last_nz < ne - 1 {
        let nsteps = first_nz + (ne - 1 - last_nz);
        let wrap: Vec<usize> = (last_nz + 1..npoints).chain(0..=first_nz).collect();
        match tt {
            Tangent::Circular | Tangent::Subd if first_nz == 1 && last_nz == ne - 1 => {
                f.tangents[0] = last_dir;
                f.tangents[1] = first_dir;
            }
            Tangent::Circular | Tangent::Subd if first_nz == 0 && last_nz == ne - 2 => {
                f.tangents[npoints - 1] = last_dir;
                f.tangents[0] = first_dir;
            }
            Tangent::Circular => {
                let c = dot(first_dir, last_dir);
                let axis = normalized(sub(first_dir, scale(last_dir, c))).0;
                let full = c.clamp(-1.0, 1.0).acos();
                for (j, &i) in wrap.iter().enumerate() {
                    let a = j as f64 * full / nsteps as f64;
                    f.tangents[i] = add(scale(last_dir, a.cos()), scale(axis, a.sin()));
                }
            }
            Tangent::Subd => {
                let mut prev_t = [0.0, 0.0, 1.0];
                for (j, &i) in wrap.iter().enumerate() {
                    let t = j as f64 / nsteps as f64;
                    let mut v = normalized(add(dirs[last_nz], scale(sub(dirs[first_nz], dirs[last_nz]), t))).0;
                    if len2(v) < 0.5 {
                        v = prev_t;
                    } else {
                        prev_t = v;
                    }
                    f.tangents[i] = v;
                }
            }
            Tangent::Prev | Tangent::Next | Tangent::None => {
                let t = if tt == Tangent::Prev { last_dir } else { first_dir };
                for &i in &wrap {
                    f.tangents[i] = t;
                }
            }
        }
    } else {
        let (t, s) = interpolate_tangent(
            tt,
            stretch,
            max_s,
            thr,
            dirs[ne - 1],
            dirs[0],
            len2(dirs[ne - 1]),
            len2(dirs[0]),
            false,
        );
        f.tangents[0] = t;
        put_stretch(&mut f, 0, s);
    }
    // middle points
    let mut prev_nz = first_nz;
    let mut prev_l2 = len2(dirs[first_nz]);
    let mut e = first_nz + 1;
    while e <= last_nz {
        let mut next_l2 = len2(dirs[e]);
        if next_l2 != 0.0 {
            let (t, s) = interpolate_tangent(tt, stretch, max_s, thr, dirs[prev_nz], dirs[e], prev_l2, next_l2, false);
            f.tangents[e] = t;
            put_stretch(&mut f, e, s);
            prev_nz = e;
            prev_l2 = next_l2;
            e += 1;
            continue;
        }
        loop {
            e += 1;
            next_l2 = len2(dirs[e]);
            if e == last_nz || next_l2 != 0.0 {
                break;
            }
        }
        let nsteps = e - prev_nz - 1;
        let pd = scale(dirs[prev_nz], 1.0 / prev_l2.sqrt());
        let nd = scale(dirs[e], 1.0 / next_l2.sqrt());
        match tt {
            Tangent::Circular | Tangent::Subd if nsteps == 1 => {
                f.tangents[e - 1] = pd;
                f.tangents[e] = nd;
            }
            Tangent::Circular => {
                let c = dot(nd, pd);
                let axis = normalized(sub(nd, scale(pd, c))).0;
                let full = c.clamp(-1.0, 1.0).acos();
                for (j, i) in (prev_nz + 1..=e).enumerate() {
                    let a = j as f64 * full / nsteps as f64;
                    f.tangents[i] = add(scale(pd, a.cos()), scale(axis, a.sin()));
                }
            }
            Tangent::Subd => {
                let mut prev_t = [0.0, 0.0, 1.0];
                for (j, i) in (prev_nz + 1..=e).enumerate() {
                    let t = j as f64 / nsteps as f64;
                    let mut v = normalized(add(dirs[prev_nz], scale(sub(dirs[e], dirs[prev_nz]), t))).0;
                    if len2(v) < 0.5 {
                        v = prev_t;
                    } else {
                        prev_t = v;
                    }
                    f.tangents[i] = v;
                }
            }
            Tangent::Prev | Tangent::Next | Tangent::None => {
                let t = if tt == Tangent::Prev { pd } else { nd };
                for i in prev_nz + 1..=e {
                    f.tangents[i] = t;
                }
            }
        }
        prev_nz = e;
        prev_l2 = next_l2;
        e += 1;
    }
    if cfg.use_normal {
        let mut sum = [0.0; 3];
        let mut max_n = [0.0; 3];
        let mut max_n2 = 0.0;
        let mut max_chord2: f64 = 0.0;
        for i in 1..nverts.saturating_sub(1) {
            let chord = sub(pos[i], pos[0]);
            sum = add(sum, cross(dirs.get(i).copied().unwrap_or([0.0; 3]), chord));
            let l2 = len2(sum);
            if l2 > max_n2 {
                max_n2 = l2;
                max_n = sum;
            }
            max_chord2 = max_chord2.max(len2(chord));
        }
        if let Some(&last) = pos.last() {
            max_chord2 = max_chord2.max(len2(sub(last, pos[0])));
        }
        let small2 = QUITE_SMALL_REL2 * max_chord2;
        let small4 = small2 * small2;
        if max_n2 >= small4 {
            let l2 = len2(sum);
            target_up = if l2 < small4 { scale(max_n, 1.0 / max_n2.sqrt()) } else { scale(sum, 1.0 / l2.sqrt()) };
        }
    }
    f.ups[0] = perp_to(target_up, f.tangents[0]);
    for i in 1..npoints {
        let prev = f.ups[i - 1];
        let u = dihedral_apply(f.tangents[i - 1], f.tangents[i], prev).unwrap_or_else(|| neg(prev));
        let t = f.tangents[i];
        f.ups[i] = normalized(sub(u, scale(t, dot(u, t)))).0;
    }
    let cont = closed && cfg.continuous;
    if cont || cfg.use_end {
        let (final_up, cmp_t) = if closed {
            let last = f.ups[npoints - 1];
            let u = dihedral_apply(f.tangents[npoints - 1], f.tangents[0], last).unwrap_or(last);
            let t0 = f.tangents[0];
            (normalized(sub(u, scale(t0, dot(u, t0)))).0, t0)
        } else {
            (f.ups[npoints - 1], f.tangents[npoints - 1])
        };
        let mut skip = false;
        let cmp_up = if cont {
            f.ups[0]
        } else {
            let c = if cfg.use_normal { target_up } else { end_up };
            let c = sub(c, scale(c, dot(c, cmp_t)));
            let l2 = len2(c);
            if l2 >= QUITE_SMALL_REL2 {
                scale(c, 1.0 / l2.sqrt())
            } else {
                skip = true;
                c
            }
        };
        let net = if skip { 0.0 } else { dot(cross(final_up, cmp_up), cmp_t).atan2(dot(final_up, cmp_up)) };
        if net.abs() > QUITE_SMALL_REL {
            let rot = |f: &mut Frames, i: usize, rev: f64| {
                let (s, c) = rev.sin_cos();
                let b = cross(f.ups[i], f.tangents[i]);
                f.ups[i] = add(scale(b, s), scale(f.ups[i], c));
            };
            // Houdini 20.5 spreads this twist by distance whatever `rollper` is (recorded knot cases: per-edge and
            // attribute rolls differ from the defaults by exactly whole turns x t); the HDK sample's per-edge /
            // attribute spreads are not what the shipped node does
            let total: f64 = dirs.iter().map(|d| len2(*d).sqrt()).sum();
            let mut cur = 0.0;
            for i in 1..npoints {
                cur += len2(dirs[i - 1]).sqrt();
                rot(&mut f, i, -cur * net / total);
            }
            if cont {
                f.twist_loop = -net;
            }
        }
    }
    if cfg.at_start {
        return f;
    }
    let (mut mx, mut my) = ([0.0; 3], [0.0; 3]);
    for i in 0..npoints {
        my = add(my, f.ups[i]);
        mx = add(mx, cross(f.ups[i], f.tangents[i]));
    }
    let (mxmx, mxmy, mymy) = (dot(mx, mx), dot(mx, my), dot(my, my));
    let (mxup, myup) = (dot(mx, target_up), dot(my, target_up));
    let det = mxmx * mymy - mxmy * mxmy;
    let scale2 = (mxmx * mxmx).max(mymy * mymy);
    let (s, c) = if det.abs() > 1e-6 * scale2 {
        let (s, c) = (mymy * mxup - mxmy * myup, mxmx * myup - mxmy * mxup);
        let n = s * s + c * c;
        if n.abs() <= EXTREMELY_SMALL_L2 {
            (0.0, 1.0)
        } else {
            let k = 1.0 / n.sqrt();
            (s * k, c * k)
        }
    } else if mxmx > mymy {
        // sign knife-edge: when the target up lies in a planar curve's plane the projection is zero in exact
        // arithmetic and +-1e-16 noise here, so anything inside the relative epsilon takes Houdini's `>= 0` branch
        (if mxup >= -SIGN_EPS_REL * mxmx.sqrt() { 1.0 } else { -1.0 }, 0.0)
    } else {
        (0.0, if myup >= -SIGN_EPS_REL * mymy.sqrt() { 1.0 } else { -1.0 })
    };
    for i in 0..npoints {
        let b = cross(f.ups[i], f.tangents[i]);
        f.ups[i] = add(scale(b, s), scale(f.ups[i], c));
    }
    f
}

/// UT `rotate(axis, angle)` (post-multiplied, row vectors).
fn axis_rot(axis: usize, a: f64) -> M3 {
    let (s, c) = a.sin_cos();
    match axis {
        0 => [[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]],
        1 => [[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]],
        _ => [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]],
    }
}

/// A rotation-angle attribute (`rollattrib` / ...): owner + first-component values.
enum RotAttr {
    Vertex(Vec<f64>),
    Point(Vec<f64>),
    Prim(Vec<f64>),
    Detail(f64),
}
fn first_comp(a: &AttrData) -> Option<Vec<f64>> {
    let col = |v: &[f32], s: usize| v.iter().step_by(s.max(1)).map(|&x| f64::from(x)).collect();
    Some(match a {
        AttrData::F32(v) => col(v, 1),
        AttrData::I32(v) => v.iter().map(|&x| f64::from(x)).collect(),
        AttrData::V2(v) => v.iter().map(|t| f64::from(t[0])).collect(),
        AttrData::V3(v) => v.iter().map(|t| f64::from(t[0])).collect(),
        AttrData::V4(v) => v.iter().map(|t| f64::from(t[0])).collect(),
        AttrData::FN { size, data } => col(data, *size as usize),
        AttrData::IN { size, data } => data.iter().step_by((*size).max(1) as usize).map(|&x| f64::from(x)).collect(),
        AttrData::Str { .. } => return None,
    })
}
fn find_rot_attr(d: &Detail, name: &str) -> Option<RotAttr> {
    for class in [AttrClass::Vertex, AttrClass::Point, AttrClass::Prim, AttrClass::Detail] {
        if let Some(v) = d.attr(class, name).and_then(|a| first_comp(&a.data)) {
            return Some(match class {
                AttrClass::Vertex => RotAttr::Vertex(v),
                AttrClass::Point => RotAttr::Point(v),
                AttrClass::Prim => RotAttr::Prim(v),
                AttrClass::Detail => RotAttr::Detail(v.first().copied().unwrap_or(0.0)),
            });
        }
    }
    None
}
fn v3_attr(d: &Detail, class: AttrClass, name: &str) -> Option<Vec<V3>> {
    match &d.attr(class, name)?.data {
        AttrData::V3(v) => Some(v.iter().map(|t| [f64::from(t[0]), f64::from(t[1]), f64::from(t[2])]).collect()),
        AttrData::FN { size, data } if *size >= 3 => {
            Some(data.chunks(*size as usize).map(|t| [f64::from(t[0]), f64::from(t[1]), f64::from(t[2])]).collect())
        }
        _ => None,
    }
}
fn unit_or_y(v: V3) -> V3 {
    let l2 = len2(v);
    if l2 < EXTREMELY_SMALL_L2 || !l2.is_finite() { [0.0, 1.0, 0.0] } else { scale(v, 1.0 / l2.sqrt()) }
}

/// A finished vertex transform: 3x3 (rows) + translation.
#[derive(Clone, Copy)]
struct Xf {
    m: M3,
    t: V3,
}

/// Polar decomposition `m = s * r` (Higham iteration); `None` when singular / non-finite.
fn polar(m: &M3) -> Option<(M3, M3)> {
    let mut r = *m;
    for _ in 0..64 {
        let d = det(&r);
        if !d.is_finite() || d.abs() <= 1e-300 {
            return None;
        }
        // inverse transpose = cofactor / det
        let cof = [
            [
                r[1][1] * r[2][2] - r[1][2] * r[2][1],
                r[1][2] * r[2][0] - r[1][0] * r[2][2],
                r[1][0] * r[2][1] - r[1][1] * r[2][0],
            ],
            [
                r[0][2] * r[2][1] - r[0][1] * r[2][2],
                r[0][0] * r[2][2] - r[0][2] * r[2][0],
                r[0][1] * r[2][0] - r[0][0] * r[2][1],
            ],
            [
                r[0][1] * r[1][2] - r[0][2] * r[1][1],
                r[0][2] * r[1][0] - r[0][0] * r[1][2],
                r[0][0] * r[1][1] - r[0][1] * r[1][0],
            ],
        ];
        let mut next = [[0.0; 3]; 3];
        let mut diff: f64 = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                next[i][j] = 0.5 * (r[i][j] + cof[i][j] / d);
                diff = diff.max((next[i][j] - r[i][j]).abs());
            }
        }
        r = next;
        if diff < 1e-15 {
            break;
        }
    }
    let s = mul(m, &transpose(&r));
    Some((s, r))
}
/// Rotation matrix (rows, row vectors) -> quaternion `[i, j, k, w]` (Houdini order).
fn quat_from_rot(m: &M3) -> [f64; 4] {
    let tr = m[0][0] + m[1][1] + m[2][2];
    if tr >= 0.0 {
        let s = (tr + 1.0).sqrt();
        let w = 0.5 * s;
        let k = 0.5 / s;
        [(m[1][2] - m[2][1]) * k, (m[2][0] - m[0][2]) * k, (m[0][1] - m[1][0]) * k, w]
    } else {
        let i = if m[1][1] > m[0][0] {
            if m[2][2] > m[1][1] { 2 } else { 1 }
        } else if m[2][2] > m[0][0] {
            2
        } else {
            0
        };
        let (j, k) = ((i + 1) % 3, (i + 2) % 3);
        let s = (m[i][i] - m[j][j] - m[k][k] + 1.0).sqrt();
        let mut q = [0.0; 4];
        q[i] = 0.5 * s;
        let h = if s != 0.0 { 0.5 / s } else { 0.0 };
        q[3] = (m[j][k] - m[k][j]) * h;
        q[j] = (m[i][j] + m[j][i]) * h;
        q[k] = (m[i][k] + m[k][i]) * h;
        q
    }
}
fn quat_to_rot(q: [f64; 4]) -> M3 {
    let [x, y, z, w] = q;
    [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)],
        [2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)],
        [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)],
    ]
}
fn qdot(a: [f64; 4], b: [f64; 4]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}
fn qnorm(q: [f64; 4]) -> [f64; 4] {
    let l = qdot(q, q).sqrt();
    if l > 0.0 && l.is_finite() { [q[0] / l, q[1] / l, q[2] / l, q[3] / l] } else { q }
}
/// `updateFromArbitraryMatrix`: rotation part (polar) -> quaternion.
fn quat_of(m: &M3) -> [f64; 4] {
    let r = polar(m).map_or(*m, |(_, r)| r);
    quat_from_rot(&r)
}
fn r32(v: f64) -> f64 {
    f64::from(v as f32)
}

/// Parameters resolved once per cook.
struct Run {
    cfg: Cfg,
    rotate_backbone: bool,
    stretch: bool,
    angles: V3,
    inc: V3,
    per: [Per; 3],
    rot_attr: [Option<RotAttr>; 3],
    order: [usize; 3],
    target_up: V3,
    end_up: V3,
    up_attr: Option<Vec<V3>>,
    end_up_attr: Option<Vec<V3>>,
    uniform: f64,
}

impl Run {
    /// Vertex transforms (len = the prim's vertex count) of one curve; `vpts` = point ids, `vids` = vertex ids.
    fn curve(&self, prim: usize, pos: &[V3], vpts: &[usize], vids: &[usize], poly_closed: bool) -> Vec<Xf> {
        let nverts = pos.len();
        if nverts == 0 {
            return Vec::new();
        }
        let (nedges, closed, unrolled) = if nverts == 1 {
            (0, false, false)
        } else if poly_closed {
            (nverts, true, false)
        } else {
            let u = vpts.first() == vpts.last();
            (nverts - 1, u, u)
        };
        let npoints = nedges + usize::from(!closed);
        let mut tup = self.target_up;
        let mut eup = self.end_up;
        if self.rotate_backbone {
            if let Some(v) = self.up_attr.as_ref().and_then(|a| a.get(prim)) {
                tup = unit_or_y(*v);
            }
            if let Some(v) = self.end_up_attr.as_ref().filter(|_| self.cfg.use_end).and_then(|a| a.get(prim)) {
                eup = unit_or_y(*v);
            }
        }
        let cont = closed && self.cfg.continuous;
        let rotate_params = len2(self.angles) != 0.0 || len2(self.inc) != 0.0;
        let mut per = self.per;
        let mut dang = self.inc;
        let (mut needs_len, mut needs_total) = (false, false);
        if rotate_params {
            needs_len = per.iter().any(|p| matches!(p, Per::Distance | Per::FullDistance));
            needs_total = per.iter().any(|p| *p == Per::FullDistance || (cont && *p == Per::Distance));
        }
        // varying (point / vertex) attribute value at vertex i of this curve
        let varying = |a: &RotAttr, i: usize| -> Option<f64> {
            match a {
                RotAttr::Vertex(v) => vids.get(i).and_then(|&k| v.get(k)).copied(),
                RotAttr::Point(v) => vpts.get(i).and_then(|&k| v.get(k)).copied(),
                _ => None,
            }
        };
        let f =
            backbone(&self.cfg, pos, nedges, npoints, closed, self.rotate_backbone, self.stretch, needs_len, tup, eup);
        let mut local = [0.0; 3];
        for ax in 0..3 {
            match &self.rot_attr[ax] {
                Some(RotAttr::Prim(v)) => local[ax] = dang[ax] * v.get(prim).copied().unwrap_or(0.0),
                Some(RotAttr::Detail(v)) => local[ax] = dang[ax] * v,
                _ => {}
            }
        }
        let total: f64 = if needs_total { f.dirs.iter().map(|d| len2(*d).sqrt()).sum() } else { 0.0 };
        let mut twist_loop = f.twist_loop;
        let rotate_dangles = len2(dang) != 0.0;
        if rotate_params && cont && rotate_dangles {
            for ax in 0..3 {
                if per[ax] == Per::Edge {
                    dang[ax] *= nedges as f64;
                    per[ax] = Per::FullEdges;
                } else if per[ax] == Per::Distance {
                    dang[ax] *= total;
                    per[ax] = Per::FullDistance;
                }
            }
            dang[2] -= twist_loop;
            twist_loop = 0.0;
            for d in &mut dang {
                *d = (*d / (2.0 * std::f64::consts::PI)).round_ties_even() * (2.0 * std::f64::consts::PI);
            }
        }
        let mut out = Vec::with_capacity(nverts);
        let mut cur_len = 0.0;
        for i in 0..npoints {
            let mut cur = add(self.angles, local);
            if rotate_dangles {
                for ax in 0..3 {
                    if ax != 2 && dang[ax] == 0.0 {
                        continue;
                    }
                    match per[ax] {
                        Per::Edge => {
                            cur[ax] += dang[ax] * i as f64;
                            if ax == 2 && twist_loop != 0.0 {
                                cur[2] -= twist_loop * (i as f64 / nedges as f64);
                            }
                        }
                        Per::FullEdges => {
                            let t = i as f64 / nedges as f64;
                            cur[ax] += dang[ax] * t;
                            if ax == 2 {
                                cur[2] -= twist_loop * t;
                            }
                        }
                        Per::Distance => {
                            cur[ax] += dang[ax] * cur_len;
                            if ax == 2 && total != 0.0 && twist_loop != 0.0 {
                                cur[2] -= twist_loop * (cur_len / total);
                            }
                        }
                        Per::FullDistance => {
                            if total != 0.0 {
                                let t = cur_len / total;
                                cur[ax] += t * dang[ax];
                                if ax == 2 {
                                    cur[2] -= twist_loop * t;
                                }
                            }
                        }
                        Per::Attrib => {
                            if let Some(t) = self.rot_attr[ax].as_ref().and_then(|a| varying(a, i)) {
                                cur[ax] += t * dang[ax];
                                if ax == 2 {
                                    cur[2] -= twist_loop * t;
                                }
                            }
                        }
                    }
                }
            }
            let mut m = IDENT;
            if rotate_params {
                // Houdini quirk (kept): step k rotates about axis `order[k]` by angle k (pitch, yaw, roll), so a
                // non-xyz order also swaps which angle drives which axis
                for (k, &ax) in self.order.iter().enumerate() {
                    let a = f64::from(cur[k] as f32);
                    if a != 0.0 {
                        m = mul(&m, &axis_rot(ax, a));
                    }
                }
            }
            if f.rotate {
                let z = f.tangents[i];
                let y = f.ups[i];
                m = mul(&m, &[cross(y, z), y, z]);
            }
            if f.stretch {
                let (d, k) = (f.sdirs[i], f.sscales[i] - 1.0);
                let mut s = IDENT;
                for (r, row) in s.iter_mut().enumerate() {
                    for (c, v) in row.iter_mut().enumerate() {
                        *v += k * d[r] * d[c];
                    }
                }
                m = mul(&m, &s);
            }
            if self.uniform != 1.0 {
                for row in &mut m {
                    for v in row.iter_mut() {
                        *v *= self.uniform;
                    }
                }
            }
            out.push(Xf { m, t: pos.get(i).copied().unwrap_or([0.0; 3]) });
            if needs_len && i + 1 < npoints {
                cur_len += f.dirs.get(i).map_or(0.0, |d| len2(*d).sqrt());
            }
        }
        if let (true, Some(&x0)) = (unrolled, out.first()) {
            out.push(x0);
        }
        // closed / unrolled curves have npoints == nverts (- 1 for unrolled); never more than nverts
        out.truncate(nverts);
        out
    }
}

/// What one output attribute extracts from a transform.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Out {
    Axis(usize),
    Trans,
    Quat,
    M3,
    M4,
}
impl Out {
    fn size(self) -> usize {
        match self {
            Out::Axis(_) | Out::Trans => 3,
            Out::Quat => 4,
            Out::M3 => 9,
            Out::M4 => 16,
        }
    }
}

/// Per-vertex f32-rounded value of `kind` (Houdini's intermediate attribute storage is float32).
fn extract(kind: Out, x: &Xf) -> Vec<f64> {
    let v = match kind {
        Out::Axis(a) => x.m[a].to_vec(),
        Out::Trans => x.t.to_vec(),
        Out::Quat => quat_of(&x.m).to_vec(),
        Out::M3 => x.m.iter().flatten().copied().collect(),
        Out::M4 => {
            let mut o = Vec::with_capacity(16);
            for r in &x.m {
                o.extend_from_slice(r);
                o.push(0.0);
            }
            o.extend_from_slice(&x.t);
            o.push(1.0);
            o
        }
    };
    v.into_iter().map(r32).collect()
}

fn m3_of(v: &[f64]) -> M3 {
    let g = |k: usize| v.get(k).copied().unwrap_or(0.0);
    [[g(0), g(1), g(2)], [g(3), g(4), g(5)], [g(6), g(7), g(8)]]
}
fn m3_of_m4(v: &[f64]) -> M3 {
    let g = |k: usize| v.get(k).copied().unwrap_or(0.0);
    [[g(0), g(1), g(2)], [g(4), g(5), g(6)], [g(8), g(9), g(10)]]
}

/// Point value from the (f32-rounded) values of its vertices, Houdini's averaging per output kind.
fn average(kind: Out, vals: &[Vec<f64>]) -> Option<Vec<f64>> {
    let first = vals.first()?;
    if vals.len() == 1 {
        return Some(first.clone());
    }
    let n = vals.len() as f64;
    match kind {
        Out::Axis(_) | Out::Trans => {
            let v3 = |v: &Vec<f64>| [v[0], v[1], v[2]];
            let (mut sum, mut lsum) = normalized(v3(first));
            for v in &vals[1..] {
                let (mut d, l) = normalized(v3(v));
                lsum += l;
                if dot(sum, d) < 0.0 {
                    d = neg(d);
                }
                sum = add(sum, d);
            }
            Some(scale(normalized(sum).0, lsum / n).to_vec())
        }
        Out::Quat => {
            let q4 = |v: &Vec<f64>| [v[0], v[1], v[2], v[3]];
            let mut sum = q4(first);
            for v in &vals[1..] {
                let mut q = q4(v);
                if qdot(sum, q) < 0.0 {
                    q = [-q[0], -q[1], -q[2], -q[3]];
                }
                sum = [sum[0] + q[0], sum[1] + q[1], sum[2] + q[2], sum[3] + q[3]];
            }
            Some(qnorm(sum).to_vec())
        }
        Out::M3 | Out::M4 => {
            let get = |v: &Vec<f64>| if kind == Out::M3 { m3_of(v) } else { m3_of_m4(v) };
            let (s0, r0) = polar(&get(first)).unwrap_or((IDENT, get(first)));
            let mut ssum = s0;
            let mut qsum = quat_from_rot(&r0);
            for v in &vals[1..] {
                let (s, r) = polar(&get(v)).unwrap_or((IDENT, get(v)));
                for (a, b) in ssum.iter_mut().flatten().zip(s.iter().flatten()) {
                    *a += b;
                }
                let mut q = quat_from_rot(&r);
                if qdot(qsum, q) < 0.0 {
                    q = [-q[0], -q[1], -q[2], -q[3]];
                }
                qsum = [qsum[0] + q[0], qsum[1] + q[1], qsum[2] + q[2], qsum[3] + q[3]];
            }
            for a in ssum.iter_mut().flatten() {
                *a /= n;
            }
            let m = mul(&ssum, &quat_to_rot(qnorm(qsum)));
            let mut o: Vec<f64> = Vec::with_capacity(kind.size());
            if kind == Out::M3 {
                o.extend(m.iter().flatten());
            } else {
                for r in &m {
                    o.extend_from_slice(r);
                    o.push(0.0);
                }
                // Houdini sums the FIRST vertex's translation n times
                o.extend_from_slice(first.get(12..15).unwrap_or(&[0.0; 3]));
                o.push(1.0);
            }
            Some(o)
        }
    }
}

fn per_of(p: &ParamSet, n: &str) -> Result<Per, CookError> {
    Ok(match menu(p, n)? {
        0 => Per::Edge,
        1 => Per::Distance,
        2 => Per::Attrib,
        3 => Per::FullEdges,
        4 => Per::FullDistance,
        m => return Err(bad(format!("`{n}`: menu index {m} out of range"))),
    })
}
fn fin(p: &ParamSet, n: &str) -> Result<f64, CookError> {
    match p.float(n) {
        Some(v) if v.is_finite() => Ok(f64::from(v)),
        _ => Err(bad(format!("`{n}`: missing or not a finite float"))),
    }
}
fn vec3(p: &ParamSet, n: &str) -> Result<V3, CookError> {
    match p.vec3(n) {
        Some(v) if v.iter().all(|c| c.is_finite()) => Ok([f64::from(v[0]), f64::from(v[1]), f64::from(v[2])]),
        _ => Err(bad(format!("`{n}`: missing or not a finite vector"))),
    }
}
fn valid_name(n: &str) -> bool {
    let mut c = n.chars();
    n.len() <= 256
        && c.next().is_some_and(|f| f.is_ascii_alphabetic() || f == '_')
        && c.all(|k| k.is_ascii_alphanumeric() || k == '_')
}

impl Node<Detail> for OrientAlongCurve {
    fn type_name(&self) -> &'static str {
        OP
    }
    fn inputs(&self) -> &[InputSpec] {
        &INPUTS
    }
    fn params(&self) -> &[ParamSpec] {
        &self.specs
    }
    fn cook(&self, ctx: &CookCtx, inputs: &[Arc<Detail>]) -> Result<Arc<Detail>, CookError> {
        let p = ctx.params;
        let st = |n: &str| p.string(n).ok_or_else(|| bad(format!("`{n}`: missing or not a string")));
        let tangent = match menu(p, "tangenttype")? {
            0 => Tangent::Circular,
            1 => Tangent::Subd,
            2 => Tangent::Prev,
            3 => Tangent::Next,
            4 => Tangent::None,
            m => return Err(bad(format!("`tangenttype`: menu index {m} out of range"))),
        };
        let up_type = menu(p, "upvectortype")?;
        if up_type > 5 {
            return Err(bad(format!("`upvectortype`: menu index {up_type} out of range")));
        }
        let order = match menu(p, "rOrd")? {
            0 => [0, 1, 2],
            1 => [0, 2, 1],
            2 => [1, 0, 2],
            3 => [1, 2, 0],
            4 => [2, 0, 1],
            5 => [2, 1, 0],
            m => return Err(bad(format!("`rOrd`: menu index {m} out of range"))),
        };
        let vertex_class = match menu(p, "class")? {
            0 => false,
            1 => true,
            m => return Err(bad(format!("`class`: menu index {m} out of range"))),
        };
        if menu(p, "enablecurvaturescaleattrib")? > 1 {
            return Err(bad("`enablecurvaturescaleattrib`: menu index out of range"));
        }
        let per = [per_of(p, "pitchper")?, per_of(p, "yawper")?, per_of(p, "rollper")?];
        let (ap, ay, ar) = (toggle(p, "applypitch")?, toggle(p, "applyyaw")?, toggle(p, "applyroll")?);
        let deg = |on: bool, v: f64| if on { v.to_radians() } else { 0.0 };
        let angles = [deg(ap, fin(p, "pitch")?), deg(ay, fin(p, "yaw")?), deg(ar, fin(p, "roll")?)];
        let twists = f64::from(p.int("fulltwists").ok_or_else(|| bad("`fulltwists`: missing or not an int"))?);
        let inc =
            [deg(ap, fin(p, "incpitch")?), deg(ay, fin(p, "incyaw")?), deg(ar, fin(p, "incroll")? + 360.0 * twists)];
        let up_custom = vec3(p, "upvector")?;
        let end_custom = vec3(p, "endupvector")?;
        let uniform = fin(p, "scale")?;
        let max_stretch = fin(p, "maxstretcharoundturns")?;
        let (at_start, use_end_on) = (toggle(p, "upvectoratstart")?, toggle(p, "useendupvector")?);
        let (continuous, extrapolate) = (toggle(p, "continuousclosed")?, toggle(p, "extrapolateendtangents")?);
        let stretch = toggle(p, "stretcharoundturns")?;
        let _ = (toggle(p, "normalize")?, fin(p, "curvaturescale")?, toggle(p, "enablecurvatureattrib")?);
        let pat = st("group")?;
        let outs_spec = [
            ("outputxaxis", "xaxisname", Out::Axis(0)),
            ("outputyaxis", "yaxisname", Out::Axis(1)),
            ("outputzaxis", "zaxisname", Out::Axis(2)),
            ("outputtranslation", "translationname", Out::Trans),
            ("outputquaternion", "quaternionname", Out::Quat),
            ("outputtransform3", "transform3name", Out::M3),
            ("outputtransform4", "transform4name", Out::M4),
        ];
        let mut outs: Vec<(String, Out)> = Vec::new();
        for (on, nm, kind) in outs_spec {
            let name = st(nm)?;
            if !toggle(p, on)? || name.is_empty() {
                continue;
            }
            if !valid_name(name) {
                (ctx.log)(&format!("{OP}: invalid attribute name `{name}` skipped"));
                continue;
            }
            if vertex_class && name == "P" {
                (ctx.log)(&format!("{OP}: unable to create a vertex attribute named `P`"));
                continue;
            }
            outs.push((name.to_string(), kind));
        }
        let Some(src) = inputs.first() else { return Ok(Arc::new(Detail::default())) };
        if toggle(p, "adjustupcurvature")? {
            return Err(CookError::Unsupported("orientalongcurve: `adjustupcurvature` is not modelled".into()));
        }
        if toggle(p, "transformbyattribs")? {
            let inst = ["N", "up", "pscale", "scale", "rot", "orient", "pivot", "trans", "transform"];
            if inst.iter().any(|n| src.attr(AttrClass::Point, n).is_some()) {
                return Err(CookError::Unsupported(
                    "orientalongcurve: `transformbyattribs` with instance attributes on the points is not modelled"
                        .into(),
                ));
            }
        }
        let sel = parse_pattern(src, AttrClass::Prim, pat, OP)?;
        let picked = |i: usize| sel.as_ref().is_none_or(|m| m.get(i).copied().unwrap_or(false));
        let rotate_backbone = tangent != Tangent::None;
        // target up vectors
        let mut up_attr = None;
        let mut target_up = match up_type {
            1 => [1.0, 0.0, 0.0],
            3 => [0.0, 0.0, 1.0],
            5 => up_custom,
            4 => {
                let name = st("upvectorattrib")?;
                if let Some(v) = v3_attr(src, AttrClass::Prim, name) {
                    up_attr = Some(v);
                    [0.0, 1.0, 0.0]
                } else if let Some(v) = v3_attr(src, AttrClass::Detail, name).and_then(|v| v.first().copied()) {
                    v
                } else {
                    (ctx.log)(&format!("{OP}: target up vector attribute not found; defaulting to y axis"));
                    [0.0, 1.0, 0.0]
                }
            }
            _ => [0.0, 1.0, 0.0],
        };
        let mut use_end = at_start && use_end_on;
        let mut end_up_attr = None;
        let mut end_up = [0.0, 1.0, 0.0];
        if use_end {
            end_up = match up_type {
                4 => {
                    let name = st("endupvectorattrib")?;
                    if let Some(v) = v3_attr(src, AttrClass::Prim, name) {
                        end_up_attr = Some(v);
                        [0.0, 1.0, 0.0]
                    } else if let Some(v) = v3_attr(src, AttrClass::Detail, name).and_then(|v| v.first().copied()) {
                        v
                    } else {
                        (ctx.log)(&format!("{OP}: end target up vector attribute not found"));
                        use_end = false;
                        [0.0, 1.0, 0.0]
                    }
                }
                5 => end_custom,
                _ => target_up,
            };
        }
        if rotate_backbone {
            target_up = unit_or_y(target_up);
            if use_end {
                end_up = unit_or_y(end_up);
            }
        }
        let attr_names = ["pitchattrib", "yawattrib", "rollattrib"];
        let mut rot_attr: [Option<RotAttr>; 3] = [None, None, None];
        for ax in 0..3 {
            if inc[ax] != 0.0 && per[ax] == Per::Attrib {
                rot_attr[ax] = find_rot_attr(src, st(attr_names[ax])?);
            }
        }
        let run = Run {
            cfg: Cfg { tangent, extrapolate, use_normal: up_type == 0, at_start, continuous, use_end, max_stretch },
            rotate_backbone,
            stretch,
            angles,
            inc,
            per,
            rot_attr,
            order,
            target_up,
            end_up,
            up_attr,
            end_up_attr,
            uniform,
        };
        // transforms per vertex
        let np = src.point_count as usize;
        let nv = src.vertex_point.len();
        let pos: Vec<V3> = src.p().iter().map(|q| [f64::from(q[0]), f64::from(q[1]), f64::from(q[2])]).collect();
        let mut xf: Vec<Option<Xf>> = vec![None; nv];
        let mut pt_verts: Vec<Vec<usize>> = vec![Vec::new(); np];
        for (i, pr) in src.prims.iter().enumerate() {
            if !picked(i) {
                continue;
            }
            let poly_closed = match pr.kind {
                PrimKind::Poly { closed } => closed,
                PrimKind::Polyline => false,
                _ => {
                    return Err(CookError::Unsupported(format!(
                        "orientalongcurve: prim {i} is not a polygon curve (NURBS / Bezier / other prims are not modelled)"
                    )));
                }
            };
            let (lo, hi) = (pr.verts.start as usize, pr.verts.end as usize);
            if lo > hi || hi > nv {
                return Err(failed(OP, &format!("prim {i}: vertex range out of bounds")));
            }
            let vids: Vec<usize> = (lo..hi).collect();
            let vpts: Vec<usize> = vids.iter().map(|&v| src.vertex_point[v] as usize).collect();
            if vpts.iter().any(|&q| q >= np || q >= pos.len()) {
                return Err(failed(OP, &format!("prim {i}: point id out of range")));
            }
            let cpos: Vec<V3> = vpts.iter().map(|&q| pos[q]).collect();
            for (k, x) in run.curve(i, &cpos, &vpts, &vids, poly_closed).into_iter().enumerate() {
                if let Some(&v) = vids.get(k) {
                    xf[v] = Some(x);
                    pt_verts[vpts[k]].push(v);
                }
            }
        }
        let mut out = (**src).clone();
        let class = if vertex_class { AttrClass::Vertex } else { AttrClass::Point };
        let n = if vertex_class { nv } else { np };
        for (name, kind) in outs {
            let size = kind.size();
            let existing = src.attr(class, &name).map(|a| &a.data);
            let mut data: Vec<f32> = match (existing, size) {
                (None, _) => vec![0.0; n * size],
                (Some(AttrData::V3(v)), 3) => v.iter().flatten().copied().collect(),
                (Some(AttrData::V4(v)), 4) => v.iter().flatten().copied().collect(),
                (Some(AttrData::FN { size: s, data }), _) if *s as usize == size => data.clone(),
                _ => {
                    return Err(CookError::Unsupported(format!(
                        "orientalongcurve: existing {class:?} attribute `{name}` has another tuple shape"
                    )));
                }
            };
            data.resize(n * size, 0.0);
            let mut write = |e: usize, v: &[f64]| {
                for (c, x) in v.iter().enumerate().take(size) {
                    if let Some(s) = data.get_mut(e * size + c) {
                        *s = *x as f32;
                    }
                }
            };
            if vertex_class {
                for (v, x) in xf.iter().enumerate() {
                    if let Some(x) = x {
                        write(v, &extract(kind, x));
                    }
                }
            } else {
                for (pt, vs) in pt_verts.iter().enumerate() {
                    // Houdini walks a point's vertices newest first (reverse vertex order)
                    let vals: Vec<Vec<f64>> =
                        vs.iter().rev().filter_map(|&v| xf[v].as_ref()).map(|x| extract(kind, x)).collect();
                    if let Some(v) = average(kind, &vals) {
                        write(pt, &v);
                    }
                }
            }
            let (data, type_info) = match kind {
                Out::Axis(_) => (
                    AttrData::V3(data.chunks(3).map(|c| [c[0], c[1], c[2]]).collect()),
                    if name == "N" { TypeInfo::Normal } else { TypeInfo::Vector },
                ),
                Out::Trans => (AttrData::V3(data.chunks(3).map(|c| [c[0], c[1], c[2]]).collect()), TypeInfo::Point),
                Out::Quat => {
                    (AttrData::V4(data.chunks(4).map(|c| [c[0], c[1], c[2], c[3]]).collect()), TypeInfo::Quaternion)
                }
                Out::M3 | Out::M4 => (AttrData::FN { size: size as u32, data }, TypeInfo::None),
            };
            out.add_attr(Attribute { name, class, data, type_info });
        }
        Ok(Arc::new(out))
    }
}

#[cfg(test)]
mod tests {
    use super::super::meshutil::testkit::{cook, defaults};
    use super::*;

    fn run(d: Detail, ov: &[(&str, ParamValue)]) -> Result<Arc<Detail>, CookError> {
        let n = new();
        let p = ov.iter().fold(defaults(&*n), |p, (k, v)| p.with(k, v.clone()));
        cook(&*n, d, &p)
    }
    fn curve(pts: &[[f32; 3]], closed: bool) -> Detail {
        let mut d = Detail::default();
        d.add_points(pts.len() as u32);
        *d.p_mut() = pts.to_vec();
        let ids: Vec<u32> = (0..pts.len() as u32).collect();
        d.add_poly(&ids, closed);
        d
    }
    fn v3s(d: &Detail, c: AttrClass, n: &str) -> Vec<[f32; 3]> {
        match d.attr(c, n).map(|a| &a.data) {
            Some(AttrData::V3(v)) => v.clone(),
            _ => Vec::new(),
        }
    }
    fn flat(a: &AttrData) -> Vec<f32> {
        match a {
            AttrData::F32(v) => v.clone(),
            AttrData::V2(v) => v.iter().flatten().copied().collect(),
            AttrData::V3(v) => v.iter().flatten().copied().collect(),
            AttrData::V4(v) => v.iter().flatten().copied().collect(),
            AttrData::FN { data, .. } => data.clone(),
            _ => Vec::new(),
        }
    }

    #[test]
    fn straight_segment_and_arbitrary_perp_match_the_probes() {
        // hython probe: a segment along the custom up vector falls back to UT arbitraryPerp
        for (d, up) in [
            ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
            ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
            ([1.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
            ([1.0, 2.0, 3.0], [0.963_624_1, -0.148_249_86, -0.222_374_8]),
            ([-3.0, 1.0, 2.0], [0.222_374_8, 0.963_624_1, -0.148_249_86]),
            ([1.0, 1.0, 1.0], [-0.408_248_3, 0.816_496_6, -0.408_248_3]),
            ([2.0, 1.0, -0.5], [0.195_180_01, 0.097_590_01, 0.975_900_05]),
        ] {
            let o = run(
                curve(&[[0.0; 3], d], false),
                &[("upvectortype", ParamValue::Menu(5)), ("upvector", ParamValue::V3(d))],
            )
            .expect("cooks");
            let got = v3s(&o, AttrClass::Point, "up")[0];
            assert!(got.iter().zip(up).all(|(a, b)| (a - b).abs() < 1e-5), "{d:?}: {got:?} vs {up:?}");
        }
    }

    /// Synthetic hython cases (`oracle/houdini/orientalongcurve/sop_orientalongcurve_cases.py`): one input with helix,
    /// zigzag, closed circle, line, coincident points, unrolled loop, closed knot, closed square with a coincident
    /// closing point, a single point, a segment and two curves sharing a point; every case's every point / vertex float
    /// attribute equals Houdini 20.5's within 1e-5 (the attribute name sets match too).
    #[test]
    fn recorded_cases_match_houdini() {
        let fx = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/sop-orientalongcurve/");
        let cases: serde_json::Value =
            serde_json::from_slice(&std::fs::read(format!("{fx}cases.json")).expect("cases.json")).expect("json");
        let read = |f: &str| hexe_geo::read_geo(&std::fs::read(format!("{fx}{f}")).expect("fixture")).expect("geo");
        let mut input = read("curves_in.geo");
        // hexe_geo::read_geo does not decode Houdini's `unordered` / `i8` group encoding (masks come back all false):
        // restore the scripted membership of `sel` (helix, circle, coincident, knot, single point = prims 0 2 4 6 8)
        for g in input.groups.iter_mut().filter(|g| g.name == "sel") {
            g.mask = (0..12).map(|i| i % 2 == 0 && i <= 8).collect();
        }
        let n = new();
        let mut fails = Vec::new();
        let obj = cases.as_object().expect("object");
        assert!(obj.len() >= 20);
        for (name, parms) in obj {
            let mut p = defaults(&*n);
            let mut vecs: std::collections::BTreeMap<String, [f32; 3]> = Default::default();
            for (k, v) in parms.as_object().expect("parms") {
                for base in ["endupvector", "upvector"] {
                    if let Some(c) = k.strip_prefix(base).filter(|c| c.len() == 1) {
                        // both vector parms default to +Y
                        let e = vecs.entry(base.into()).or_insert([0.0, 1.0, 0.0]);
                        e["xyz".find(c).expect("xyz")] = v.as_f64().expect("f") as f32;
                    }
                }
                if k.starts_with("upvector") && k.len() == 9 || k.starts_with("endupvector") && k.len() == 12 {
                    continue;
                }
                let spec = n.params().iter().find(|s| s.name == *k).unwrap_or_else(|| panic!("parm {k}"));
                let val = match &spec.default {
                    ParamValue::Menu(_) => ParamValue::Menu(v.as_u64().expect("menu") as u32),
                    ParamValue::B(_) => ParamValue::B(v.as_i64().expect("toggle") != 0),
                    ParamValue::S(_) => ParamValue::S(v.as_str().expect("s").into()),
                    ParamValue::F(_) => ParamValue::F(v.as_f64().expect("f") as f32),
                    ParamValue::I(_) => ParamValue::I(v.as_i64().expect("i") as i32),
                    other => panic!("unexpected parm {k}: {other:?}"),
                };
                p = p.with(k, val);
            }
            for (k, v) in vecs {
                p = p.with(&k, ParamValue::V3(v));
            }
            let h = read(&format!("{name}_houdini.geo"));
            let d = cook(&*n, input.clone(), &p).expect("cooks");
            let names = |x: &Detail| {
                let mut v: Vec<String> = x
                    .attrs
                    .iter()
                    .filter(|a| a.class != AttrClass::Detail)
                    .map(|a| format!("{:?}:{}", a.class, a.name))
                    .collect();
                v.sort();
                v
            };
            if names(&d) != names(&h) {
                fails.push(format!("{name}: attrs {:?} vs {:?}", names(&d), names(&h)));
                continue;
            }
            for a in h.attrs.iter().filter(|a| matches!(a.class, AttrClass::Point | AttrClass::Vertex) && a.name != "P")
            {
                let (x, y) = (flat(&a.data), flat(&d.attr(a.class, &a.name).expect("attr").data));
                if x.len() != y.len() {
                    fails.push(format!("{name}: `{}` len {} vs {}", a.name, y.len(), x.len()));
                    continue;
                }
                let bad: Vec<usize> =
                    x.iter().zip(&y).enumerate().filter(|(_, (p, q))| (*p - *q).abs() > 1e-5).map(|(k, _)| k).collect();
                if let Some(&k) = bad.first() {
                    fails.push(format!(
                        "{name}: {:?} `{}`[{k}] hexe {} vs houdini {} ({} bad: {:?})",
                        a.class,
                        a.name,
                        y[k],
                        x[k],
                        bad.len(),
                        &bad[..bad.len().min(12)]
                    ));
                }
            }
        }
        assert!(fails.is_empty(), "{} mismatches:\n{}", fails.len(), fails.join("\n"));
    }

    #[test]
    fn hostile_inputs_never_panic() {
        assert_eq!(run(Detail::default(), &[]).expect("empty").point_count, 0);
        // single point, two coincident points, all-coincident closed curve, NaN / huge coordinates
        for (pts, closed) in [
            (vec![[1.0, 2.0, 3.0]], false),
            (vec![[1.0, 2.0, 3.0]; 2], false),
            (vec![[0.5; 3]; 5], true),
            (vec![[f32::NAN, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]], false),
            (vec![[f32::MAX, 0.0, 0.0], [-f32::MAX, 0.0, 0.0], [0.0, f32::MAX, 0.0]], true),
            (vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0; 3], [1.0, 0.0, 0.0]], false),
        ] {
            for tt in 0..5 {
                let o = run(
                    curve(&pts, closed),
                    &[
                        ("tangenttype", ParamValue::Menu(tt)),
                        ("stretcharoundturns", ParamValue::B(true)),
                        ("extrapolateendtangents", ParamValue::B(true)),
                        ("upvectoratstart", ParamValue::B(false)),
                        ("outputquaternion", ParamValue::B(true)),
                        ("outputtransform4", ParamValue::B(true)),
                        ("applyroll", ParamValue::B(true)),
                        ("incroll", ParamValue::F(1e30)),
                        ("fulltwists", ParamValue::I(i32::MAX)),
                    ],
                )
                .expect("cooks");
                assert_eq!(o.point_count as usize, pts.len());
            }
        }
        // a lone point keeps the identity frame
        let o = run(curve(&[[1.0, 2.0, 3.0]], false), &[]).expect("cooks");
        assert_eq!(v3s(&o, AttrClass::Point, "N"), vec![[0.0, 0.0, 1.0]]);
        assert_eq!(v3s(&o, AttrClass::Point, "up"), vec![[0.0, 1.0, 0.0]]);
        // ... and, like Houdini 20.5 (hython: `-nan(ind)`), a NaN frame under pitch per full curve by edges (0 / 0 edges)
        let o = run(
            curve(&[[1.0, 2.0, 3.0]], false),
            &[
                ("applypitch", ParamValue::B(true)),
                ("pitchper", ParamValue::Menu(3)),
                ("incpitch", ParamValue::F(90.0)),
            ],
        )
        .expect("cooks");
        assert!(v3s(&o, AttrClass::Point, "N")[0].iter().any(|c| c.is_nan()));
        let line = || curve(&[[0.0; 3], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0]], false);
        assert!(matches!(run(line(), &[("scale", ParamValue::F(f32::NAN))]), Err(CookError::BadParam(_))));
        assert!(matches!(
            run(line(), &[("upvector", ParamValue::V3([f32::INFINITY, 0.0, 0.0]))]),
            Err(CookError::BadParam(_))
        ));
        assert!(matches!(run(line(), &[("tangenttype", ParamValue::Menu(5))]), Err(CookError::BadParam(_))));
        assert!(matches!(run(line(), &[("rollper", ParamValue::Menu(99))]), Err(CookError::BadParam(_))));
        assert!(matches!(run(line(), &[("adjustupcurvature", ParamValue::B(true))]), Err(CookError::Unsupported(_))));
        assert!(run(line(), &[("group", ParamValue::S("nope".into()))]).is_err());
        // zero / NaN-free custom up of length 0 falls back to +Y; max stretch 0 never divides into a panic
        assert!(run(line(), &[("upvectortype", ParamValue::Menu(5)), ("upvector", ParamValue::V3([0.0; 3]))]).is_ok());
        assert!(
            run(line(), &[("stretcharoundturns", ParamValue::B(true)), ("maxstretcharoundturns", ParamValue::F(0.0))])
                .is_ok()
        );
        // vertex class: P is refused with a warning, other outputs still written
        let o =
            run(line(), &[("class", ParamValue::Menu(1)), ("outputtranslation", ParamValue::B(true))]).expect("cooks");
        assert!(o.attr(AttrClass::Vertex, "P").is_none() && o.attr(AttrClass::Vertex, "N").is_some());
        // an existing attribute of another shape is refused, never silently reshaped
        let mut l = line();
        l.add_attr(Attribute {
            name: "up".into(),
            class: AttrClass::Point,
            data: AttrData::F32(vec![0.0; 3]),
            type_info: TypeInfo::None,
        });
        assert!(matches!(run(l, &[]), Err(CookError::Unsupported(_))));
        // bad topology is an error, not a panic
        let mut b = line();
        b.prims[0].verts = 0..99;
        assert!(run(b, &[]).is_err());
        let mut b = line();
        b.vertex_point[1] = 77;
        assert!(run(b, &[]).is_err());
    }
}
