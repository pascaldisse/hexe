// three.js viewport: headlight flat-shaded double-sided surface (earcut), edges, round points, sphere/tube wire proxies,
// ground grid with coloured axes, corner axis gizmo, P/N number overlays, draw caps. Houdini camera (camera.ts).
import * as THREE from "three";
import { autoFrame, clip, type Cam, defaultCam, dolly, dragDolly, eye, fit as fitCam, FOV, orbit, pan, wheelDolly } from "./camera";
import type { Cooked } from "./api";
import { describe } from "./pick";
import { CLICK_SLOP, clampSel, combine, inBox, nearest, projectAll } from "./pick";
import { frameScheduler } from "./frame";
import { fmtPos, groundHit } from "./place";
import { cornerColors, cornerGeometry } from "./cd";
import { bounds, fitPositions, capCheck, CAP_LABELS, CAP_LABEL_SCAN, edgeSegments, primCentroids, shapeWire, triangulate, triEstimate, viewState } from "./geom";

/** Viewport tunables (egui `Style` parity). Colours are 0xRRGGBB; light = headlight ambient + diffuse*|N.V|. */
export const STYLE = {
  /** vertical background gradient, top -> bottom (Houdini-style light blue-grey) */
  background: [0x6b8494, 0xb3c3cc] as [number, number], surface: [0.73, 0.76, 0.81] as [number, number, number], ambient: 0.28, diffuse: 0.72,
  point: [1, 0.64, 0.29] as [number, number, number], pointPx: 6, edge: 0x14171a, wire: 0xd6dae0, openEdge: 0xffd24a, shape: 0x7fd4ff,
  /** infinite ground grid: major line every `gridStep`, each cell split `gridSub` x `gridSub`, integer labels on X and Z
   *  (at most `gridLabelMax` each side of the view target); Y axis line `axisYLen` tall */
  gridStep: 1, gridSub: 5, gridMajor: 0x7d8f9c, gridMinor: 0x9aabb6, gridLabel: 0x5e6e7a, gridLabelSize: 0.22, gridLabelMax: 200, axisYLen: 10, axisX: 0xe0453a, axisY: 0x56c456, axisZ: 0x3a7bf0,   gizmoPx: 84,
   /** picked points: colour, size px */
   pick: [1, 0.95, 0.2] as [number, number, number], pickPx: 11,
   /** shelf placement preview, selected-node edge highlight */
   place: 0xff2020, selected: 0xff3030, objSelected: 0xffd400, template: 0x5a6670,
   /** Null indicator (control SOP) colour; template ghost: surface + edge opacity */
   nullControl: 0x00cc00, ghostSurface: 0.22, ghostEdge: 0.35,
   /** edge-group highlight (Group node selected / displayed) */
   edgeGroup: 0xffb000,
 };

// USE_CD (material define): per-corner colour attribute `cd` replaces the flat base colour
const SURFACE_VS = "varying vec3 vP;\n#ifdef USE_CD\nattribute vec3 cd; varying vec3 vC;\n#endif\nvoid main(){\n#ifdef USE_CD\nvC = cd;\n#endif\nvec4 mv = modelViewMatrix * vec4(position, 1.0); vP = mv.xyz; gl_Position = projectionMatrix * mv; }";
// flat per-face normal from screen derivatives (== the Newell normal), headlight: ambient + diffuse * |N.V|
const SURFACE_FS = "varying vec3 vP; uniform vec3 base; uniform float amb; uniform float dif;\n#ifdef USE_CD\nvarying vec3 vC;\n#endif\n" +
  "void main(){ vec3 b = base;\n#ifdef USE_CD\nb = vC;\n#endif\nvec3 n = normalize(cross(dFdx(vP), dFdy(vP))); float d = abs(dot(n, normalize(-vP))); gl_FragColor = vec4(b * (amb + dif * d), 1.0); }";
/** template ghost: the shaded surface at `alpha` */
const GHOST_FS = "varying vec3 vP; uniform vec3 base; uniform float amb; uniform float dif; uniform float alpha;\n" +
  "void main(){ vec3 n = normalize(cross(dFdx(vP), dFdy(vP))); float d = abs(dot(n, normalize(-vP))); gl_FragColor = vec4(base * (amb + dif * d), alpha); }";
const POINT_VS = "uniform float size;\n#ifdef USE_CD\nattribute vec3 cd; varying vec3 vC;\n#endif\nvoid main(){\n#ifdef USE_CD\nvC = cd;\n#endif\nvec4 mv = modelViewMatrix * vec4(position, 1.0); gl_Position = projectionMatrix * mv; gl_Position.z -= 2e-4 * gl_Position.w; gl_PointSize = size; }";
const POINT_FS = "uniform vec3 col;\n#ifdef USE_CD\nvarying vec3 vC;\n#endif\nvoid main(){ if (length(gl_PointCoord - vec2(0.5)) > 0.5) discard; vec3 c = col;\n#ifdef USE_CD\nc = vC;\n#endif\ngl_FragColor = vec4(c, 1.0); }";

const GRID_VS = "varying vec2 vW;\nvoid main(){ vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xz; gl_Position = projectionMatrix * viewMatrix * w; }";
// infinite ground grid: constant ~1 px anti-aliased lines (box-filtered coverage, so a line keeps one brightness along
// its length instead of pulsing between pixel centres), screen density measured per axis (gradient length, not
// fwidth's |dx|+|dy|, which thins diagonals); each line family fades on its own axis once its cells drop below a few
// px, so lines running away from the camera stay while the crowded crossing ones melt. X axis (z = 0) red, Z axis
// (x = 0) blue on top. Colours are raw sRGB.
const GRID_FS = "varying vec2 vW; uniform float step; uniform float sub; uniform vec3 cMinor; uniform vec3 cMajor; uniform vec3 cX; uniform vec3 cZ;\n" +
  "const float HALF_PX = 0.55;\n" +
  "vec2 perPx(vec2 c){ return max(vec2(length(vec2(dFdx(c.x), dFdy(c.x))), length(vec2(dFdx(c.y), dFdy(c.y)))), vec2(1e-6)); }\n" +
  "float lines(float s){ vec2 c = vW / s, d = perPx(c); vec2 px = abs(fract(c - 0.5) - 0.5) / d;\n" +
  "  vec2 l = clamp(HALF_PX + 0.5 - px, 0.0, 1.0) * (1.0 - smoothstep(0.15, 0.4, d)); return max(l.x, l.y); }\n" +
  "float axis(float v){ float d = max(length(vec2(dFdx(v), dFdy(v))), 1e-6); return clamp(HALF_PX + 0.6 - abs(v) / d, 0.0, 1.0); }\n" +
  "void main(){ float mi = lines(step / sub), ma = lines(step), ax = axis(vW.y), az = axis(vW.x);\n" +
  "  vec3 c = cMinor; float a = mi; c = mix(c, cMajor, ma); a = max(a, ma); c = mix(c, cX, ax); a = max(a, ax); c = mix(c, cZ, az); a = max(a, az);\n" +
  "  if (a <= 0.0) discard; gl_FragColor = vec4(c, a); }";

/** Screen-filling vertical gradient as a scene background (part of the render, so `--render` snapshots keep it). */
function gradient(top: number, bottom: number): THREE.Texture | THREE.Color {
  const c = document.createElement("canvas");
  c.width = 2; c.height = 256;
  const g = c.getContext("2d");
  if (!g) return new THREE.Color(top);
  const hex = (x: number) => `#${x.toString(16).padStart(6, "0")}`;
  const lg = g.createLinearGradient(0, 0, 0, 256);
  lg.addColorStop(0, hex(top));
  lg.addColorStop(1, hex(bottom));
  g.fillStyle = lg;
  g.fillRect(0, 0, 2, 256);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

export class Viewport {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  /** ground pass, drawn before `scene`: background gradient, infinite grid, axis labels */
  private ground = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera((FOV * 180) / Math.PI, 1, 0.01, 10000);
  private cam: Cam = defaultCam();
  private content = new THREE.Group();
  private mesh: THREE.Mesh | null = null;
  private open: THREE.LineSegments | null = null;
  private wire: THREE.LineSegments | null = null;
  private points: THREE.Points | null = null;
  private pointMat = new THREE.ShaderMaterial({ vertexShader: POINT_VS, fragmentShader: POINT_FS, uniforms: { size: { value: STYLE.pointPx }, col: { value: new THREE.Vector3(...STYLE.point) } } });
  private gizmo = new THREE.Scene();
  private gizmoCam = new THREE.OrthographicCamera(-1.5, 1.5, 1.5, -1.5, 0.1, 10);
  private labels = document.createElement("div");
  private tag = document.createElement("div");
  private axisTags: HTMLElement[] = [];
  private wireOn = false;
  private pointsOn = true;
  private showP = false;
  private showN = false;
  private shown: Cooked | null = null;
  private primC = new Float32Array(0);
private bb: ReturnType<typeof bounds> = null;
  /** Fit-only bounds (displayed geometry minus prims in groups matching `fitExclude`); `bb` stays the full extent. */
  private fitBb: ReturnType<typeof bounds> = null;
  /** Globs of prim groups Fit ignores (comma/space separated, empty = none): a param, not hardcoded. */
  fitExclude = "*background*";
    private schedule = frameScheduler(() => this.render());
    private pickOn = false;
    private sel: number[] = [];
    private hi: THREE.Points | null = null;
    private hiMat = new THREE.ShaderMaterial({ vertexShader: POINT_VS, fragmentShader: POINT_FS, depthTest: false, uniforms: { size: { value: STYLE.pickPx }, col: { value: new THREE.Vector3(...STYLE.pick) } } });
    private boxEl = document.createElement("div");
  /** true while the mouse is over the viewport (hotkeys are hover scoped) */
  hover = false;
  /** fired when P / N / wire state changes (status label) */
    onState: (s: string) => void = () => {};
    /** fired when the point selection changes: ascending ptnums + a status line (`pt 5  P (x, y, z)`) */
    onPick: (sel: number[], info: string) => void = () => {};
    /** fired when shelf placement starts (`title` = tool name) or ends (`null`) */
    onMode: (title: string | null) => void = () => {};
    private placing: { seg: THREE.LineSegments; label: HTMLElement; done: (p: [number, number, number] | null) => void } | null = null;
    private ray = new THREE.Raycaster();
    private highlightOn = false;
    private highlightMode: "obj" | "sop" | null = null;
    /** last pointer position in viewport px (may be outside the viewport) */
    private lastPointer: { x: number; y: number } | null = null;
    /** the other visible Geometry objects (`setOthers`) */
    private others = new THREE.Group();
    /** a Group node's edge group (`setEdgeHighlight`) */
    private edgeHi = new THREE.Group();
    /** template-flag overlays (grey wireframe) */
    private templ = new THREE.Group();
    /** Null object markers */
    private nullGroup = new THREE.Group();
    /** /obj selectable flag: false = no point picking */
    private pickable = true;
    /** infinite ground grid quad (follows the camera target) + its integer labels, rebuilt when `gridKey` changes */
    private gridPlane!: THREE.Mesh;
    private gridText = new THREE.Group();
    private gridKey = "";
    private labelMats = new Map<string, [THREE.MeshBasicMaterial, number]>();
    private labelQuad = new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2);

  constructor(private host: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    host.appendChild(this.renderer.domElement);
    this.labels.className = "vp-labels";
    this.tag.className = "vp-tag";
        this.boxEl.className = "vp-box";
        host.append(this.labels, this.tag, this.boxEl);
    for (const [t, c] of [["X", STYLE.axisX], ["Y", STYLE.axisY], ["Z", STYLE.axisZ]] as const) {
      const e = document.createElement("span");
      e.className = "vp-axis"; e.textContent = t; e.style.color = `#${c.toString(16).padStart(6, "0")}`;
      this.axisTags.push(e); host.appendChild(e);
    }
    host.addEventListener("mouseenter", () => (this.hover = true));
    host.addEventListener("mouseleave", () => (this.hover = false));
    // ground pass (background + grid + labels) first, then the scene over it with a cleared depth buffer: geometry always
    // covers the grid, even below the floor (Houdini), and the grid never cuts into geometry
    this.ground.background = gradient(STYLE.background[0], STYLE.background[1]);
    this.ground.add(this.grid());
    this.scene.add(this.content, this.others, this.templ, this.nullGroup, this.edgeHi);
    this.gizmo.add(this.axes(1, [STYLE.axisX, STYLE.axisY, STYLE.axisZ]));
    this.bindInput(this.renderer.domElement);
    this.applyCam();
    new ResizeObserver(() => this.resize()).observe(host);
    this.resize();
    this.tag.textContent = this.state();
  }
  /** Infinite XZ ground grid: one camera-following quad, lines drawn in the fragment shader (major per `gridStep`,
   *  `gridSub` x `gridSub` minor cells, X axis red, Z axis blue) that fade out once denser than a few px, so the grid
   *  melts into the horizon instead of ending. Y axis is a plain green line; integer labels follow the view (`updateGrid`). */
  private grid(): THREE.Group {
    const g = new THREE.Group(), rgb = (x: number) => new THREE.Vector3((x >> 16) / 255, ((x >> 8) & 255) / 255, (x & 255) / 255);
    const mat = new THREE.ShaderMaterial({
      vertexShader: GRID_VS, fragmentShader: GRID_FS, transparent: true, depthWrite: false,
      uniforms: { step: { value: STYLE.gridStep }, sub: { value: STYLE.gridSub }, cMinor: { value: rgb(STYLE.gridMinor) }, cMajor: { value: rgb(STYLE.gridMajor) }, cX: { value: rgb(STYLE.axisX) }, cZ: { value: rgb(STYLE.axisZ) } },
    });
    this.gridPlane = new THREE.Mesh(new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2), mat);
    this.gridPlane.frustumCulled = false;
    this.gridPlane.renderOrder = -1;
    const y = new THREE.BufferGeometry();
    y.setAttribute("position", new THREE.Float32BufferAttribute([0, 0, 0, 0, STYLE.axisYLen, 0], 3));
    g.add(this.gridPlane, new THREE.LineSegments(y, new THREE.LineBasicMaterial({ color: STYLE.axisY })), this.gridText);
    return g;
  }
  /** Follow the camera: centre + size the grid quad on the target, and relabel the integers around it along X and Z
   *  (window grows with distance, capped; rebuilt only when the integer window changes). */
  private updateGrid(far: number): void {
    const [tx, , tz] = this.cam.target, st = STYLE.gridStep;
    this.gridPlane.position.set(tx, 0, tz);
    this.gridPlane.scale.set(2 * far, 1, 2 * far);
    const n = Math.min(STYLE.gridLabelMax, Math.max(10, Math.ceil((this.cam.dist * 3) / st)));
    const cx = Math.round(tx / st), cz = Math.round(tz / st), key = `${cx},${cz},${n}`;
    if (key === this.gridKey) return;
    this.gridKey = key;
    this.gridText.clear();
    const s = STYLE.gridLabelSize, d = s * 0.6;
    const put = (t: string, x: number, z: number) => {
      const [m, aspect] = this.labelMat(t), q = new THREE.Mesh(this.labelQuad, m);
      q.scale.set(s * aspect, 1, s); q.position.set(x, 0.001, z);
      this.gridText.add(q);
    };
    for (let k = -n; k <= n; k++) {
      const x = (cx + k) * st, z = (cz + k) * st;
      put(String(x), x, d);
      put(String(z), d, z);
    }
  }
  /** Flat label texture for `t` (read from +Z, like Houdini), cached per string: [material, width/height]. */
  private labelMat(t: string): [THREE.MeshBasicMaterial, number] {
    let m = this.labelMats.get(t);
    if (!m) {
      const c = document.createElement("canvas"), px = 64;
      const ctx = c.getContext("2d")!;
      ctx.font = `${px}px sans-serif`;
      c.width = Math.ceil(ctx.measureText(t).width) + 8; c.height = px + 8;
      ctx.font = `${px}px sans-serif`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillStyle = `#${STYLE.gridLabel.toString(16).padStart(6, "0")}`;
      ctx.fillText(t, c.width / 2, c.height / 2);
      const tex = new THREE.CanvasTexture(c);
      tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 8;
      m = [new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false }), c.width / c.height];
      this.labelMats.set(t, m);
    }
    return m;
  }
  /** Three coloured axis lines of length `len` from the origin (corner gizmo). */
  private axes(len: number, col: [number, number, number]): THREE.LineSegments {
    const p = [0, 0, 0, len, 0, 0, 0, 0, 0, 0, len, 0, 0, 0, 0, 0, 0, len];
    const c = col.flatMap((x) => [(x >> 16) / 255, ((x >> 8) & 255) / 255, (x & 255) / 255]);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(p, 3));
    geo.setAttribute("color", new THREE.Float32BufferAttribute([...c.slice(0, 3), ...c.slice(0, 3), ...c.slice(3, 6), ...c.slice(3, 6), ...c.slice(6, 9), ...c.slice(6, 9)], 3));
    return new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ vertexColors: true }));
  }
  private resize(): void {
    const w = Math.max(1, this.host.clientWidth), h = Math.max(1, this.host.clientHeight);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.draw();
  }
  private draw(): void {
    this.schedule();
  }
    private render(w = this.host.clientWidth, h = this.host.clientHeight): void {
      const r = this.renderer, g = STYLE.gizmoPx;
    r.setViewport(0, 0, w, h);
    r.render(this.ground, this.camera);
    r.autoClear = false;
    r.clearDepth();
    r.render(this.scene, this.camera);
    // corner axis gizmo: same view rotation, own ortho camera, bottom-left
    const e = eye(this.cam), t = this.cam.target, d = new THREE.Vector3(e[0] - t[0], e[1] - t[1], e[2] - t[2]).normalize();
    this.gizmoCam.position.copy(d.multiplyScalar(4));
    this.gizmoCam.up.set(0, 1, 0);
    this.gizmoCam.lookAt(0, 0, 0);
    this.gizmoCam.updateMatrixWorld();
    r.autoClear = false;
    r.clearDepth();
    r.setViewport(8, 8, g, g);
    r.render(this.gizmo, this.gizmoCam);
    r.autoClear = true;
    r.setViewport(0, 0, w, h);
    [new THREE.Vector3(1.25, 0, 0), new THREE.Vector3(0, 1.25, 0), new THREE.Vector3(0, 0, 1.25)].forEach((v, i) => {
      v.project(this.gizmoCam);
      const s = this.axisTags[i].style;
      s.left = `${8 + ((v.x + 1) / 2) * g - 4}px`; s.top = `${h - 8 - ((v.y + 1) / 2) * g - 7}px`;
    });
    this.updateLabels();
  }
    private state(): string { return viewState(this.wireOn, this.showP, this.showN) + (this.pickOn ? " [pick]" : ""); }
  private changed(): void { this.tag.textContent = this.state(); this.onState(this.state()); this.applyVisibility(); this.draw(); }
  private applyVisibility(): void {
    if (this.mesh) this.mesh.visible = !this.wireOn;
    if (this.open) this.open.visible = !this.wireOn;
    if (this.wire) { this.wire.visible = this.wireOn || this.highlightOn; (this.wire.material as THREE.LineBasicMaterial).color.set(this.highlightMode === "obj" ? STYLE.objSelected : this.highlightOn ? STYLE.selected : STYLE.wire); }
    if (this.points) this.points.visible = this.pointsOn;
  }
  /** W: edges only (no shaded surface), like egui. */
  setWireframe(on: boolean): void { this.wireOn = on; this.changed(); }
  get wireframe(): boolean { return this.wireOn; }
  setPoints(on: boolean): void { this.pointsOn = on; this.changed(); }
  setPointSize(px: number): void { this.pointMat.uniforms.size.value = px * this.renderer.getPixelRatio(); this.draw(); }
  setNumbers(kind: "p" | "n", on: boolean): void {
    if (kind === "p") this.showP = on; else this.showN = on;
    if (this.showN && this.shown && !this.primC.length) this.primC = primCentroids(this.shown.prims, this.shown.positions);
    this.changed();
  }
  get numbers(): { p: boolean; n: boolean } { return { p: this.showP, n: this.showN }; }
  /** Point / prim number overlay: first <=500 items inside the frustum (scan <=100k). */
  private updateLabels(): void {
    const c = this.shown, want: [number, number, number, string][] = []; // x, y, 0=point|1=prim, text
    if (c && !c.error && (this.showP || this.showN)) {
      this.camera.updateMatrixWorld();
      const v = new THREE.Vector3(), put = (x: number, y: number, z: number, t: string, kind: number) => {
        if (!Number.isFinite(x)) return;
        v.set(x, y, z).project(this.camera);
        if (v.z < -1 || v.z > 1 || Math.abs(v.x) > 1 || Math.abs(v.y) > 1) return;
        want.push([((v.x + 1) / 2) * this.host.clientWidth, ((1 - v.y) / 2) * this.host.clientHeight, kind, t]);
      };
      if (this.showP) {
        const n = Math.min(c.counts.points, CAP_LABEL_SCAN);
        for (let i = 0, k = 0; i < n && k < CAP_LABELS; i++) { const b = want.length; put(c.positions[i * 3], c.positions[i * 3 + 1], c.positions[i * 3 + 2], `${i}`, 0); k += want.length - b; }
      }
      if (this.showN) {
        const n = this.primC.length / 3;
        for (let i = 0, k = 0; i < n && k < CAP_LABELS; i++) { const b = want.length; put(this.primC[i * 3], this.primC[i * 3 + 1], this.primC[i * 3 + 2], `${i}`, 1); k += want.length - b; }
      }
    }
    // pool: reuse spans
    while (this.labels.childElementCount < want.length) this.labels.appendChild(document.createElement("span"));
    const kids = this.labels.children;
    for (let i = 0; i < kids.length; i++) {
      const s = kids[i] as HTMLElement;
      if (i >= want.length) { s.style.display = "none"; continue; }
      s.style.display = ""; s.className = want[i][2] ? "n" : "p"; s.textContent = want[i][3]; s.style.left = `${want[i][0]}px`; s.style.top = `${want[i][1]}px`;
    }
  }
  private clear(): void {
    for (const o of [...this.content.children]) {
      this.content.remove(o);
      o.traverse((c) => {
        const m = c as THREE.Mesh;
        m.geometry?.dispose();
        if (m.material !== this.pointMat) (Array.isArray(m.material) ? m.material : [m.material]).forEach((x) => x?.dispose());
      });
    }
    this.mesh = this.open = this.wire = this.points = null;
this.bb = this.fitBb = null;
this.primC = new Float32Array(0);
  }
  /** Display identity last auto-framed (null = nothing framed yet). */
  private framedKey: string | null = null;
  /** Next `show` frames again (new scene opened). */
  resetAutoFrame(): void { this.framedKey = null; }
  /** Replace the displayed geometry with a fresh cook. The camera is fitted only when `key` (display identity) changed
   *  (first display / display-node switch), never on a same-key recook (timeline frame, param tweak). Returns draw-cap warnings. */
  show(c: Cooked, key = ""): string[] {
        this.clear();
        const sameGeo = !!this.shown && !this.shown.error && !c.error && this.shown.counts.points === c.counts.points;
        this.shown = c;
    const warnings: string[] = [];
    const n = c.counts.points;
    if (!c.error && n > 0) {
      const pos = new THREE.Float32BufferAttribute(Float32Array.from(c.positions), 3);
      const geo = () => { const g = new THREE.BufferGeometry(); g.setAttribute("position", pos); return g; };
      const segEst = c.prims.reduce((a, p, k) => a + (p.length < 2 ? 0 : p.length - 1 + (c.closed[k] && p.length > 2 ? 1 : 0)), 0);
      const caps = capCheck(triEstimate(c.prims, c.closed), segEst * 2, n);
      warnings.push(...caps.warnings);
      if (caps.mesh) {
        const cd = cornerColors(c);
        // with colours the mesh is built per CORNER (a point can carry a different colour on each face)
        const cg = cd ? cornerGeometry(c) : null;
        const tris = cg ? triangulate(cg.prims, cg.closed, cg.positions) : triangulate(c.prims, c.closed, c.positions);
        if (tris.length) {
          const g = cg ? new THREE.BufferGeometry() : geo();
          if (cg && cd) {
            g.setAttribute("position", new THREE.BufferAttribute(cg.positions, 3));
            g.setAttribute("cd", new THREE.BufferAttribute(cd, 3));
          }
          g.setIndex(new THREE.BufferAttribute(tris, 1));
          const mat = new THREE.ShaderMaterial({
            defines: cd ? { USE_CD: "" } : {},
            vertexShader: SURFACE_VS, fragmentShader: SURFACE_FS, side: THREE.DoubleSide,
            uniforms: { base: { value: new THREE.Vector3(...STYLE.surface) }, amb: { value: STYLE.ambient }, dif: { value: STYLE.diffuse } },
            polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
          });
          this.mesh = new THREE.Mesh(g, mat);
          this.content.add(this.mesh);
        }
      }
      if (caps.edges) {
        const open = edgeSegments(c.prims, c.closed, n, true);
        if (open.length) {
          const g = geo();
          g.setIndex(new THREE.BufferAttribute(open, 1));
          this.open = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: STYLE.openEdge }));
          this.content.add(this.open);
        }
        const all = edgeSegments(c.prims, c.closed, n, false);
        if (all.length) {
          const g = geo();
          g.setIndex(new THREE.BufferAttribute(all, 1));
          this.wire = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: STYLE.edge }));
          this.content.add(this.wire);
        }
      }
      const verts: number[] = [];
      for (const s of c.shapes) verts.push(...shapeWire(s));
      if (verts.length) {
        const w = new THREE.BufferGeometry();
        w.setAttribute("position", new THREE.BufferAttribute(Float32Array.from(verts), 3));
        this.content.add(new THREE.LineSegments(w, new THREE.LineBasicMaterial({ color: STYLE.shape })));
      }
      if (caps.points) {
        const cdp = c.cd_point && c.cd_point.length === n * 3 ? Float32Array.from(c.cd_point) : null;
          const pg = geo();
          if (cdp) pg.setAttribute("cd", new THREE.BufferAttribute(cdp, 3));
          // the coloured variant shares the size/col uniforms of pointMat
          const pm = cdp ? new THREE.ShaderMaterial({ defines: { USE_CD: "" }, vertexShader: POINT_VS, fragmentShader: POINT_FS, uniforms: this.pointMat.uniforms }) : this.pointMat;
          this.points = new THREE.Points(pg, pm);
        this.content.add(this.points);
      }
      if (this.showN) this.primC = primCentroids(c.prims, c.positions);
this.bb = bounds(verts.length ? [...c.positions, ...verts] : c.positions);
      this.computeFitBb();
      this.applyVisibility();
      const r = autoFrame(this.cam, this.framedKey, key, this.fitBb ?? this.bb);
      this.framedKey = r.framed;
      if (r.cam !== this.cam) { this.cam = r.cam; this.applyCam(); }
    }
        // the selection survives a re-cook of the same point count (param tweaks), else it is dropped
        if (this.sel.length) { if (sameGeo) this.setSelection(this.sel, false); else this.setSelection([]); }
        this.draw();
        return warnings;
  }
    /** S: pick mode - LMB drag = box select (shift adds), click = nearest point; off = LMB orbits (a click still picks). */
    setPickMode(on: boolean): void {
      this.pickOn = on;
      this.host.classList.toggle("picking", on);
      this.changed();
    }
    get pickMode(): boolean { return this.pickOn; }
    get selection(): number[] { return [...this.sel]; }
    /** Replace the selection (ids beyond the point count are dropped), rebuild the highlight, notify. */
    setSelection(ids: number[], notify = true): void {
      const n = this.shown?.counts.points ?? 0;
      const next = clampSel(ids, n);
      const same = next.length === this.sel.length && next.every((v, i) => v === this.sel[i]);
      this.sel = next;
      this.buildHighlight();
      this.draw();
      if (notify && !same) this.onPick([...next], describe(next, this.shown?.positions ?? []));
    }
    private buildHighlight(): void {
      if (this.hi) { this.scene.remove(this.hi); this.hi.geometry.dispose(); this.hi = null; }
      const c = this.shown;
      if (!c || !this.sel.length) return;
      const pos = new Float32Array(this.sel.length * 3);
      this.sel.forEach((p, k) => pos.set(c.positions.slice(p * 3, p * 3 + 3), k * 3));
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      this.hi = new THREE.Points(g, this.hiMat);
      this.hi.renderOrder = 10;
      this.scene.add(this.hi);
    }
    /** All displayed points in viewport CSS px (`NaN` = off screen / behind the eye). */
    private screenPoints(): Float32Array {
      const c = this.shown;
      if (!c || c.error) return new Float32Array(0);
      this.camera.updateMatrixWorld();
      this.camera.updateProjectionMatrix();
      const m = new THREE.Matrix4().multiplyMatrices(this.camera.projectionMatrix, this.camera.matrixWorldInverse);
      return projectAll(c.positions, m.elements, this.host.clientWidth, this.host.clientHeight);
    }
    /** Viewport px of point `i` (proof driver + tests), or null when not visible. */
    pointScreen(i: number): { x: number; y: number } | null {
      const s = this.screenPoints();
      return i >= 0 && i * 2 < s.length && Number.isFinite(s[i * 2]) ? { x: s[i * 2], y: s[i * 2 + 1] } : null;
    }
    /** Click pick at viewport px: nearest point within 8 px; nothing there = clear (unless `add`). */
    pickAt(x: number, y: number, add: boolean): void {
      if (!this.pickable) return; // /obj selectable flag off
      const hit = nearest(this.screenPoints(), x, y);
      this.setSelection(combine(this.sel, hit < 0 ? [] : [hit], add));
    }
    /** Box pick between two viewport px corners. */
    pickBox(x0: number, y0: number, x1: number, y1: number, add: boolean): void {
      if (!this.pickable) return;
      this.setSelection(combine(this.sel, inBox(this.screenPoints(), x0, y0, x1, y1), add));
    }
    /** Shelf placement: a red preview (`segs`, xyz pairs around the origin) follows the cursor on the y = 0 plane with a
     *  coordinate readout; LMB calls `done(position)`, `cancelPlace()` calls `done(null)`. One placement at a time. */
    startPlace(title: string, segs: number[], done: (p: [number, number, number] | null) => void): void {
      this.cancelPlace();
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(segs, 3));
      const seg = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: STYLE.place, depthTest: false }));
      seg.renderOrder = 20;
      seg.visible = false;
      this.scene.add(seg);
      const label = document.createElement("div");
      label.className = "vp-place";
      label.hidden = true;
      this.host.append(label);
      this.placing = { seg, label, done };
      this.host.classList.add("placing");
      this.onMode(title);
      // visible at once (Houdini shows the outline before you move): under the pointer if it is over the viewport, else centred
      const w = this.host.clientWidth, h = this.host.clientHeight, m = this.lastPointer;
      const inside = m && m.x >= 0 && m.y >= 0 && m.x <= w && m.y <= h;
      this.placeMove(inside ? m.x : w / 2, inside ? m.y : h / 2);
    }
    get isPlacing(): boolean { return this.placing !== null; }
    cancelPlace(): void { if (this.placing) this.endPlace(null); }
    private endPlace(p: [number, number, number] | null): void {
      const pl = this.placing;
      if (!pl) return;
      this.placing = null;
      this.scene.remove(pl.seg);
      pl.seg.geometry.dispose();
      (pl.seg.material as THREE.Material).dispose();
      pl.label.remove();
      this.host.classList.remove("placing");
      this.onMode(null);
      this.draw();
      pl.done(p);
    }
    private placeHit(x: number, y: number): [number, number, number] | null {
      const w = this.host.clientWidth, h = this.host.clientHeight;
      if (!(w > 0 && h > 0)) return null;
      this.camera.updateMatrixWorld();
      this.ray.setFromCamera(new THREE.Vector2((x / w) * 2 - 1, 1 - (y / h) * 2), this.camera);
      const { origin: o, direction: d } = this.ray.ray;
      return groundHit([o.x, o.y, o.z], [d.x, d.y, d.z]);
    }
    private placeMove(x: number, y: number): void {
      const pl = this.placing;
      if (!pl) return;
      const hit = this.placeHit(x, y);
      pl.seg.visible = hit !== null;
      pl.label.hidden = hit === null;
      if (hit) {
        pl.seg.position.set(...hit);
        pl.label.textContent = fmtPos(hit);
        pl.label.style.left = `${x + 6}px`;
        pl.label.style.top = `${y - 8}px`;
      }
      this.draw();
    }
    /** Template-flagged nodes: their cooks drawn as grey wireframe (no surface, no points) next to the display geometry.
     *  Failed cooks are skipped; each template is capped like the display edges (2M edge verts). */
    /** Edge-group highlight (a Group node making an edge group, selected or displayed): the group's edges of `c` drawn
     *  in the group colour over the surface. `null` / unknown group = cleared. */
    setEdgeHighlight(c: Cooked | null, group: string): void {
      for (const o of [...this.edgeHi.children]) {
        this.edgeHi.remove(o);
        const l = o as THREE.LineSegments;
        l.geometry.dispose();
        (l.material as THREE.Material).dispose();
      }
      const pairs = c && !c.error ? c.edge_groups?.[group] : undefined;
      const n = c ? c.positions.length / 3 : 0;
      if (c && pairs && pairs.length >= 2) {
        const idx = Uint32Array.from(pairs.slice(0, pairs.length - (pairs.length % 2)));
        if (idx.every((p) => p < n)) {
          const g = new THREE.BufferGeometry();
          g.setAttribute("position", new THREE.Float32BufferAttribute(Float32Array.from(c.positions), 3));
          g.setIndex(new THREE.BufferAttribute(idx, 1));
          this.edgeHi.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: STYLE.edgeGroup })));
        }
      }
      this.draw();
    }
    /** The other visible Geometry objects (/obj: every object shows, not just the current one): shaded surface + edges,
     *  per-corner colour when the geometry has Cd. Replaces the previous set; capped like the main mesh. */
    setOthers(cooks: Cooked[]): void {
      for (const o of [...this.others.children]) {
        this.others.remove(o);
        const m = o as THREE.Mesh;
        m.geometry.dispose();
        (m.material as THREE.Material).dispose();
      }
      for (const c of cooks) {
        const n = c.counts.points;
        if (c.error || n <= 0) continue;
        const caps = capCheck(triEstimate(c.prims, c.closed), 0, n);
        const cd = cornerColors(c), cg = cd ? cornerGeometry(c) : null;
        const tris = caps.mesh ? (cg ? triangulate(cg.prims, cg.closed, cg.positions) : triangulate(c.prims, c.closed, c.positions)) : new Uint32Array(0);
        if (tris.length) {
          const g = new THREE.BufferGeometry();
          g.setAttribute("position", new THREE.BufferAttribute(cg ? cg.positions : Float32Array.from(c.positions), 3));
          if (cg && cd) g.setAttribute("cd", new THREE.BufferAttribute(cd, 3));
          g.setIndex(new THREE.BufferAttribute(tris, 1));
          this.others.add(new THREE.Mesh(g, new THREE.ShaderMaterial({
            defines: cd ? { USE_CD: "" } : {}, vertexShader: SURFACE_VS, fragmentShader: SURFACE_FS, side: THREE.DoubleSide,
            uniforms: { base: { value: new THREE.Vector3(...STYLE.surface) }, amb: { value: STYLE.ambient }, dif: { value: STYLE.diffuse } },
            polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
          })));
        }
        const all = edgeSegments(c.prims, c.closed, n, false);
        if (all.length && all.length <= 2_000_000) {
          const g = new THREE.BufferGeometry();
          g.setAttribute("position", new THREE.Float32BufferAttribute(Float32Array.from(c.positions), 3));
          g.setIndex(new THREE.BufferAttribute(all, 1));
          this.others.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: STYLE.edge })));
        }
      }
      this.draw();
    }
    setTemplates(cooks: Cooked[]): void {
      for (const o of [...this.templ.children]) {
        this.templ.remove(o);
        const l = o as THREE.LineSegments;
        l.geometry.dispose();
        (l.material as THREE.Material).dispose();
      }
      // Houdini template flag: the node's geometry drawn like the display geometry but ghosted (see-through surface +
      // faint edges, no depth writes so the display geometry behind it shows)
      for (const c of cooks) {
        const n = c.counts.points;
        if (c.error || n <= 0) continue;
        const caps = capCheck(triEstimate(c.prims, c.closed), 0, n);
        const tris = caps.mesh ? triangulate(c.prims, c.closed, c.positions) : new Uint32Array(0);
        if (tris.length) {
          const g = new THREE.BufferGeometry();
          g.setAttribute("position", new THREE.Float32BufferAttribute(Float32Array.from(c.positions), 3));
          g.setIndex(new THREE.BufferAttribute(tris, 1));
          const m = new THREE.Mesh(g, new THREE.ShaderMaterial({
            vertexShader: SURFACE_VS, fragmentShader: GHOST_FS, side: THREE.DoubleSide, transparent: true, depthWrite: false,
            uniforms: { base: { value: new THREE.Vector3(...STYLE.surface) }, amb: { value: STYLE.ambient }, dif: { value: STYLE.diffuse }, alpha: { value: STYLE.ghostSurface } },
          }));
          m.renderOrder = 5;
          this.templ.add(m);
        }
        // sphere / tube prims: their wire proxies, ghosted
        const sv: number[] = [];
        for (const sh of c.shapes) sv.push(...shapeWire(sh));
        if (sv.length) {
          const w = new THREE.BufferGeometry();
          w.setAttribute("position", new THREE.BufferAttribute(Float32Array.from(sv), 3));
          const l = new THREE.LineSegments(w, new THREE.LineBasicMaterial({ color: STYLE.shape, transparent: true, opacity: STYLE.ghostEdge * 1.6, depthWrite: false }));
          l.renderOrder = 6;
          this.templ.add(l);
        }
        const idx = edgeSegments(c.prims, c.closed, n, false);
        if (!idx.length || idx.length > 2_000_000) continue;
        const g = new THREE.BufferGeometry();
        g.setAttribute("position", new THREE.Float32BufferAttribute(Float32Array.from(c.positions), 3));
        g.setIndex(new THREE.BufferAttribute(idx, 1));
        const l = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: STYLE.edge, transparent: true, opacity: STYLE.ghostEdge, depthWrite: false }));
        l.renderOrder = 6;
        this.templ.add(l);
      }
      this.draw();
    }
    /** Selected-node highlight: the displayed geometry's edges drawn in the selection colour over the shaded surface. */
    /** `obj` = the Geometry object is selected at /obj (yellow), `sop` = the display SOP is selected inside (red). */
    setHighlight(mode: "obj" | "sop" | null): void {
      if (this.highlightMode === mode) return;
      this.highlightMode = mode;
      this.highlightOn = mode !== null;
      this.applyVisibility();
      this.draw();
    }
    /** /obj selectable flag: off = clicks / drag-boxes never pick the object's points */
    setPickable(on: boolean): void {
      this.pickable = on;
      if (!on && this.sel.length) this.setSelection([]);
    }
    /** Null objects: each indicator (`lines` = segment end points of its `control` SOP cook, drawn green; null = the
     *  cook failed: a plain RGB axes cross 0.5 long) at the Null's position; selected ones in the selection yellow. */
    /** last `setNulls` input (F frames the selected ones) */
    private nullState: { at: [number, number, number]; selected: boolean; lines: number[] | null }[] = [];
    /** `m` = the Null object's transform (Houdini row-vector matrix, row-major = three.js `fromArray`); else placed at `at` */
    setNulls(nulls: { at: [number, number, number]; m?: number[] | null; selected: boolean; lines: number[] | null }[]): void {
      this.nullState = nulls;
      for (const o of [...this.nullGroup.children]) {
        this.nullGroup.remove(o);
        const l = o as THREE.LineSegments;
        l.geometry.dispose();
        (l.material as THREE.Material).dispose();
      }
      for (const n of nulls) {
        const pts = n.lines ?? [0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0.5];
        if (pts.length < 6 || pts.length > 600_000) continue;
        const g = new THREE.BufferGeometry();
        g.setAttribute("position", new THREE.Float32BufferAttribute(pts.slice(0, pts.length - (pts.length % 6)), 3));
        const hex = (c: number) => [(c >> 16) / 255, ((c >> 8) & 255) / 255, (c & 255) / 255];
        const nv = Math.floor(pts.length / 6) * 2;
        const cols = n.selected ? Array(nv).fill(hex(STYLE.objSelected)).flat()
          : n.lines ? Array(nv).fill(hex(STYLE.nullControl)).flat()
          : [STYLE.axisX, STYLE.axisX, STYLE.axisY, STYLE.axisY, STYLE.axisZ, STYLE.axisZ].flatMap(hex);
        g.setAttribute("color", new THREE.Float32BufferAttribute(cols, 3));
        const l = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ vertexColors: true, depthTest: false }));
        l.renderOrder = 15;
        if (n.m && n.m.length === 16 && n.m.every(Number.isFinite)) { l.matrixAutoUpdate = false; l.matrix.fromArray(n.m); }
        else l.position.set(...n.at);
        this.nullGroup.add(l);
      }
      this.draw();
    }
    /** /obj display flag: hide / show the whole object (display geometry + templates) */
    setObjectVisible(on: boolean): void {
      this.content.visible = on;
      this.templ.visible = on;
      this.draw();
    }
    /** Draw the scene at exactly w x h px (pixel ratio 1, camera refitted) and return it as a PNG data URL: the headless render. */
    snapshot(w: number, h: number): string {
      const r = this.renderer, ratio = r.getPixelRatio();
      r.setPixelRatio(1);
      r.setSize(w, h, false);
      this.camera.aspect = w / h;
      this.fitCurrent();
      this.camera.updateMatrixWorld();
      this.render(w, h);
      const url = r.domElement.toDataURL("image/png");
      r.setPixelRatio(ratio);
      this.resize();
      return url;
    }
    /** Push the orbit state into the three.js camera. */
  private applyCam(): void {
    const e = eye(this.cam), k = clip(this.cam);
    this.camera.position.set(...e);
    this.camera.up.set(0, 1, 0);
    this.camera.lookAt(...this.cam.target);
    this.camera.near = k.near;
    this.camera.far = k.far;
    this.camera.updateProjectionMatrix();
    this.updateGrid(k.far);
    this.draw();
  }
  private bindInput(el: HTMLElement): void {
    type Drag = { mode: "orbit" | "pan" | "dolly" | "box"; x: number; y: number; x0: number; y0: number; moved: boolean; shift: boolean };
    let drag: Drag | null = null;
    const local = (e: PointerEvent) => { const r = el.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };
    el.addEventListener("contextmenu", (e) => e.preventDefault());
    el.addEventListener("pointerdown", (e) => {
      const lmb = e.button === 0;
      // placement: plain LMB drops the node at the previewed spot (alt+LMB still orbits, MMB / RMB still pan / dolly)
      if (this.placing && lmb && !e.altKey) {
        e.preventDefault();
        const p = local(e), hit = this.placeHit(p.x, p.y);
        if (hit) this.endPlace(hit);
        return;
      }
      // pick mode: LMB drag = box select (shift adds, so shift+LMB no longer pans there; MMB still does)
      const mode = e.button === 2 ? "dolly" : e.button === 1 ? "pan" : lmb && this.pickOn ? "box" : lmb && e.shiftKey ? "pan" : lmb ? "orbit" : null;
      if (!mode) return;
      const p = local(e);
      drag = { mode, x: e.clientX, y: e.clientY, x0: p.x, y0: p.y, moved: false, shift: e.shiftKey };
      try { el.setPointerCapture(e.pointerId); } catch { /* synthetic pointer (proof driver): no capture needed */ }
      e.preventDefault();
    });
    // pointer tracked at window level: placement follows it even across overlays, and starts under it
    window.addEventListener("pointermove", (e) => {
      const r = el.getBoundingClientRect();
      this.lastPointer = { x: e.clientX - r.left, y: e.clientY - r.top };
      const p = this.lastPointer;
      if (this.placing && !drag && p.x >= 0 && p.y >= 0 && p.x <= r.width && p.y <= r.height) this.placeMove(p.x, p.y);
    });
    el.addEventListener("pointermove", (e) => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      const p = local(e);
      if (!drag.moved && Math.hypot(p.x - drag.x0, p.y - drag.y0) <= CLICK_SLOP) return; // still a click
      drag.moved = true;
      if (drag.mode === "box") {
        const b = this.boxEl.style;
        b.display = "block"; b.left = `${Math.min(drag.x0, p.x)}px`; b.top = `${Math.min(drag.y0, p.y)}px`;
        b.width = `${Math.abs(p.x - drag.x0)}px`; b.height = `${Math.abs(p.y - drag.y0)}px`;
        return;
      }
      drag.x = e.clientX; drag.y = e.clientY;
      this.cam = drag.mode === "orbit" ? orbit(this.cam, dx, dy) : drag.mode === "pan" ? pan(this.cam, dx, dy, el.clientHeight) : dolly(this.cam, dragDolly(dx, dy));
      this.applyCam();
    });
    const end = (e: PointerEvent) => {
      const d = drag;
      drag = null;
      this.boxEl.style.display = "none";
      if (el.hasPointerCapture(e.pointerId)) el.releasePointerCapture(e.pointerId);
      if (!d || e.type === "pointercancel") return;
      const p = local(e);
      if (d.mode === "box" && d.moved) this.pickBox(d.x0, d.y0, p.x, p.y, d.shift);
      else if (!d.moved && e.button === 0) this.pickAt(p.x, p.y, d.shift); // a click without travel picks in either mode
    };
    el.addEventListener("pointerup", end);
    el.addEventListener("pointercancel", end);
    el.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.cam = dolly(this.cam, wheelDolly(e.deltaY, e.deltaMode));
      this.applyCam();
    }, { passive: false });
  }
/** Recompute the Fit bounds from the shown cook + `fitExclude`. */
  private computeFitBb(): void {
    const c = this.shown;
    this.fitBb = c && !c.error ? bounds(fitPositions(c.positions, c.prims, c.prim_groups, this.fitExclude)) : null;
  }
  /** Set the Fit exclusion globs and refit. */
  setFitExclude(globs: string): void { this.fitExclude = globs; this.computeFitBb(); }
  private frame(): void {
    const b = this.fitBb ?? this.bb;
    if (!b) return;
    this.cam = fitCam(this.cam, b);
    this.applyCam();
  }
  /** H / toolbar Fit: frame the displayed geometry (default camera when nothing is displayed). */
  fitCurrent(): void {
    if (this.bb) this.frame();
    else { this.cam = defaultCam(); this.applyCam(); }
  }
  /** F (Houdini frame selected): picked points, else selected Null objects, else `node` (the selected SOP's own cook;
   *  "shown" = the displayed geometry), else everything (= H). A tiny selection (one point, one Null) still gets a
   *  readable view: radius at least 5% of the scene (0.25 units when empty). */
  frameSelection(node: Cooked | "shown" | null): void {
    const pts: number[] = [];
    const pos = this.shown && !this.shown.error ? this.shown.positions : [];
    for (const i of this.sel) if (3 * i + 2 < pos.length) pts.push(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
    if (!pts.length) for (const n of this.nullState) if (n.selected) pts.push(...n.at);
    let b = pts.length ? bounds(pts) : null;
    if (!b && node) {
      const c = node === "shown" ? this.shown : node;
      b = c && !c.error ? bounds(fitPositions(c.positions, c.prims, c.prim_groups, this.fitExclude)) : null;
    }
    if (!b) { this.fitCurrent(); return; }
    const full = this.fitBb ?? this.bb;
    this.cam = fitCam(this.cam, { ...b, radius: Math.max(b.radius, full ? full.radius * 0.05 : 0.25, 1e-3) });
    this.applyCam();
  }
}
