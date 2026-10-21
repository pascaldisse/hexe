// hexe UI: viewport (three.js) + param pane + network editor over the Tauri engine commands. State lives in Rust;
// every edit = command -> graph_get -> render, every graph/param change -> cook -> viewport.
import "./style.css";
import "./t4.css";
import { api, msg, type Cooked, type GraphView, type NodeType, type NodeView, type PVal, type RenderReq } from "./api";
import { Network } from "./network";
import { NODE_H, NODE_W } from "./geom";
import { buildShelf } from "./shelf";
import { rulerTicks } from "./ruler";
import { PLACE, previewSegments } from "./place";
import { labelOf } from "./tabcats";
import { renderOpBar, renderParams, setSpareInterfaceHook, showCookError } from "./params";
import { openParmDialog } from "./parmdialog";
import { Spreadsheet } from "./sheet";
import { Viewport } from "./viewport";
import { openFileDialog } from "./filewin";
import { baseName, FILTERS, matchesFilter } from "./filedlg";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { type Nav, navBack, navForward, navStart, navVisit } from "./navhist";
import { frameScheduler } from "./frame";
import { Player } from "./play";
import { objectMatrix, type ObjXform } from "./objxform";
import "./shell";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const status = $("status");
const say = (m: string, err = false, warn = false) => {
  status.textContent = m;
  status.className = err ? "err" : warn ? "warn" : "";
  $("bar").classList.toggle("err", err);
};
/** red border on the node whose cook failed: a style rule keyed by node id survives network re-renders */
const errStyle = document.head.appendChild(document.createElement("style"));
const markError = (id: number | null) => { errStyle.textContent = id == null ? "" : `.node[data-node="${id}"] rect.body { stroke: var(--bad); stroke-width: 3.5; }`; };

let types: NodeType[] = [];
let graph: GraphView = { nodes: [], edges: [], display: null };
let net: Network;
let lastCook: Cooked | null = null;
/** one-shot suffix for the next status line (e.g. `grouped 4 points -> selection`) */
let note = "";
const viewport = new Viewport($("viewport"));
const sheet = new Spreadsheet($("sheet"), (c, o, l) => api.spreadsheet(c, o, l));

/** `display: <name> · pts · prims · verts · cook ms` (+ undrawn / cap warnings) */
function statusLine(c: Cooked, g: GraphView, ms: string, warn: string[]): string {
  const d = g.nodes.find((n) => n.id === g.display);
  if (!d) return "no display node (right-click a node or press D)";
  const verts = c.prims.reduce((a, p) => a + p.length, 0) + c.shapes.length;
  const un = c.counts.unsupported ? ` · ${c.counts.unsupported} undrawn` : "";
    const n = note ? ` · ${note}` : "";
    note = "";
    return `display: ${d.name} · ${c.counts.points} pts · ${c.counts.prims} prims · ${verts} verts · ${ms} ms${un}${warn.length ? ` · WARNING: ${warn.join("; ")}` : ""}${n}`;
}
// cook requests never overlap: while one is in flight only the latest follow-up is kept
let cooking = false, again = false;
async function recook(): Promise<void> {
  if (cooking) { again = true; return; }
  cooking = true;
  try {
    do {
      again = false;
      const t0 = performance.now();
      const c = await api.cook();
      const ms = (performance.now() - t0).toFixed(0);
            lastCook = c;
            const warn = viewport.show(c, String(graph.display));
            await refreshTemplates();
            await refreshOthers();
            syncObjects(); // a Null being edited redraws its indicator
            syncHighlight();
            showCookError(c.error && c.error !== "no display node" ? c.error_node : null, c.error && c.error !== "no display node" ? c.error : null);
      sheet.invalidate(!c.error);
      sheet.setNode(graph.nodes.find((n) => n.id === graph.display)?.name ?? null);
      markError(c.error ? c.error_node : null);
      net.setCookError(c.error && c.error !== "no display node" ? c.error_node : null, c.error ?? "");
      if (c.error === "no display node") say("no display node (right-click a node or press D)");
      else if (c.error) say(c.error, true);
      else say(statusLine(c, graph, ms, warn), false, warn.length > 0);
    } while (again);
  } catch (e) {
    say(msg(e), true);
  } finally {
    cooking = false;
  }
}

setSpareInterfaceHook(async (id, parms) => { await api.nodeSpareSet(id, parms); await refresh(false); });
function showParams(): void {
  const id = net.selectedNode;
  const node = graph.nodes.find((n) => n.id === id) ?? null;
  const type = types.find((t) => t.type === node?.type);
  // pane + viewport operator bar edit the same node: each repaints the other after a commit (never itself: a scrub survives)
  // spare parameters (Edit Parameter Interface): values + the dialog, SOP nodes only
  const opts = node && type ? {
    spareSet: async (n: string, v: PVal | number[]) => { await api.nodeSpareValue(node.id, n, v); await recook(); },
    setChannel: async (c: string, e: string | null) => { await api.nodeSetChannel(node.id, c, e); },
    setLock: async (p: string, on: boolean) => { await api.nodeSetLock(node.id, p, on); },
    net: `/obj/${net.objPath.join("/")}`,
    changed: () => void refresh().then(showParams),
    editInterface: () => openParmDialog(`/obj/${[...net.objPath, node.name].join("/")}`, type.params, node.spare ?? [], async (parms) => {
      await api.nodeSpareSet(node.id, parms);
      await refresh(false);
      showParams();
    }),
  } : {};
  const pane = () => renderParams($("params"), node, type, (i, n, v) => setParam(i, n, v).then(bar), opts);
  const bar = () => renderOpBar($("opbar"), node, type, setParam, pane);
  pane();
  bar();
  if (net.level === "obj") showObjectParams();
  paintCrumbs();
  syncHighlight();
}

/** /obj objects -> viewport: Geometry display flag (show / hide), selectable flag (point picking), Null indicators.
 *  A Null's network (its `control` SOP) is drawn as its indicator at its position, so the current object's own geometry
 *  is hidden while it is a Null. */
function syncObjects(): void {
  const geo = net.geo;
  viewport.setObjectVisible(geo.visible && geo.kind !== "null");
  viewport.setPickable(geo.selectable);
  void refreshNulls();
}

/** each drawn Null's indicator: its display cook (the current object's = the last cook) as line pairs */
let nullSeq = 0;
/** latest object-param fetch for the pane (a slower earlier one must not paint over it) */
let objParamSeq = 0;
async function refreshNulls(): Promise<void> {
  const seq = ++nullSeq;
  const out = [];
  for (const n of net.nulls) {
    if (!n.drawn) continue;
    const c = n.name === net.current ? lastCook : await api.cookObject(n.name).catch(() => null);
    // its Transform tab (Rust-stored object params) places the indicator: Houdini's object matrix
    const v = await api.objectParams(n.name).then(([, vals]) => vals, () => null);
    const m = v ? objectMatrix(v as ObjXform) : null;
    out.push({ at: m ? [m[12], m[13], m[14]] as [number, number, number] : n.at, m, selected: n.selected, lines: c && !c.error ? linePairs(c) : null });
  }
  if (seq === nullSeq) viewport.setNulls(out);
}
/** a cook's prims as flat segment end points `[x0,y0,z0, x1,y1,z1, ...]` (open prims; closed ones wrap) */
function linePairs(c: Cooked): number[] {
  const p = c.positions, out: number[] = [];
  const at = (i: number) => out.push(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]);
  c.prims.forEach((prim, k) => {
    const m = prim.length, wrap = c.closed[k] && m > 2;
    for (let i = 0; i + 1 < m + (wrap ? 1 : 0); i++) { at(prim[i]); at(prim[(i + 1) % m]); }
  });
  return out.every(Number.isFinite) ? out : [];
}

/** viewport F: the selection - picked points / selected Null objects (viewport side), else the selected SOP (its own cook
 *  when it is not the display node), else the Geometry object selected at /obj (= the displayed geometry), else all */
async function frameSelected(): Promise<void> {
  const id = net.level === "sop" ? net.selectedNode : null;
  const node = id !== null && id !== graph.display ? await api.cookNode(id).catch(() => null) : id !== null || net.objectSelected ? "shown" as const : null;
  viewport.frameSelection(node);
}

/** viewport highlight: yellow = the Geometry object selected at /obj, red = the display SOP selected inside */
function syncHighlight(): void {
  viewport.setHighlight(net.objectSelected ? "obj" : net.level === "sop" && net.selectedNode !== null && net.selectedNode === graph.display ? "sop" : null);
  void syncGroupHighlight();
}

/** Houdini: a Group node making an EDGE group shows its edges in the viewport while it is selected (first) or carries the
 *  display flag. The selected node is cooked on its own when it is not the display node. */
let groupHiSeq = 0;
async function syncGroupHighlight(): Promise<void> {
  const seq = ++groupHiSeq;
  const isEdgeGroup = (n: NodeView | undefined) => !!n && n.type === "group" && n.params.grouptype === 3;
  const sel = net.level === "sop" ? graph.nodes.find((n) => n.id === net.selectedNode) : undefined;
  const disp = graph.nodes.find((n) => n.id === graph.display);
  const node = isEdgeGroup(sel) ? sel : isEdgeGroup(disp) ? disp : undefined;
  if (!node || net.level !== "sop") { viewport.setEdgeHighlight(null, ""); return; }
  const c = node.id === graph.display ? lastCook : await api.cookNode(node.id).catch(() => null);
  if (seq !== groupHiSeq) return; // a newer selection won
  viewport.setEdgeHighlight(c, String(node.params.groupname ?? ""));
}

/** /obj selection in the pane: the object header; its transform / render params are not in hexe yet (said, not faked) */
function showObjectParams(): void {
  const host = $("params");
  // Houdini keeps the generator's operator bar up after placing (Box  Primitive Type  Size ...): the display SOP's params
  const disp = graph.nodes.find((n) => n.id === graph.display) ?? null;
  if (net.objectSelected && disp) renderOpBar($("opbar"), disp, types.find((t) => t.type === disp.type), setParam, () => {});
  else $("opbar").replaceChildren();
  const nul = net.nulls.find((o) => o.selected);
  if (!net.objectSelected && nul) {
    // the Null object's own parameters (Houdini Transform / Render / Misc tabs), stored in Rust with the object
    const seq = ++objParamSeq;
    void api.objectParams(nul.name).then(([specs, values]) => {
      if (seq !== objParamSeq) return;
      const view: NodeView = { id: -1, type: "null", name: nul.name, x: 0, y: 0, inputs: [], required: 0, bypass: false, params: values };
      renderParams(host, view, { type: "null", label: "Null", inputs: [], required: 0, params: specs }, async (_id, parm, value) => {
        await api.objectSetParam(nul.name, parm, value);
        void refreshNulls();
      });
    }, (e) => say(msg(e), true));
    return;
  }
  if (!net.objectSelected) {
    host.replaceChildren(Object.assign(document.createElement("div"), { className: "hint center", textContent: "No Operator Selected" }));
    return;
  }
  const head = Object.assign(document.createElement("div"), { className: "phead" });
  head.append(Object.assign(document.createElement("span"), { className: "phead-ty", textContent: "Geometry" }), Object.assign(document.createElement("b"), { textContent: net.objName }));
  const note = Object.assign(document.createElement("div"), { className: "hint", textContent: `Object-level parameters (Transform, Render, Misc) are not in hexe yet. Double-click ${net.objName} in the network to edit its geometry (${graph.nodes.length} SOP node${graph.nodes.length === 1 ? "" : "s"}).` });
  host.replaceChildren(head, note);
}

/** network pane back / forward (Houdini's path arrows): every level change is a visit, the arrows walk them */
let nav = navStart<"obj" | "sop">("obj");
let navigating = false;
function navGo(to: Nav<"obj" | "sop">): void {
  if (to === nav) return;
  nav = to;
  navigating = true;
  net.setLevel(nav.cur, nav.cur === "obj");
  navigating = false;
}
const ARROW = (d: string) => `<svg viewBox="0 0 24 18" width="24" height="18"><path d="${d}"/></svg>`;
function navArrows(): HTMLElement {
  const box = Object.assign(document.createElement("span"), { className: "pb-nav" });
  const mk = (cls: string, tip: string, path: string, on: boolean, go: () => void) => {
    const b = Object.assign(document.createElement("button"), { className: `pb-arrow ${cls}`, title: tip, disabled: !on });
    b.innerHTML = ARROW(path);
    b.addEventListener("click", go);
    return b;
  };
  box.append(
    mk("back", "Back (previous network)", "M2 9 L10 2 V6 H21 V12 H10 V16 Z", nav.back.length > 0, () => navGo(navBack(nav))),
    mk("fwd", "Forward", "M22 9 L14 2 V6 H3 V12 H14 V16 Z", nav.fwd.length > 0, () => navGo(navForward(nav))),
  );
  return box;
}

/** path bars + network tab: `obj`, then the containers dived into, then the object entered (SOP level); the params bar
 *  adds the selected SOP. Every network part but the last is a link up to that level. */
function paintCrumbs(): void {
  const inside = net.level === "sop";
  const sel = inside ? graph.nodes.find((n) => n.id === net.selectedNode)?.name : undefined;
  const path = net.objPath;
  for (const bar of document.querySelectorAll<HTMLElement>("[data-crumbs]")) {
    const parts = ["obj", ...path, ...(bar.dataset.crumbs === "param" && sel ? [sel] : [])];
    bar.replaceChildren(Object.assign(document.createElement("span"), { className: "pb-ico" }));
    if (bar.dataset.crumbs === "net") bar.prepend(navArrows());
    parts.forEach((p, i) => {
      if (i) bar.append(Object.assign(document.createElement("span"), { className: "pb-sep", textContent: "›" }));
      const link = i < path.length;
      const c = Object.assign(document.createElement("span"), { className: link ? "pb-txt link" : "pb-txt", textContent: p });
      if (link) c.addEventListener("click", () => net.goToPath(i - 1));
      bar.append(c);
    });
  }
  $("nettab").textContent = `/obj${path.map((p) => `/${p}`).join("")}`;
}

/** the template-flagged node ghosted in the viewport (one per network; on the display node it just looks displayed) */
async function refreshTemplates(): Promise<void> {
  const ids = net.templated.filter((i) => i !== graph.display);
  const cooks = [];
  for (const i of ids) cooks.push(await api.cookNode(i).catch(() => null));
  viewport.setTemplates(cooks.filter((c): c is Cooked => c !== null));
}

/** Shelf tool: a generator lands right of the network; a filter lands under the selected node and takes it as input 0.
 *  Either way the new node becomes the display + selected node (Houdini shelf behaviour, SOP level). */
async function shelfAdd(type: string): Promise<void> {
  // at /obj the Null tool makes a Null OBJECT (a transform marker placed in the viewport), not a null SOP
  if (type === "null" && net.level === "obj") {
    const at: [number, number, number] | null = await new Promise((done) => viewport.startPlace("Null", previewSegments("nullobj", {}), done));
    if (!at) { say("Null: placement cancelled"); return; }
    const o = await net.addNull(at);
    if (!o) return;
    // the click position is its Translate (Houdini places the Null object there)
    if (at.some((c) => c !== 0)) await api.objectSetParam(o.name, "t", at).catch((e) => say(msg(e), true));
    syncObjects();
    showParams();
    say(`placed ${o.name} at (${at.map((v) => +v.toFixed(3)).join(", ")})`);
    return;
  }
  const ty = types.find((t) => t.type === type);
  // inside the object a filter tool wires under the selected SOP; at /obj tools never wire (they make the object)
  const src = net.level === "sop" ? graph.nodes.find((n) => n.id === net.selectedNode) : undefined;
  const wire = !!src && !!ty && ty.inputs.length > 0;
  const x = wire ? src.x : graph.nodes.length ? Math.max(...graph.nodes.map((n) => n.x)) + NODE_W + 80 : 0;
  const y = wire ? src.y + NODE_H + 40 : graph.nodes.length ? Math.min(...graph.nodes.map((n) => n.y)) : 0;
  // generators place interactively (Houdini: "Select position for Box"): red preview on the ground under the cursor
  let at: [number, number, number] | null = null;
  const posParam = PLACE[type];
  if (posParam && ty && !wire) {
    const defaults = Object.fromEntries(ty.params.map((p) => [p.name, p.default]));
    at = await new Promise((done) => viewport.startPlace(labelOf(type), previewSegments(type, defaults), done));
    if (!at) { say(`${labelOf(type)}: placement cancelled`); return; }
  }
  // the first placed generator names the object (Houdini: box_object1)
  if (at && !graph.nodes.length) {
    const old = net.objName;
    await api.objectRename(old, `${type}_object1`).then((n) => net.renameGeo(old, n)).catch((e) => say(msg(e), true));
  }
  await structural(async () => {
    const id = await api.nodeAdd(type, x, y);
    if (at && posParam) await api.nodeSetParam(id, posParam, at);
    if (wire) await api.edgeConnect(src.id, id, 0);
    await api.setDisplay(id);
    await refresh(false);
    net.selectNode(id);
    // a placed generator leaves you at /obj with the Geometry object selected (double-click it to see the SOP)
    if (at) net.setLevel("obj", true);
  })();
}

// param edits: serialized + coalesced per param (a slider drag fires many input events)
const pendingParams = new Map<string, [number, string, PVal]>();
let flushing = false;
async function setParam(id: number, name: string, value: PVal): Promise<void> {
  pendingParams.set(`${id}:${name}`, [id, name, value]);
  if (flushing) return;
  flushing = true;
  try {
    while (pendingParams.size) {
      const [k, [i, n, v]] = pendingParams.entries().next().value!;
      pendingParams.delete(k);
      await api.nodeSetParam(i, n, v);
      await recook();
    }
  } catch (e) {
    say(msg(e), true);
    throw e;
  } finally {
    flushing = false;
  }
}

/** cooks of the other visible Geometry objects (viewport `setOthers`). An object's network only changes while it is
 *  current, so a cached cook stays valid until it stops being current, the frame changes, or a scene loads */
const othersCache = new Map<string, Cooked>();
let cacheCurrent = "";
async function refreshOthers(): Promise<void> {
  if (net.current !== cacheCurrent) { othersCache.delete(cacheCurrent); othersCache.delete(net.current); cacheCurrent = net.current; }
  const names = net.otherVisibleGeos;
  for (const n of names) if (!othersCache.has(n)) { const c = await api.cookObject(n).catch(() => null); if (c) othersCache.set(n, c); }
  for (const k of [...othersCache.keys()]) if (!names.includes(k) && k !== cacheCurrent) othersCache.delete(k);
  viewport.setOthers(names.flatMap((n) => othersCache.get(n) ?? []));
}

async function refresh(cook = true): Promise<void> {
  const [infos, cur] = await api.objectsInfo();
  net.syncObjects(infos, cur);
  graph = await api.graphGet();
  net.setGraph(graph);
  if (cook) await recook();
}

/** run a structural command, then re-sync the graph + viewport; failures land in the status bar */
const structural = (f: () => Promise<unknown>) => async () => {
  try { await f(); } catch (e) { say(msg(e), true); }
  await refresh();
  showParams();
};

async function main(): Promise<void> {
  types = await api.listNodeTypes();
  net = new Network($("network"), types, {
    select: () => { showParams(); syncHighlight(); },
    add: async (t, x, y, into) => { await structural(async () => { const id = into ? await api.nodeAddWired(t, x, y, into.from, into.to, into.input) : await api.nodeAdd(t, x, y); await refresh(false); net.selectNode(id); })(); },
    remove: (id) => structural(() => api.nodeRemove(id))(),
    move: async (id, x, y) => { await api.nodeMove(id, x, y); const n = graph.nodes.find((q) => q.id === id); if (n) { n.x = x; n.y = y; } },
    connect: (f, t, i) => structural(() => api.edgeConnect(f, t, i))(),
    disconnect: (t, i) => structural(() => api.edgeDisconnect(t, i))(),
    display: (id) => structural(() => api.setDisplay(id))(),
    refresh: () => structural(async () => {})(),
    error: (m) => say(m, true),
    templates: () => void refreshTemplates(),
    level: (l) => { if (!navigating) nav = navVisit(nav, l); showParams(); },
    objects: () => { syncObjects(); showParams(); void refreshOthers(); },
  });

  // Houdini tooltips: any element with `data-tip` (multi-line text) shows a black box near the pointer after a short hover
  {
    const tip = Object.assign(document.createElement("div"), { id: "tip" });
    document.body.append(tip);
    let timer = 0, over: Element | null = null;
    const hide = () => { clearTimeout(timer); tip.style.display = "none"; over = null; };
    window.addEventListener("mouseover", (e) => {
      const el = (e.target as Element).closest?.("[data-tip]") ?? null;
      if (el === over) return;
      hide();
      if (!el) return;
      over = el;
      const x = e.clientX, y = e.clientY;
      timer = window.setTimeout(() => {
        tip.textContent = el.getAttribute("data-tip") ?? "";
        tip.style.display = "block";
        const w = tip.offsetWidth, h = tip.offsetHeight;
        tip.style.left = `${Math.max(0, Math.min(x + 12, window.innerWidth - w - 4))}px`;
        tip.style.top = `${Math.max(0, Math.min(y + 16, window.innerHeight - h - 4))}px`;
      }, 450);
    });
    window.addEventListener("mousedown", hide, true);
    window.addEventListener("wheel", hide, { capture: true, passive: true });
  }
  // desktop app: no browser context menu (Back / Refresh / Print) anywhere except in text fields
  window.addEventListener("contextmenu", (e) => { if (!(e.target as Element).closest("textarea, input:not([type=checkbox]):not([type=range])")) e.preventDefault(); });
  buildShelf($("shelfL"), $("shelfR"), types.map((t) => t.type), (t) => void shelfAdd(t), (m) => say(m, false, true));
  // desktop / radial menu selectors: `data-list` = groups split by `|`; group 1 = radio rows (`*` current, `#` section head),
  // later groups = commands. Only the current entry exists in hexe: everything else is listed disabled (honest, not a no-op).
  for (const drop of document.querySelectorAll<HTMLElement>(".mdrop.radio")) {
    (drop.dataset.list ?? "").split("|").forEach((group, gi) => {
      if (gi) drop.append(Object.assign(document.createElement("div"), { className: "msep" }));
      for (const raw of group.split(",")) {
        if (raw.startsWith("#")) { drop.append(Object.assign(document.createElement("div"), { className: "mhead", textContent: raw.slice(1) })); continue; }
        const cur = raw.startsWith("*"), b = document.createElement("button");
        b.className = gi ? "mi" : `mi rad${cur ? " on" : ""}`;
        b.textContent = cur ? raw.slice(1) : raw;
        b.disabled = !cur;
        if (!cur) b.title = "not in hexe yet";
        drop.append(b);
      }
    });
  }
  // menu bar: click opens a menu, hovering another top item while one is open switches to it, any item / outside click closes
  const menus = [...document.querySelectorAll<HTMLElement>("#menubar .menu")];
  const closeMenus = () => menus.forEach((m) => m.classList.remove("open"));
  for (const m of menus) {
    m.querySelector(".mtop")?.addEventListener("click", () => { const was = m.classList.contains("open"); closeMenus(); m.classList.toggle("open", !was); });
    m.addEventListener("mouseenter", () => { if (menus.some((q) => q.classList.contains("open"))) { closeMenus(); m.classList.add("open"); } });
    m.querySelector(".mdrop")?.addEventListener("click", (e) => { if ((e.target as Element).closest("button.mi")) closeMenus(); });
  }
  window.addEventListener("mousedown", (e) => { if (!(e.target as Element).closest("#menubar .menu")) closeMenus(); });
  for (const b of document.querySelectorAll<HTMLElement>("[data-cmd]")) b.addEventListener("click", () => void net.undo(b.dataset.cmd === "redo"));
  // left pane tabs: Scene View / Geometry Spreadsheet (pages stay laid out; the hidden one is only invisible)
  const showTab = (id: string) => {
    for (const p of document.querySelectorAll<HTMLElement>("#leftpane .tabpage")) p.classList.toggle("on", p.id === id);
    for (const t of document.querySelectorAll<HTMLElement>("#leftpane .ptab")) t.classList.toggle("on", t.dataset.tab === id);
  };
  for (const b of document.querySelectorAll<HTMLElement>("[data-tab]")) b.addEventListener("click", () => showTab(b.dataset.tab ?? "sceneview"));
  $("netadd").addEventListener("click", () => net.openMenu());

  const wire = $<HTMLInputElement>("wire");
  wire.addEventListener("change", () => viewport.setWireframe(wire.checked));
  $("ptsize").addEventListener("input", (e) => viewport.setPointSize(Number((e.target as HTMLInputElement).value)));
  $("pts").addEventListener("change", (e) => viewport.setPoints((e.target as HTMLInputElement).checked));
$("fit").addEventListener("click", () => viewport.fitCurrent());
$<HTMLInputElement>("fitx").addEventListener("change", (e) => { viewport.setFitExclude((e.target as HTMLInputElement).value); viewport.fitCurrent(); });
    // viewport picking: selection lives here + in Rust (`selection_get`); the group button writes it into a new group SOP
    const grp = $<HTMLButtonElement>("grpsel"), pickBtn = $("pick");
    // viewport tool columns mirror the viewport state (hotkeys change it too)
    const syncPick = () => {
      pickBtn.classList.toggle("on", viewport.pickMode);
      $("selmode").classList.toggle("on", !viewport.pickMode);
      $("ptnums").classList.toggle("on", viewport.numbers.p);
      $("primnums").classList.toggle("on", viewport.numbers.n);
      wire.checked = viewport.wireframe;
    };
    viewport.onState = syncPick;
    // shelf placement: the viewport title becomes the tool, the hint line tells what a click does
    const hint = $("vphint"), idleHint = hint.textContent ?? "";
    viewport.onMode = (t) => {
      $("vptitle").textContent = t ?? "View";
      hint.textContent = t ? `Select position for ${t}. Middle / right mouse still pan / dolly. Esc cancels.` : idleHint;
      hint.classList.toggle("placing", !!t);
    };
    syncPick();
    $("selmode").addEventListener("click", () => viewport.setPickMode(false));
    $("ptnums").addEventListener("click", () => viewport.setNumbers("p", !viewport.numbers.p));
    $("primnums").addEventListener("click", () => viewport.setNumbers("n", !viewport.numbers.n));
    viewport.onPick = (sel, info) => {
      grp.disabled = sel.length === 0;
      api.selectionSet(sel).catch((e) => say(msg(e), true));
      say(info);
    };
    pickBtn.addEventListener("click", () => viewport.setPickMode(!viewport.pickMode));
    grp.addEventListener("click", structural(async () => {
      const n = viewport.selection.length;
      const id = await api.groupFromSelection();
      await refresh(false);
      net.selectNode(id);
      note = `grouped ${n} point${n === 1 ? "" : "s"} -> group node #${id}`;
    }));
  const path = $<HTMLInputElement>("path");
  const open = async () => {
    const p = path.value.trim();
    // no path typed: Open would hand "" to the OS ("The system cannot find the path specified"); ask for one instead
    if (!p) { path.focus(); say("type a scene path (.json / .ens / .nmc), then Enter"); return; }
    try {
      viewport.resetAutoFrame(); // a newly opened scene frames on its first cook
      if (/\.(ens|nmc|nmb|nma)$/i.test(p)) {
        // Endorphin file: one `file` SOP over it (hexe-geo read_any opens .ens/.nmc as rest-pose geometry).
        await api.graphNew();
        const id = await api.nodeAdd("file", 0, 0);
        await api.nodeSetParam(id, "file", p);
        await api.setDisplay(id);
        graph = await api.graphGet();
      } else {
        graph = await api.graphLoad(p);
      }
      net.level = "sop"; // an opened scene shows its SOP network (the object is the whole scene)
      nav = navVisit(nav, "sop");
      // the scene's Geometry objects (a multi-object file shows them all, empty ones too)
      net.resetObjects();
      othersCache.clear();
      const [infos, cur] = await api.objectsInfo();
      net.syncObjects(infos, cur);
      if (infos.length > 1) net.placeAllGeos();
      net.selectNode(null);
      net.setGraph(graph);
      await recook();
      showParams();
    } catch (e) { say(`open ${p}: ${msg(e)}`, true); }
  };
  // File > Open... / Ctrl+O: Houdini-style browser at $HIP (the open file's folder; home when untitled); the path box + Enter opens directly
  const browse = async () => { const p = await openFileDialog(path.value.trim()); if (p) { path.value = p; await open(); } };
  $("open").addEventListener("click", () => void browse());
  // drag a scene file (.hip / .json / .ens ...) from Explorer onto the window = open it (Tauri hands over the real path)
  const dropVeil = Object.assign(document.createElement("div"), { className: "drop-veil", textContent: "Drop to open" });
  document.body.append(dropVeil);
  const sceneOf = (paths: string[]) => paths.find((p) => matchesFilter(baseName(p), FILTERS[0]));
  void getCurrentWebview().onDragDropEvent(({ payload: d }) => {
    if (d.type === "enter") { const p = sceneOf(d.paths); dropVeil.textContent = p ? `Drop to open ${baseName(p)}` : "Not a scene file (.hip, .json, .ens, .nmc)"; dropVeil.classList.toggle("bad", !p); dropVeil.classList.add("on"); }
    else if (d.type === "leave") dropVeil.classList.remove("on");
    else if (d.type === "drop") {
      dropVeil.classList.remove("on");
      const p = sceneOf(d.paths);
      if (!p) { say(`not a scene file: ${d.paths.map(baseName).join(", ")}`, true); return; }
      path.value = p;
      void open();
    }
  }).catch((e) => say(`drag and drop unavailable: ${msg(e)}`, true));
  path.addEventListener("keydown", (e) => { if (e.key === "Enter") open(); });
  $("save").addEventListener("click", async () => {
    try { await api.graphSave(path.value.trim()); say(`saved ${path.value.trim()}`); } catch (e) { say(msg(e), true); }
  });
  $("new").addEventListener("click", structural(async () => { await api.graphNew(); net.resetObjects(); othersCache.clear(); net.level = "obj"; nav = navStart("obj"); net.selectNode(null); }));

  $("save").title = "Cmd+S"; $("open").title = "Cmd+O";
  // hotkeys: viewport-hover scoped H/F W P N; network-hover scoped D; Cmd+S / Cmd+O / R global; never while typing
  let mouse = { x: 0, y: 0 };
  window.addEventListener("mousemove", (e) => (mouse = { x: e.clientX, y: e.clientY }));
  const typing = (t: EventTarget | null) => {
    const e = t as HTMLInputElement | null;
    return !!e && (e.tagName === "TEXTAREA" || e.tagName === "SELECT" || (e.tagName === "INPUT" && e.type !== "checkbox" && e.type !== "range"));
  };
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && viewport.isPlacing) { e.preventDefault(); viewport.cancelPlace(); return; }
    const k = e.key.toLowerCase();
    if ((e.metaKey || e.ctrlKey) && !e.altKey && (k === "s" || k === "o")) {
      e.preventDefault();
      if (k === "s") $("save").click();
      else void browse();
      return;
    }
    if (e.metaKey || e.ctrlKey || e.altKey || typing(e.target)) return;
    if (k === "r") { e.preventDefault(); void recook(); return; }
    if (viewport.hover) {
      if (k === "h") viewport.fitCurrent();
      else if (k === "f") void frameSelected();
      else if (k === "w") { wire.checked = !viewport.wireframe; viewport.setWireframe(wire.checked); }
      else if (k === "p") viewport.setNumbers("p", !viewport.numbers.p);
            else if (k === "n") viewport.setNumbers("n", !viewport.numbers.n);
            else if (k === "s") viewport.setPickMode(!viewport.pickMode);
      else return;
      e.preventDefault();
      return;
    }
    if (k === "d") {
      const at = document.elementFromPoint(mouse.x, mouse.y);
      const id = at && $("network").contains(at) ? at.closest<SVGElement>("[data-node]")?.dataset.node : undefined;
      if (id !== undefined) { e.preventDefault(); void structural(() => api.setDisplay(Number(id)))(); }
    }
  });
  // timeline: `$F` lives in Rust (`--frame`/`--fps` seed it); number box + slider scrub it, every change re-cooks at ($F - 1) / fps
  const frameBox = $<HTMLInputElement>("frame"), frameSlider = $<HTMLInputElement>("frameslider");
  const fStart = $<HTMLInputElement>("fstart"), fEnd = $<HTMLInputElement>("fend");
  let rate = 30;
  const range = (): [number, number] => {
    const a = Math.max(1, Math.round(Number(fStart.value)) || 1);
    return [a, Math.max(a + 1, Math.round(Number(fEnd.value)) || a + 1)];
  };
  // ruler: labelled major ticks + minor ticks over the playback range, positioned like the slider thumb travel
  const drawTicks = () => {
    const [a, b] = range();
    const host = $("ticks");
    host.replaceChildren(...rulerTicks(a, b, host.clientWidth).map(([f, major]) => {
      const t = document.createElement("span");
      t.className = major ? "tk major" : "tk";
      t.style.left = `calc(7px + (100% - 14px) * ${(f - a) / (b - a)})`;
      if (major) t.dataset.f = String(f);
      return t;
    }));
  };
  new ResizeObserver(drawTicks).observe($("ticks"));
  const showFrame = (f: number, fps: number) => {
    const [a, b] = range();
    rate = fps;
    frameBox.value = String(f);
    frameSlider.min = String(a);
    frameSlider.max = String(Math.max(b, f));
    frameSlider.value = String(f);
    $("fpslab").textContent = `${fps} fps`;
  };
  const scrub = async (raw: string) => {
    const f = Math.round(Number(raw));
    if (!Number.isFinite(f) || f < 1) return;
    try { await api.timelineSet(f); } catch (e) { say(msg(e), true); return; }
    const [cur, fps] = await api.timelineGet();
    showFrame(cur, fps);
    othersCache.clear(); // other objects may be time-dependent
    await recook();
  };
  // transport: Play advances `$F` at the timeline fps against the wall clock (upstream Player: a cook slower than 1/fps
  // drops frames instead of slowing the clip) and loops over the start..end range; the frame scheduler is rAF with a
  // timer fallback, so playback also runs in an occluded window. Touching the slider / box / step buttons stops it.
  const player = new Player();
  let playBusy = false;
  const playTick = frameScheduler(() => void (async () => {
    if (!player.running) return;
    const [a, b] = range();
    let want = player.frameAt(performance.now());
    if (want > b || want < a) { player.start(a, performance.now(), rate); want = a; }
    if (want !== Number(frameBox.value) && !playBusy) {
      playBusy = true;
      try { await scrub(String(want)); } finally { playBusy = false; }
    }
    if (player.running) playTick();
  })());
  const setPlaying = async (on: boolean) => {
    if (on === player.running) return;
    if (on) {
      const [a, b] = range();
      const f = Number(frameBox.value);
      player.start(f >= b || f < a ? a : f, performance.now(), rate);
      $("tplay").classList.add("on");
      playTick();
    } else {
      player.stop();
      $("tplay").classList.remove("on");
    }
  };
  const stop = () => void setPlaying(false);
  frameSlider.addEventListener("input", () => { stop(); void scrub(frameSlider.value); });
  frameBox.addEventListener("change", () => { stop(); void scrub(frameBox.value); });
  for (const r of [fStart, fEnd]) r.addEventListener("change", () => { drawTicks(); showFrame(Number(frameBox.value), rate); });
  $("tplay").addEventListener("click", () => void setPlaying(!player.running));
  $("tstop").addEventListener("click", stop);
  $("tstart").addEventListener("click", () => { stop(); void scrub(String(range()[0])); });
  $("tend").addEventListener("click", () => { stop(); void scrub(String(range()[1])); });
  $("tback").addEventListener("click", () => { stop(); void scrub(String(Math.max(range()[0], Number(frameBox.value) - 1))); });
  { const [f, fps] = await api.timelineGet(); showFrame(f, fps); drawTicks(); }
  const start = await api.startupScene();
  // a fresh session starts at /obj (Houdini), an opened scene inside its object
  if (start) { path.value = start; await open(); } else { net.level = "obj"; await refresh(); showParams(); }
  // proof / test door (read by the T1 proof script): drive the real param pane through the DOM
    (window as unknown as Record<string, unknown>).__hexe = { net, select: (name: string) => { const n = graph.nodes.find((q) => q.name === name); if (n) net.selectNode(n.id); }, recook, refresh: () => refresh(false).then(showParams), viewport, setFrame: scrub, play: setPlaying, open: async (p: string) => { path.value = p; await open(); } };
    const job = await api.renderRequest();
    if (job) await headlessRender(job);
}

/** `--render`: the scene is loaded + cooked; draw it at the requested size, hand the PNG to Rust (which writes it and exits). */
async function headlessRender(job: RenderReq): Promise<void> {
  try {
        if (!lastCook || lastCook.error) throw new Error(lastCook?.error ?? (status.textContent || "nothing cooked"));
    viewport.setWireframe(job.wire);
    await api.renderSave(viewport.snapshot(job.width, job.height));
  } catch (e) {
    await api.renderFail(msg(e));
  }
}

main().catch((e) => say(msg(e), true));
