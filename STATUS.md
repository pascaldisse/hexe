# STATUS: h47 (Endorphin scene PLAY path, branch h45/endo-behaviours, 2026-10-21)
- PLAYS the authored scene, not just a ragdoll: `Sim::from_scene_play` (= `SimOptions{behaviours,forces,secondary}` all on) adds, on top of h46 CatchFall + ForceEvents, the `behave.rs` layer (pose hold servos, ankle balance, Stagger / Arms Windmill / Legs Straighten, catch-fall brace) and the authored event path: `<DesiredPoseEvent>` poses as servo targets, `<AnimationEvent>` clips (clip time = frameOffset + time since the event start, 255 = no offset) kinematic until the `<IncarnationEvent>` FullPhysics frame, then a velocity handover (finite difference over ONE ens frame).
- Scene timeline: the scene's own `<Animation>` channels drive any character whose first IncarnationEvent is not FullPhysics (imported cars / props / mocap actors) kinematically, so `Car Hit`'s car reaches character01 instead of free-falling; single-body characters now get a controller too.
- endosim SOP param `behaviours`: 0 passive, 1 CatchFall + forces, 2 (default) play.
- S4/S5 corpus: ALL 65 installed `.ens` (Sample + Behaviour Scenes + Tutorials) played 300 frames through `cargo run --release -p endo-sim --example corpus` — 65/65 finite, no panic, no NaN, no explosion (|x| < 1e3), nobody sinking through the scene ground; per-scene rows (events decoded, frames, final root/head, contacts, minY) in proof/h47/table.tsv.
- Proof renders: proof/h47/{Balcony,Car_Hit,Gymnast_Rings,Sword_Play}_play_F{1,30,60,90,300}.png + graph JSONs (`sh proof/h47/render.sh`, hexe-tauri `--render`, needs `--features tauri/custom-protocol`).
- Tests: `cargo test -p endo-sim` 12 green (timeline drive, FullPhysics handover velocity, Car Hit car height + contacts, CatchFall A/B, determinism, Balcony fall).
- UNVERIFIED / still differs from Endorphin: controllers are authored, NOT NaturalMotion's (no .nmb network, no tuned solver); only CatchFall / Stagger / Windmill / LegsStraighten of 47 behaviours are simulated, the rest are parsed and ignored; Constraints (`<Constraint>` inside AnimationEvents, rope/ring attachments) are NOT built; `momentumScale`, CollisionOnly vs CollisionOnlyWithInertia are treated alike; no stepping; visuals NOT judged by a human.
---
# STATUS: h46 (Endorphin behaviours, branch h46/nmb-behaviours, 2026-10-20)
- oracle/endorphin/NMB.md: .nmb plaintext interface DECODED (47 names, root params, types, defaults, tooltips; 135/135 scene instances, 1457/1457 params match); network blobs (`lo lt vw sd sb`) = base64 over a permuted alphabet, NOT decoded (3 attempts failed). Behaviours are therefore re-authored, not interpreted.
- crates/geo/src/endorphin_behaviour.rs: NmbInterface, BehaviourInstance, ForceEvent, `Scene::behaviour_instances/force_events`.
- crates/endo-sim/src/behaviour.rs: ONE behaviour = `Bob2: CatchFall` (2.0 + 2.5-era name): arms reach for predicted hand-ground point, elbows extend, head tuck; PD drive with gains from the character `Normal` calibration, strength/speed/apart/ground params from the scene. `Sim::from_scene_active` / `SimOptions`; endosim SOP param `behaviours` (default 1; 0 = passive). All other behaviours ignored.
- Proof: proof/h46/{passive,active}_F{1,30,60,90}.png + graph JSONs (F1 identical, F30/60/90 differ). Play: `hexe-tauri proof/h46/catchfall_active.json` (build with `--features tauri/custom-protocol`, plain cargo build loads devUrl).
- UNVERIFIED: controller is authored (not NaturalMotion's); gain units, ForceEvent magnitude mapping (relativeAcceleration x |Vk| x g), reach constants (REACT/TUCK/RATE_*) are guesses; visuals not judged; Play button not clicked.
---
# STATUS — h44 (Endorphin simulation, branch h44/endo-sim, 2026-10-18)
- PASSIVE RAGDOLL ONLY: behaviours / balance / catch-fall muscle controllers / drives are OUT of scope and NOT simulated; characters fall + slump under gravity within joint limits. (Behaviour mode of `motion` SOP unchanged.)
- crates/endo-sim (rapier3d 0.35 f32, L1): `Sim::from_scene(&Scene)`, `step(dt)` (substeps <= 1/120), `pose() -> Vec<CharPose{name, world: Vec<Xf>}>`, `root_y(name)`. One dynamic body per MetaBody (Sphyl/Sphere/Cylinder/Cuboid; File/Mesh -> bbox capsule; mass = density x inertial volume), spherical joints (hard else soft limits, symmetric half-ranges about the start pose, twist about joint X) / fixed where no limits, Environment bodies static, default ground = static slab top y=0, same-character collisions off. Balcony: 151 bodies, 108 joints.
- `endosim` SOP (crates/sop): input file SOP on .ens (via detail attr `endo_file`), param `frame` 0 = follow `$F` (frame F = F-1 steps of 1/fps), poses cached per frame, scrub back = replay; restart on file/fps change. Output = endorphin_geo::scene_detail_posed.
- UI: Play/Stop button beside the timeline (ui/src/play.ts wall-clock clock, frame scheduler = rAF + timer fallback; slider/box touch stops). `hexe-cli open <scene.ens>` graph = file -> endosim.
- OPEN/UNVERIFIED: rapier is not NaturalMotion's solver (no tuned stiffness/damping, no soft-limit springs, twist axis inferred from limb data); seated characters rest on the set only via their own start pose; Play/Stop button logic bun-tested (Player clock) but not clicked in a live window; render proof uses graph file->endosim->blast(char_background ground) so Fit frames the characters.
---
# STATUS: app = Tauri (T2a)

- Desktop app = `apps/hexe-tauri` (Tauri 2: Rust commands over hexe-graph + vite/TS UI, three.js viewport). egui `apps/hexe` DELETED (last at 805b5aa).
- Bundle: `cd apps/hexe-tauri && bun run tauri build` -> `target/release/bundle/{macos/Hexe.app,dmg/Hexe_0.1.0_aarch64.dmg}`; ad-hoc `codesign --force --deep --sign -`.
- egui->Tauri port checklist: apps/hexe-tauri/PORT-TODO.md (0 open).

---

# STATUS — Oracle 2026-10-11

- Oracle 2026-10-11: 143 MATCH / 1 TOPO-DIFF (boolean split-poly start vertex) / 12 STAT-PASS over 156 (N=156); 79 SOPs; /Applications/Hexe.app = h38 merge (sops-b + sops-c), built from main HEAD.
- Oracle 2026-10-12 (H35b): histogram unchanged (143/12/1). h33/order-twins = already on main (e039ca3, rebase = no-op); h34/model-twins merged (probes only, dc956f8). sop-boolean start vertex NOT derived: `cargo xtask boolstart` table (best 42/128), hython sweep parity/ref/boolean/start-sweep.txt (start flips with offset at identical topology) -> TOPO-DIFF stays, no per-case hack.

---

# STATUS — h1/bgeo (bjson decoder, WIP salvage)

- crates/geo/src/bjson.rs: binary-JSON (`0x7f NSJb`) → serde Value; json_read.rs: p_r/c_r/run prim abbreviations → Detail.
- Probe over Houdini 20.5.487 install (tests/bjson_tests.rs, path /Applications/Houdini/...; no-op when absent): 160 files; 153 bjson → 153/153 value-ok, 153/153 detail-ok; 7 NOT bjson (legacy `BgeoV` magic 42 67 65 6f 56 00 00 00: slug, mountain, carousel, fabric, head_poly, GPD/all.bgeo, deflamp) → legacy reader: see below.
- Unknown token bytes encountered: NONE (all 153 decode). Decoder errs on unknown token / length prefix / uniform elem (bjson.rs).
- Failing fixtures: none in bjson set. Legacy BgeoV: 7/7 read (H13+H39), counts + P bbox == Houdini (fixtures/bgeo-legacy/counts.json, hython bgeo-legacy-counts.py); bgeo_binary_reads un-ignored.
- fixtures/bgeo-bjson/: 8 files (axis cross ctrl-box ctrl-disc-xy ctrl-planes-xy sg_geolight simple_sphere tree0) + MANIFEST rows; golden tests (H39): crates/geo/src/bjson_golden_tests.rs vs hython-saved .geo twins (counts + vertex map + P max|dP| 0).
- Prim types seen: p_r 42, c_r 90, run 16 (PackedGeometry 55, Mesh/NURBMesh/Circle/Sphere/Tube/Bezier/NURBCurve/PackedDisk) — only polygon/curve runs mapped to Detail; others pass-through/unverified.
- Writer round trip on the 8 fixtures: bjson_golden_tests::writer_round_trips_all_shipped_fixtures (H39).
- TODO: 84-fixture target unmeasured.

---

# STATUS — Endorphin hitreact oracle 2026-10-12 (branch endo/finish)

- Oracle row `sop-motion-hitreact` (oracle/houdini/sop-motion-hitreact.py): hexe `motion` mode=hitreact seed 12 f30 vs gaia-motion HDA `gaia::motion::1.0` hit_reaction cooked in hython 20.5.487 → MATCH (24/23/46, max|dP| 0.0, name + pelvis_z equal). Histogram 144 MATCH / 12 STAT-PASS / 1 TOPO-DIFF (N=157).
- OPEN: reader-vs-reader only (both read the same MuJoCo CLI sim, no Houdini-native twin of the hit reaction); scene motion_root = `$GAIA_MOTION_ROOT` (env, default `~/projects/gaia-motion` main); re-record if hitreact.py changes seed-12 f30. Details: gaia-motion docs/STATUS.md.

---

# STATUS — h39 (trunk 2026-10-14)

- SOPs shipped (hython-twin MATCH): groupinvert, groupexpand, attribswap, edgedivide. Registry = 84 types (`hexe-cli nodes`, SOPS.md regen).
- bgeo: bjson decoder + legacy BgeoV reader + writer round trip closed (see h1/bgeo above).
- `trail` BLOCKED: needs time-history cook API. `align`, `bulge` open.
- Oracle 2026-10-14: MATCH 155 / STAT-PASS 12 / TOPO-DIFF 1 (boolean start vertex) / HEXE-ERR 1 (sop-motion-hitreact: gaia-motion venv python missing on box = env, not engine) over N=169.
- ci GREEN (12 steps) after clippy fixes 9af5672; parity OK 103 rows.
---
# STATUS — h40+h41 trunk (merged 2026-10-14)
- hip: 12/12 parse; cook vs hython 13 MATCH / 25 geos; 5/12 files fully MATCH. endorphin: 207/207 parse, rest-pose only, .nmb opaque.
- `hexe-cli open <file.hip|.ens|.nmc|.nmb|.nma> [format flags]` = ONE dispatch by extension (case-insens): .hip/.hipnc/.hiplc/.hip.gz → open::open_cmd [--geo N --scene-out --cook-out --all-out]; .ens/.nmc/.nmb/.nma → endo::open_cmd [--out --geo]; unknown ext → 1-line error listing both; `open --help` prints per-format flags. (h40 line above `open <f.hip> [--out dir]` superseded.)
- vendored third_party/tauri-debug-server: clippy-clean + registered Exempt in xtask layers. ci GREEN 12 steps.

---

# STATUS — h43 (scene pose, branch h43/pose, 2026-10-14)
- Endorphin scene pose rule: rig rest ← `<Poses>` full parent-local matrix ← `<Animation>` channel; Character.TransformMatrix places each character (oracle/endorphin/SCENE_POSE.md); quat order on disk = wxyz; keys sampled via file SOP `endo_time` (s, <0 = $T); viewport Fit excludes groups matching `fitExclude` glob (default `*background*`) so characters frame separated.

# STATUS — h42 (viewport, branch h42/viewport, 2026-10-14)
- Hidden-window rAF pause → frameScheduler rAF∨timer 50 ms default (ui/src/frame.ts + tests); proof/h42-balcony-after.png. Fit still counts `char_background` slab = open.

---

# STATUS — h41 (Open Endorphin files, branch h41/endo, 2026-10-14)
- Corpus (CrossOver bottle, endorphin 2.5.2 Resources): 65 .ens, 52 .nma, 47 .nmb, 43 .nmc (41 `Character` + 2 bare `Skeleton` roots); NO .nmp/.nmw/.nmm/.nme present (task listed 67/44/5/2/1). 207/207 parse. Vocabulary: 156 element names, 657 (element, attr) pairs → oracle/endorphin/SURVEY.md (generated: `hexe-cli endo-survey <Resources> --out oracle/endorphin/SURVEY.md`).
- Code: crates/geo/src/endorphin.rs (inflate, xml, bfloat, typed Scene/Character/Joint/Body/Behaviour/Pose/Channel, raw trees kept), endorphin_geo.rs (rest-pose Detail), io.rs (`read_any` .ens .nmc .nmb .nma read-only), apps/hexe-cli/src/endo.rs (`open`, `endo-survey`; `cook <file.ens>` direct).
- Output geo: bones = open 2-pt polys (joint→parent); joint points attrs joint/jtype/character/parent/{twist,swing1,swing2}_{soft,hard}; body meshes = closed polys prim attrs kind/body/material/shape/mass/character; prim groups per kind + `char_<name>`; detail attrs endo_kind/endo_gravity/endo_behaviours/endo_animation/endo_poses (JSON). Default ground = 1 quad.
- Proof: proof/h41/{Balcony,Car_Hit,Gymnast_Rings,Sword_Play}.png (installed /Applications/Hexe.app `--render` over `open --geo` output, environment group blasted).
- OPEN: rest pose only (anim quats/Poses not applied); .nmb blobs undecoded; File/Mesh geometry = placeholder box; Sphyl/Cylinder axis + Cuboid extents conventions inferred; UI File>Open route is source-only until next Tauri build; mass of File/Mesh bodies = 0.

---

# STATUS — h40 (.hip reader, branch h40/hip)

- `io.hip-read` (partial): `crates/graph/src/hip.rs` cpio odc/newc + gzip → IR; `crates/sop/src/hip_import.rs` IR → hexe scenes (per /obj geo); `hexe-cli open <f.hip> [--out dir]`; Tauri File→Open (`Session::graph_load` routes .hip/.hipnc/.hiplc/.hip.gz → first geo, no UI rebuild).
- Corpus 12 Houdini 20.5 help/files hips (`nodes/sop/**` ships 0 hips) → fixtures/hip/MANIFEST.md: parse 12/12 FULL; cook 25 geos = 13 MATCH / 5 HEXE-ERR / 7 UNSUPPORTED-NODE (hython `oracle/houdini/hip-twin.py`); files with ≥1 MATCH 7/12, fully MATCH 5/12. 11/13 MATCH = 1-node object geos.
- Gaps (ordered by corpus hits): `copy` stamp, for-each block_begin/end, `point` SOP, `grouprange`, `material`, `object_merge`, HDAs (log_spiral, color_falloff), wrangle `relbbox`, tube non-poly.
- Not oracle-covered: MIME-wrapped hips (error), animated parms (value at save time).
