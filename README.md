# hexe

A Rust port of Houdini, with Endorphin (NaturalMotion's ragdoll / behaviour simulator) as a built-in plugin.
Desktop app = Tauri 2 + three.js viewport; core = Rust crates; parity is measured against the real Houdini 20.5 (`hython` oracle) and the real `endorphin.exe` (rizin disassembly).

## State (2026-10-23, main `e811e06`)
- **SOPs:** 94 registered (`crates/sop/src/builtins/mod.rs`) → per-SOP params / parity / example: [docs/SOPS.md](docs/SOPS.md).
- **Houdini oracle:** 144 MATCH · 12 STAT-PASS · 1 TOPO-DIFF of 157 cases (boolean split-poly start vertex) → [docs/HOUDINI-ORACLE.md](docs/HOUDINI-ORACLE.md).
- **Files:** `.hip` scene import (nested /obj containers), bgeo (binary-JSON + legacy `BgeoV`) read + write, `.geo`.
- **Examples:** 52 scenes in `examples/*.json`.
- **Endorphin (`endosim` SOP, `crates/endo-sim`, rapier3d):** opens and PLAYS `.ens` scenes — all 65 installed sample / behaviour / tutorial scenes play 300 frames finite, no NaN / explosion / sinking (`proof/h47/table.tsv`); 600-frame runs of Balcony, Car Hit, Car Crash, Gymnast Rings, Sword Play are deterministic (`proof/h57d` run1 == run2).
  - Authored events: DesiredPose, Animation clips, Incarnation (kinematic → full physics handover with authored velocity), ForceEvents, Lock / JoinBodies holds, scene `<Animation>` channels drive cars / props / mocap actors.
  - Behaviours: CatchFall, Stagger, Arms Windmill, Legs Straighten, pose hold, ankle balance (4 of 47; rest parsed, not simulated).
  - Taken from the exe: ForceEvent impulse (`applyImpulseViaVelocity`), hold strength (Karma row force bound), per-surface-pair materials (friction / restitution / softness / adhesion), self-collision rules.
  - Open: `.nmb` behaviour networks not decoded (controllers re-authored); `<Constraint>` attachments (ropes, rings) not built; hold softness mapping (Karma ε → rapier) unread.
- **motion SOP:** chain / push / hitreact modes via gaia-motion → [docs/MOTION-CHAIN.md](docs/MOTION-CHAIN.md).

## Layout
| path | what |
|---|---|
| `crates/core`, `geo`, `graph`, `vex` | geometry (Detail), file formats, node graph + cook, VEX |
| `crates/sop` | SOP library |
| `crates/endo-sim` | Endorphin simulation |
| `crates/sop-motion` | motion SOP |
| `apps/hexe-tauri` | desktop app |
| `apps/hexe-cli` | headless `open` / `cook` |
| `oracle/`, `parity/` | Houdini + Endorphin oracles, feature ledger (`parity/features.toml`) |
| `proof/h46..h57d` | render + number proofs per milestone |

## Build / run
```sh
cd apps/hexe-tauri && bun install && bun run tauri build      # → target/release/bundle/macos/Hexe.app
target/release/hexe-tauri examples/<scene>.json --render out.png --size 640x360   # headless render
cargo run --release -p hexe-cli -- open <scene.ens>            # Endorphin scene → file → endosim graph
```

History per milestone: [STATUS.md](STATUS.md) · plan: [ROADMAP.md](ROADMAP.md) · rules: [AGENTS.md](AGENTS.md) · licences: [ATTRIBUTION.md](ATTRIBUTION.md).
