# .nmb behaviour files: what the data gives (h46)
Clean-room, data only: 47 `.nmb` (Resources/Behaviours/M2001..M2026, M3001..M3021) + the `SecondaryBehaviour` elements of 65 `.ens`. No vendor code/binary/docs read. Probes = throwaway scripts outside the repo; counts below came from scratch XML dumps; the 135/1457/961 numbers are asserted by a test. Reader: `crates/geo/src/endorphin_behaviour.rs`.

## Verdict
| layer | status |
|---|---|
| container (zlib -> XML) | DECODED (same as all Endorphin files) |
| behaviour identity (`RefNames/Name`, 47 names) | DECODED |
| behaviour **parameter interface** (id, type, default, UI label, tooltip) | DECODED, plaintext (`ry/wx`) |
| behaviour **instance parameters in a scene** (name, type, value, label) | DECODED, plaintext (`.ens` `<SecondaryBehaviour><Parameters><Variable>`) |
| tuning constants inside the network (`wx ja=`: floats, ints, bools, `Vector(x, y, z)`, strings, dev tooltips `gu`) | DECODED, plaintext, **but attached to unknown node ids** |
| network topology / module kinds / wiring (`cc lo= lt= vw= sd=`, `xa sb=`, `wx kg=6`) | **NOT DECODED**: payload = base64 over a permuted 64-symbol alphabet, alphabet NOT recovered; byte layout unknown |
| element + attribute names (`cc hi ry wx df xa mv bh za`, `sv kg ja de gu`) | roles inferred from correlation with `.ens` (below); original names unknown (scrambled) |
Net: you can read WHICH behaviour runs, WHEN, on WHOM, with WHICH parameters. You can NOT read WHAT the network does. A behaviour must be re-authored from its name + parameter semantics (that is what endo-sim does).

## 1. Structure (measured)
Root `<cc vw sv hs gr releaseVersion>` (always 4 attrs + 3 children) -> `<RefNames><Name>`, `<hi>` (children = nested `cc`, recursion to depth ~10), `<ry>` (root parameter list), sometimes `<mv><xa/></mv>`, `<df><xa/></df>`.
- 1141 `cc` nodes in 47 files (912 `hs=0`, 186 `hs=1`, 43 without `sv/hs` = group wrappers); `sv` = id local to the file (288 distinct values; of 158 ids seen in >1 file only 8 keep the same `lo` length => NOT a global module-class id).
- `wx` (5891) = one slot/parameter. Attrs: `sv` (always), `kg` (type code), `ja` (value text), `de` (UI label), `gu` (tooltip/dev note), `numRequiredObjects`, `seed`, `bq`.
- `kg` histogram: 0 float 2751 | 4 vector 2441 | 2 bool 278 | 7 object-slot 135 | 1 int 111 | 5 selection/string 70 | 3 string-ref 55 | 6 blob 50.
  - Evidence kg 0/1/2: `ja` parses as float / integer / `true|false`; kg 4: `ja="Vector(-1.4, 0.0, 0.0)"`; kg 3: `ja="$character.Calibrations.Normal"`, `"background.MetaBodies.Prop0.Position.x"`; kg 5: `ja="MetaBodyAndCollisionSelection" | "DesiredPoseEventData" | "Randomizer"`; kg 6: `ja` == `bq` == an encoded blob; kg 7: no value, only `de`.
- Most slots (kg 4: 2155 of 2441, kg 0: 1166 of 2751) carry NO default: those are wired by the network (outputs/intermediates).

## 2. Parameter interface == `.ens` instance parameters (the decoded part)
Test over all `.ens`: 135 `SecondaryBehaviour` instances; **135/135** `behaviourName` equals a `.nmb` `RefNames/Name`; of 1457 instance params that have a UI label (`de`), **1457/1457** exist in that behaviour's root `ry` with the **same type code**; 961/1457 equal the `.nmb` default (the other 496 are user overrides).
`.ens` `Variable` attrs: `fk` = parameter key, `kg` = type (0 float, 1 int, 2 bool), `ja` = value, `de` = UI label (`_undefined` = no label). Instances also carry `startFrameNumber endFrameNumber priority enabled character`, `<Objects>`, `<TargetSelection>`.
Catalogue = `NmbInterface::from_node` over the 47 files (test `corpus_instances_match_nmb_interfaces` asserts the 135/1457/1457/961 numbers above); e.g. CatchFall 2.0 (M3012): `ArmsApart 0.5 | ArmsSpeed 0.5 | ArmsStrength 0.5 | PredictGroundHeight bool | GroundHeight`; Balance (M3015): `Balance_COM_OffsetX/Z, Balance_Pitch_Offset, Balance_PelvisHeight 0.99, Balance_UseArms true`; Stagger 2.0 (M3004): `ApplyArmsWindmill, StaggerSquarely, PredictGroundHeight, GroundHeight, MaxSteps 5, ArmsSpeed, ArmsStrength`; Arms Windmill (M3017): `ShoulderAmplitude, ElbowAmplitude, Frequency, SyncOffset, PhaseOffset, Strength`.
Semantics = from the labels + tooltips only (`gu` text, e.g. "Weights the proportional Roll component", "this will turn of hand orientating towards target and setting of hand calibrations", "setting these to true will set all calibrations of VM to 1 (i.e. off)").

## 3. Plaintext constants inside the network (Balance M3015, partial)
Unattached to node kinds, but they show the controller shape: two `cc` blocks (left/right foot or pitch/roll) with PD-style gains `ja` = 4.0 / -1.5 (block A), 3.5 / -1.75 (block B); component weights 1.0/1.0/1.0/1.0 (A) and 0.5 (P pitch) 0.85 (P roll) 0.5 (D pitch) 0.85 (D roll) (B) with tooltips "Weights the proportional Roll/Pitch component" / "... derivative ..."; `10000.0` x2 ("common to both Left and Right, not evolvables"); `Vector(0,1,0)` up axis. => balance = PD on COM/pitch/roll error. CatchFall M3012: kg 2 flags with tooltips about direct targets; `Vector(10, 0, 0)`; hand-reach nodes `ja=0.01/0.99/0.5/0.5`. Not enough to rebuild the graph.

## 4. The encoded blobs (negative result)
1918 blobs in 47 files (`vw` 47, `lo` 647, `lt` 490, `sb` 734; `kg=6` values 50) = 2.45 M chars.
- Alphabet: 64 data symbols + `O` (pad, only ever as the last 1-2 chars) + `0x7f` terminator at the end of every blob. Length minus terminator is always a multiple of 4. 4 chars = 3 bytes (24 bits): standard Base64 grouping with a **permuted** alphabet. `|` is the zero value (1.31 M of 2.45 M chars; runs `||||` = 3 zero bytes).
- Symbols used: 26 lowercase, 10 digits, `Y`, 27 punctuation; NO other uppercase (so the alphabet is not RFC4648-ordered).
- Group-aligned repeats prove byte alignment: identical 4-char groups recur at the same phase (`[*8k ?%2, zr:)` ...), prefixes like `|+aw[*8k?%2,zr:)` start many `lo` records.
- Attempts to recover the permutation (all FAILED, so no byte layout can be claimed):
  1. crib-dragging with the known plaintext strings (`de` labels, `RefNames`, 155 distinct) via 6-bit equality-pattern matching at the 3 byte phases: hundreds of false positives (patterns of repetitive records collide), no mutually consistent mapping.
  2. hill-climb on a printable-ASCII score over unique 4-groups, 6 restarts x 40k swaps: scores 354k-371k, no agreement between restarts, output unreadable.
  3. simulated annealing against a byte-bigram model of the sample-scene identifiers (joint/object names), 8 restarts x 60k: scores -914k..-917k, the runs agree only on the first 3 values (`|`=0, `s`=1, `+`/`$` = 2/3 interchangeably: small-integer symbols), nothing beyond.
  Likely cause: the payload is mostly small-integer binary records (ids, counts, floats), not text, so a text prior cannot lock the alphabet; the string records (names) are too few. Next step if wanted: derive the alphabet from structure (e.g. `sb` of fixed-size records of known counts) or from the `kg=6` values whose decoded form is a count-prefixed list — needs a known plaintext anchor that this corpus does not give.

## 5. What this buys the simulation
Behaviour layer (endo-sim `behaviour.rs`) takes: behaviour name (selects a hand-authored controller in Rust), `[start,end]` frames, `character`, priority, and the decoded parameters (`ArmsStrength`, `ArmsSpeed`, `ArmsApart`, `PredictGroundHeight`, `GroundHeight`, ...). The per-joint motor stiffness/damping/force limit come from the character's own `<Calibrations>` (`SkeletalEffectorParameters` swing/twist stiffness+damping+forceLimit, already decoded in the `.nmc/.ens`), not from the `.nmb`.
