#!/bin/sh
# h57c: rebuild clearance, then h56-settings noise runs (gamma 1 eps 1 +-1e-6) -> proof/h57c
cd "$(dirname "$0")/../.." || exit 1
D=proof/h57c
cargo build -q --release -p endo-sim --example clearance > $D/build.log 2>&1 || { echo FAIL > $D/n.done; exit 1; }
HOLD_GAMMA=1 HOLD_EPS=1 sh proof/h56/run.sh $D/r_g1e1.txt
for g in 1.000001 0.999999 1.000002; do HOLD_GAMMA=$g HOLD_EPS=1 sh proof/h56/run.sh $D/r_h56g_$g.txt; done
echo ok > $D/n.done
