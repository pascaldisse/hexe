#!/bin/sh
# h57c A/B on Balcony: (gamma,eps) x HOLD_PERTURB {none,1e-6,-1e-6,2e-6} -> proof/h57c/ab_g<g>_e<e>.txt
cd "$(dirname "$0")/../.." || exit 1
S="$HOME/Downloads/Endorphin 2.5.2 Full Version/PATCH/Resources/Scenes/Sample Scenes/Balcony.ens"
for ge in 0.2:0.01 1:0.01 0.2:1 1:1; do g=${ge%%:*}; e=${ge#*:}; o=proof/h57c/ab_g${g}_e${e}.txt; : > $o
  for p in "" 1e-6 -1e-6 2e-6; do echo "# perturb=${p:-0}" >> $o
    HOLD_GAMMA=$g HOLD_EPS=$e HOLD_PERTURB=$p ./target/release/examples/clearance "$S" 2 600 2>/dev/null >> $o; done; done
echo ok > proof/h57c/ab.done
