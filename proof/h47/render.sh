#!/bin/sh
# h47 proof: play four Endorphin scenes through hexe's own headless render path and shoot F1/30/60/90/300 (environment = `char_background` kept).
# Usage: sh proof/h47/render.sh   (repo root; needs target/release/hexe-tauri built with --features tauri/custom-protocol)
set -e
ROOT="$HOME/Library/Application Support/CrossOver/Bottles/endorphin/drive_c/Program Files (x86)/NaturalMotion/endorphin 2.5.2/Resources/Scenes/Sample Scenes"
BIN="${CARGO_TARGET_DIR:-target}/release/hexe-tauri"
OUT=proof/h47
for row in "Balcony:Balcony" "Car Hit:Car_Hit" "Gymnast Rings:Gymnast_Rings" "Sword Play:Sword_Play"; do
  SRC="${row%%:*}"; NAME="${row##*:}"
  G="$OUT/${NAME}_play.json"
  printf '{"nodes":[{"name":"endo","type":"file","params":{"file":"%s/%s.ens"}},{"name":"sim","type":"endosim","params":{"behaviours":2}}],"edges":[["endo",0,"sim",0]],"display":"sim"}\n' "$ROOT" "$SRC" > "$G"
  for F in 1 30 60 90 300; do
    "$BIN" "$G" --frame "$F" --fps 30 --size 800x450 --render "$OUT/${NAME}_play_F${F}.png"
  done
done
ls -l "$OUT"
