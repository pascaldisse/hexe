#!/bin/sh
# usage: run.sh out.txt  (env knobs pass through)
S="$HOME/Downloads/Endorphin 2.5.2 Full Version/PATCH/Resources/Scenes/Sample Scenes"
for p in "Balcony:Balcony" "Car:Car Hit" "CarCrash:Car Crash" "Gymnast:Gymnast Rings" "Sword:Sword Play"; do
  n=${p%%:*}; f=${p#*:}; ./target/release/examples/clearance "$S/$f.ens" 2 600 2>/dev/null | sed "s/^/$n /"
done > "$1"
