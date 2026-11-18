#!/bin/bash
# Build the whole course dataset from LUNA16, ONE SUBSET AT A TIME.
#
#   ./build_dataset.sh /data/luna_work luna16-lung-slices            # all ten subsets
#   ./build_dataset.sh /data/luna_work luna16-lung-slices "0 1 2"    # only some (always start from 0)
#
# For each subset: download it from Zenodo, extract it, add its slices and its nodule patches to the
# dataset, delete the scans.  Peak disk use is about 20 GB instead of 130 GB.  It can be interrupted
# and started again: finished subsets are skipped, an interrupted subset starts again from its download.
# Needs: curl, unzip, md5sum, python with numpy pandas nibabel SimpleITK pydicom tqdm.
set -euo pipefail

WORK=${1:?work folder for the temporary LUNA16 files}
OUT=${2:?output dataset folder}
SUBSETS=${3:-"0 1 2 3 4 5 6 7 8 9"}
VOLUMES=${VOLUMES:-28}          # low-resolution 3D scans for notebook 05 (taken from the first subset)
CONNECTIONS=${CONNECTIONS:-16}  # parallel connections per download
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$WORK" "$OUT"

record_of() { if [ "$1" -le 6 ]; then echo 3723295; else echo 4121926; fi; }

# fetch <record> <file name> : parallel download with resume, then md5 check
fetch() {
  local rec=$1 name=$2 url size md5 chunk i s e
  read -r size md5 < <(curl -sS --retry 5 "https://zenodo.org/api/records/$rec" | python3 -c "
import json, sys
f = [f for f in json.load(sys.stdin)['files'] if f['key'] == '$name'][0]
print(f['size'], f['checksum'].replace('md5:', ''))")
  url="https://zenodo.org/api/records/$rec/files/$name/content"
  rm -f "$WORK/$name" "$WORK/$name".part??
  chunk=$(( (size + CONNECTIONS - 1) / CONNECTIONS ))
  for i in $(seq 0 $((CONNECTIONS - 1))); do
    s=$((i * chunk)); e=$((s + chunk - 1)); [ "$e" -ge "$size" ] && e=$((size - 1))
    [ "$s" -ge "$size" ] && continue
    curl -sS -L --retry 20 --retry-delay 5 --retry-all-errors -r "$s-$e" \
         -o "$WORK/$name.part$(printf %02d "$i")" "$url" &
  done
  wait
  : > "$WORK/$name"
  for p in "$WORK/$name".part??; do cat "$p" >> "$WORK/$name"; rm "$p"; done
  echo "$md5  $WORK/$name" | md5sum -c - >/dev/null || { echo "checksum of $name is wrong: delete it and run again"; exit 1; }
}

# things needed once: lung masks and nodule annotations
if [ ! -d "$WORK/seg-lungs-LUNA16" ]; then
  fetch 3723295 seg-lungs-LUNA16.zip
  unzip -q -o "$WORK/seg-lungs-LUNA16.zip" -d "$WORK" && rm "$WORK/seg-lungs-LUNA16.zip"
fi
[ -f "$WORK/annotations.csv" ] || fetch 3723295 annotations.csv

for k in $SUBSETS; do
  if [ -f "$OUT/.subset$k.done" ]; then echo "subset $k: already done"; continue; fi
  echo "=== subset $k: download ($(date +%H:%M))"
  if [ ! -d "$WORK/subset$k" ]; then
    fetch "$(record_of "$k")" "subset$k.zip"
    # extract in a temporary folder and rename at the end: an interrupted extraction is never taken for a complete one
    rm -rf "$WORK/extracting"; mkdir "$WORK/extracting"
    unzip -q -o "$WORK/subset$k.zip" -d "$WORK/extracting"
    mv "$WORK/extracting/subset$k" "$WORK/subset$k"
    rm -rf "$WORK/extracting" "$WORK/subset$k.zip"
  fi
  echo "=== subset $k: slices"
  python3 "$HERE/prepare_luna16_slices.py" --luna-dir "$WORK" --out "$OUT" --subsets "$k" --append --volumes "$VOLUMES"
  echo "=== subset $k: nodules"
  python3 "$HERE/prepare_nodule_patches.py" --luna-dir "$WORK" --out "$OUT"
  rm -rf "$WORK/subset$k"
  touch "$OUT/.subset$k.done"
done

python3 "$HERE/prepare_luna16_slices.py" --check --out "$OUT"
echo "Dataset ready in $OUT. Zip it and upload it to Kaggle (see README.md)."
