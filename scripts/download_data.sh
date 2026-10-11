#!/bin/bash
# Fetch the thermal videos and object masks used by notebooks/real.ipynb.
#
# bear, bunny, soda_can and sphere were captured for the shape-from-heat
# dataset and are kept in its public folders, which are the defaults below.
# heart, droplet and paintedbear were captured for this paper; point RAW_URL
# and MASK_URL at the dataset folder for this project to get those as well.
#
#   RAW_URL=<drive folder> MASK_URL=<drive folder> ./scripts/download_data.sh
#
# Afterwards the captures are renamed to the short names that
# config/real_objects.py refers to (see SOURCE_RECORDINGS there).

set -e
cd "$(dirname "$0")/.."

RAW_URL="${RAW_URL:-https://drive.google.com/drive/folders/1PBby4Sja-j1e8alLoX8eFvUw0BH-LCcd}"
MASK_URL="${MASK_URL:-https://drive.google.com/drive/folders/1yelMdtds9VTstfxDjqux0K6vlZeAJyFQ}"

if ! command -v gdown >/dev/null 2>&1; then
    echo "gdown is required: pip install gdown" >&2
    exit 1
fi

mkdir -p DATA/obj_study_raw DATA/obj_study_proc

echo "Downloading thermal videos into DATA/obj_study_raw ..."
gdown --folder "$RAW_URL" -O DATA/obj_study_raw

echo "Downloading masks into DATA/obj_study_proc ..."
gdown --folder "$MASK_URL" -O DATA/obj_study_proc

echo
python3 scripts/prepare_data.py
