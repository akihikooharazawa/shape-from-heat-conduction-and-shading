#!/bin/bash
# Fetch the thermal videos that notebooks/real.ipynb reads.
#
# The object masks ship with this repository, so only the recordings are
# downloaded. They are attached to a GitHub release rather than committed,
# because they are tens to hundreds of megabytes each.
#
# heart, droplet and paintedbear were captured for this paper and are in the
# release. bear, bunny, soda_can and sphere come from the shape-from-heat
# dataset and are several hundred megabytes each, so they are not re-hosted
# here; asking for one prints where to get it.
#
#   ./scripts/download_data.sh                 # paintedbear, the default object
#   ./scripts/download_data.sh bear sphere     # the objects you name
#   ./scripts/download_data.sh --all           # every object
#   ./scripts/download_data.sh --list          # what is available, and its size
#
# REPO and TAG override where the files come from, e.g. to pull from a fork:
#
#   REPO=someone/their-fork TAG=data-v2 ./scripts/download_data.sh --all

set -e
cd "$(dirname "$0")/.."

REPO="${REPO:-akihikooharazawa/shape-from-heat-conduction-and-shading}"
TAG="${TAG:-data-v1}"
BASE="https://github.com/$REPO/releases/download/$TAG"
DEST="DATA/obj_study_raw"

if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required" >&2
    exit 1
fi

# Resolve object names to the recordings they need, straight from the config
# the notebook uses, so the two can never drift apart.
objects_for() {
    python3 - "$@" <<'PY'
import sys
from config.real_objects import OBJECTS, get_config

args = sys.argv[1:]
if not args or args == ["--default"]:
    names = ["paintedbear"]
elif args == ["--all"]:
    names = sorted(OBJECTS)
else:
    unknown = [a for a in args if a not in OBJECTS]
    if unknown:
        sys.exit(f"unknown object(s): {', '.join(unknown)}; "
                 f"choose from {', '.join(sorted(OBJECTS))}")
    names = args

wanted = []
for name in names:
    for video in get_config(name).videos:
        if video not in wanted:
            wanted.append(video)
print("\n".join(wanted))
PY
}

if [ "$1" = "--list" ]; then
    echo "Objects: $(python3 -c 'from config.real_objects import OBJECTS; print(" ".join(sorted(OBJECTS)))')"
    echo
    for f in $(objects_for --all); do
        size=$(curl -fsIL "$BASE/$f" 2>/dev/null \
               | awk 'BEGIN{IGNORECASE=1} /^content-length:/{n=$2} END{print n+0}')
        if [ "$size" -gt 0 ]; then
            printf "  %-20s %6.1f MB\n" "$f" "$(echo "$size/1048576" | bc -l)"
        else
            printf "  %-20s not in release %s\n" "$f" "$TAG"
        fi
    done
    exit 0
fi

mkdir -p "$DEST"
files=$(objects_for "$@")

echo "Downloading from $BASE into $DEST"
for f in $files; do
    if [ -s "$DEST/$f" ]; then
        echo "  have     $f"
        continue
    fi
    echo "  fetching $f"
    # -C - resumes a partial file, so an interrupted run can just be repeated.
    if ! curl -fL --retry 3 -C - --progress-bar -o "$DEST/$f" "$BASE/$f"; then
        rm -f "$DEST/$f"
        case "$f" in
            bear-*|bunny.npz|soda_can-*|sphere-*)
                echo "  not in release: $f" >&2
                echo "    This one was captured for the shape-from-heat dataset and is" >&2
                echo "    distributed there rather than re-hosted here:" >&2
                echo "    https://github.com/nnsriram97/shape-from-heat" >&2
                echo "    Download it, then run scripts/prepare_data.py to give it the" >&2
                echo "    short name above (SOURCE_RECORDINGS says which capture it is)." >&2
                ;;
            *)
                echo "  FAILED   $f  (not in release $TAG, or no network)" >&2
                ;;
        esac
    fi
done

echo
echo "In $DEST:"
ls -1sh "$DEST" | grep -v '^total' | grep -v '\.gitkeep' || echo "  (nothing yet)"
