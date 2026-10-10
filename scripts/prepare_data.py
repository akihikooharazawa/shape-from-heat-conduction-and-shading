#!/usr/bin/env python3
"""Rename downloaded captures to the short names the notebooks refer to.

``config/real_objects.py`` refers to each recording by a short name
(``bear-0.npz``, ``paintedbear.npz``, ...). The captures themselves carry the
name they were recorded under, and ``SOURCE_RECORDINGS`` maps one to the other.
This script applies that mapping to ``DATA/obj_study_raw`` and renames the
matching mask in ``DATA/obj_study_proc`` along with it.

Running it twice is harmless: anything already named correctly is left alone.
"""

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from config.real_objects import SOURCE_RECORDINGS  # noqa: E402

RAW_DIR = os.path.join(REPO, "DATA", "obj_study_raw")
PROC_DIR = os.path.join(REPO, "DATA", "obj_study_proc")
MASK_SUFFIXES = ("_mask.png", "_mask.pkl")


def rename(src_dir, src_name, dst_name, dry_run):
    """Rename one file, reporting what happened."""
    src = os.path.join(src_dir, src_name)
    dst = os.path.join(src_dir, dst_name)
    rel = os.path.relpath(src_dir, REPO)

    if os.path.exists(dst):
        return "present", f"{rel}/{dst_name}"
    if not os.path.exists(src):
        return "missing", f"{rel}/{dst_name}"
    if not dry_run:
        os.rename(src, dst)
    return "renamed", f"{rel}/{src_name} -> {dst_name}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change without touching files")
    args = parser.parse_args()

    counts = {"renamed": 0, "present": 0, "missing": 0}
    missing = []

    for short_name, original in sorted(SOURCE_RECORDINGS.items()):
        state, message = rename(RAW_DIR, original, short_name, args.dry_run)
        counts[state] += 1
        if state == "missing":
            missing.append(short_name)
        else:
            print(f"  {state:8s} {message}")

        # The mask is named after the video it belongs to.
        src_stem = os.path.splitext(original)[0]
        dst_stem = os.path.splitext(short_name)[0]
        for suffix in MASK_SUFFIXES:
            if os.path.exists(os.path.join(PROC_DIR, src_stem + suffix)) or \
                    os.path.exists(os.path.join(PROC_DIR, dst_stem + suffix)):
                state, message = rename(PROC_DIR, src_stem + suffix,
                                        dst_stem + suffix, args.dry_run)
                print(f"  {state:8s} {message}")

    print(f"\n{counts['renamed']} renamed, {counts['present']} already in place, "
          f"{counts['missing']} not downloaded")
    if missing:
        print("Not found: " + ", ".join(missing))
        print("Set RAW_URL/MASK_URL in scripts/download_data.sh to a dataset "
              "folder that holds these captures.")


if __name__ == "__main__":
    main()
