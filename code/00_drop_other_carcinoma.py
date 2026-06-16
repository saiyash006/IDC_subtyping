#!/usr/bin/env python3
"""
00_drop_other_carcinoma.py
==========================
Remove 'Other type' carcinoma patients from the BCNB pipeline.

This script cleans 3 artefacts:
  1. Split ID files   — removes IDs from train_id.txt / val_id.txt / test_id.txt
  2. Feature .pt files — deletes extracted embeddings
  3. Patch directories — deletes tile folders

Usage:
    # Preview what will be removed (safe — changes nothing):
    python code/00_drop_other_carcinoma.py

    # Actually remove everything:
    python code/00_drop_other_carcinoma.py --execute
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────────────────────
DATASET_ROOT   = Path("/home/cgntlab5/bcnb/BCNB dataset")
CLINICAL_XLSX  = DATASET_ROOT / "patient-clinical-data.xlsx"
SPLIT_DIR      = DATASET_ROOT / "dataset-splitting"
FEATURES_ROOT  = DATASET_ROOT / "outputs" / "features"
PATCHES_ROOT   = DATASET_ROOT / "outputs" / "patches"

SPLITS = ["train", "val", "test"]
TUMOUR_TYPE_COL = "Tumour Type"
DROP_VALUE      = "Other type"


def get_other_ids() -> set:
    """Return the set of patient IDs (str) with 'Other type' carcinoma."""
    df = pd.read_excel(str(CLINICAL_XLSX))
    df.columns = df.columns.str.strip()
    mask = df[TUMOUR_TYPE_COL].str.strip() == DROP_VALUE
    return set(df.loc[mask, "Patient ID"].astype(str).tolist())


def clean_split_files(other_ids: set, dry_run: bool) -> int:
    """Remove Other-type IDs from the split .txt files."""
    total_removed = 0
    for split in SPLITS:
        path = SPLIT_DIR / f"{split}_id.txt"
        if not path.exists():
            print(f"  [SKIP] {path} does not exist")
            continue

        with open(path) as f:
            all_ids = [line.strip() for line in f if line.strip()]

        keep = [pid for pid in all_ids if pid not in other_ids]
        removed = len(all_ids) - len(keep)
        total_removed += removed

        print(f"  [{split}] {len(all_ids)} → {len(keep)} IDs  (dropping {removed})")

        if not dry_run and removed > 0:
            # Back up original
            backup = path.with_suffix(".txt.bak")
            if not backup.exists():
                shutil.copy2(path, backup)
                print(f"    ↳ backup: {backup}")
            with open(path, "w") as f:
                f.write("\n".join(keep) + "\n")

    return total_removed


def clean_features(other_ids: set, dry_run: bool) -> int:
    """Delete .pt feature files for Other-type patients."""
    total_deleted = 0
    # Iterate over all feature model directories (e.g. MahmoodLab_UNI, owkin_phikon)
    if not FEATURES_ROOT.exists():
        print(f"  [SKIP] Features root does not exist: {FEATURES_ROOT}")
        return 0

    for model_dir in sorted(FEATURES_ROOT.iterdir()):
        if not model_dir.is_dir():
            continue
        for split in SPLITS:
            split_dir = model_dir / split
            if not split_dir.exists():
                continue
            for pid in other_ids:
                pt_file = split_dir / f"{pid}.pt"
                if pt_file.exists():
                    total_deleted += 1
                    print(f"  [DEL] {pt_file.relative_to(DATASET_ROOT)}")
                    if not dry_run:
                        pt_file.unlink()

    return total_deleted


def clean_patches(other_ids: set, dry_run: bool) -> int:
    """Delete patch directories for Other-type patients."""
    total_deleted = 0
    if not PATCHES_ROOT.exists():
        print(f"  [SKIP] Patches root does not exist: {PATCHES_ROOT}")
        return 0

    for split in SPLITS:
        split_dir = PATCHES_ROOT / split
        if not split_dir.exists():
            continue
        for pid in other_ids:
            patch_dir = split_dir / pid
            if patch_dir.exists():
                n_patches = len(list(patch_dir.glob("*.jpg")))
                total_deleted += 1
                print(f"  [DEL] {patch_dir.relative_to(DATASET_ROOT)}  ({n_patches} patches)")
                if not dry_run:
                    shutil.rmtree(patch_dir)

    return total_deleted


def main():
    parser = argparse.ArgumentParser(
        description="Drop 'Other type' carcinoma patients from the BCNB pipeline."
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually perform deletions. Without this flag, runs in dry-run mode."
    )
    args = parser.parse_args()
    dry_run = not args.execute

    other_ids = get_other_ids()
    print(f"\n{'='*60}")
    print(f"  Drop '{DROP_VALUE}' carcinoma patients")
    print(f"  Found {len(other_ids)} patients to remove")
    print(f"  Mode: {'DRY RUN (preview only)' if dry_run else 'EXECUTE (deleting!)'}")
    print(f"{'='*60}\n")

    print("─── 1. Split ID files ───")
    n_ids = clean_split_files(other_ids, dry_run)

    print("\n─── 2. Feature .pt files ───")
    n_feats = clean_features(other_ids, dry_run)

    print("\n─── 3. Patch directories ───")
    n_patches = clean_patches(other_ids, dry_run)

    print(f"\n{'='*60}")
    print(f"  Summary:")
    print(f"    IDs removed from splits : {n_ids}")
    print(f"    Feature files deleted   : {n_feats}")
    print(f"    Patch dirs deleted      : {n_patches}")
    if dry_run:
        print(f"\n  ⚠  This was a DRY RUN. Nothing was changed.")
        print(f"     Re-run with --execute to apply changes.")
    else:
        print(f"\n  ✓  All changes applied. Backups of split files saved as .bak")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
