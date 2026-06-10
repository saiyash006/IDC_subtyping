# -*- coding: utf-8 -*-
"""
01_preprocessing.py
===================
BCNB WSI preprocessing pipeline — faithful rewrite of BALNMP's
cut_tumor_regions.py + cut_patches.py with three practical improvements:

  1. Large-WSI loading via TurboJPEG (bypasses OpenCV's 178 MP pixel cap)
     Falls back to PIL when TurboJPEG is unavailable.
  2. Boundary-safe crop + mask (fixes IndexError on slides whose annotation
     bounding boxes extend past the WSI edge)
  3. Parallel patient processing via multiprocessing.Pool

Core logic is identical to the originals:
  - Polygon fill:   cv2.fillPoly with (1, N, 2) int32 points, same as original
  - Blank filter:   np.sum(patch == (255,255,255)) / 3 / (H*W), exact formula
  - Patch naming:   {wsi_id}_{region_idx}_{x}_{y}.jpg, x before y
  - JPEG quality:   100 throughout

Usage:
    python 01_preprocessing.py                         # all 3 splits, both stages
    python 01_preprocessing.py --splits train          # train only
    python 01_preprocessing.py --splits train val      # two splits
    python 01_preprocessing.py --stage A               # Stage A only
    python 01_preprocessing.py --stage B               # Stage B only (regions must exist)
    python 01_preprocessing.py --overwrite             # remove existing output dirs first
    python 01_preprocessing.py --workers 8             # parallelism (default: cpu_count)

Dependencies:
    pip install opencv-python numpy tqdm
    pip install "PyTurboJPEG<2.0"   # optional but strongly recommended
    apt install libturbojpeg         # native lib for TurboJPEG
"""

import argparse
import gc
import json
import logging
import multiprocessing as mp
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── PIL: disable decompression-bomb guard before cv2 import ───────────────────
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
#  PATHS  — edit only DATASET_ROOT
# ─────────────────────────────────────────────────────────────────────────────
DATASET_ROOT = Path("/home/cgntlab5/bcnb/BCNB dataset")
WSI_DIR      = DATASET_ROOT / "WSIs"
SPLIT_DIR    = DATASET_ROOT / "dataset-splitting"
OUT_ROOT     = DATASET_ROOT / "outputs"

TUMOR_ROOT   = OUT_ROOT / "tumor_regions"
PATCHES_ROOT = OUT_ROOT / "patches"
LOG_DIR      = OUT_ROOT / "logs"

SPLIT_FILES: Dict[str, Path] = {
    "train": SPLIT_DIR / "train_id.txt",
    "val":   SPLIT_DIR / "val_id.txt",
    "test":  SPLIT_DIR / "test_id.txt",
}

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS  (match original BALNMP defaults)
# ─────────────────────────────────────────────────────────────────────────────
FILL_COLOR        = (255, 255, 255)   # BGR — white fill for non-tumour pixels
JPEG_QUALITY      = 100
DEFAULT_PATCH_SIZE  = 256
DEFAULT_BLANK_RATIO = 0.3

# TurboJPEG: try to locate the native library automatically, then fall back
_TURBOJPEG_LIB: Optional[str] = None
for _candidate in [
    "/usr/lib/x86_64-linux-gnu/libturbojpeg.so.0",
    "/usr/lib/aarch64-linux-gnu/libturbojpeg.so.0",
    "/usr/local/lib/libturbojpeg.so",
]:
    if os.path.exists(_candidate):
        _TURBOJPEG_LIB = _candidate
        break


# ─────────────────────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────────────────────
def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("preprocess")
    if logger.handlers:        # avoid duplicate handlers on re-import
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
#  SPLIT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def load_split_ids(path: Path) -> List[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def verify_no_leakage(split_ids: Dict[str, List[str]],
                      logger: logging.Logger) -> None:
    seen: Dict[str, str] = {}
    leaks: List[str] = []
    for split, ids in split_ids.items():
        for pid in ids:
            if pid in seen:
                leaks.append(f"Patient {pid} in '{seen[pid]}' AND '{split}'")
            else:
                seen[pid] = split
    if leaks:
        for msg in leaks:
            logger.error(f"DATA LEAKAGE: {msg}")
        raise ValueError(f"{len(leaks)} leakage violation(s) — aborting.")
    logger.info("Leakage check passed: all splits are disjoint.")


# ─────────────────────────────────────────────────────────────────────────────
#  WSI LOADING
#  Strategy: TurboJPEG (fastest) → PIL (safe fallback for any JPEG size)
#  Both return a BGR uint8 numpy array identical to what cv2.imread returns.
# ─────────────────────────────────────────────────────────────────────────────
def _imread_turbojpeg(path: Path, jpeg_reader) -> Optional[np.ndarray]:
    """Decode a JPEG with libjpeg-turbo. Returns BGR numpy array or None."""
    try:
        with open(path, "rb") as f:
            return jpeg_reader.decode(f.read())   # already BGR
    except Exception:
        return None


def _imread_pil(path: Path) -> Optional[np.ndarray]:
    """Decode a JPEG with Pillow. Converts RGB → BGR. Returns array or None."""
    try:
        rgb = np.asarray(Image.open(str(path)).convert("RGB"), dtype=np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def imread_wsi(path: Path, jpeg_reader=None) -> Optional[np.ndarray]:
    """
    Load a WSI JPEG robustly, bypassing OpenCV's 178 MP pixel-count validator.

    OpenCV 4.11+ compiled with HAVE_IMGCODEC_HDR enforces a hard cap in
    loadsave.cpp that the OPENCV_IO_MAX_IMAGE_PIXELS env-var no longer overrides.
    TurboJPEG and PIL are both immune to this cap.
    """
    if jpeg_reader is not None:
        img = _imread_turbojpeg(path, jpeg_reader)
        if img is not None:
            return img
    # fallback: PIL
    return _imread_pil(path)


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE A HELPERS  (faithful port of cut_tumor_regions.py)
# ─────────────────────────────────────────────────────────────────────────────
def get_annotation_points_and_bboxes(
    json_path: Path,
) -> Tuple[List[np.ndarray], List[List[float]]]:
    """
    Parse BCNB annotation JSON.

    Returns:
        annotation_points : list of (N_i, 2) float64 arrays (one per region)
        bboxes            : list of [start_x, start_y, width, height] floats
    """
    with open(json_path) as f:
        asap_json = json.load(f)
    annotation_points: List[np.ndarray] = []
    bboxes: List[List[float]] = []
    for region in asap_json.get("positive", []):
        pts = np.asarray(region["vertices"])   # (N, 2) float64
        annotation_points.append(pts)
        bboxes.append(_get_bbox(pts))
    return annotation_points, bboxes


def _get_bbox(points: np.ndarray) -> List[float]:
    """Axis-aligned bounding box → [start_x, start_y, width, height]."""
    max_xy = points.max(axis=0)
    min_xy = points.min(axis=0)
    return (min_xy.tolist() + (max_xy - min_xy).tolist())


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE A WORKER  — one patient, called inside a Pool worker or main thread
# ─────────────────────────────────────────────────────────────────────────────
def _stage_a_one_patient(
    wsi_id: str,
    wsi_dir: Path,
    out_dir: Path,
    fill_non_tumor: bool,
    jpeg_reader,           # TurboJPEG instance or None
) -> Tuple[str, int, List[str]]:
    """
    Extract annotated tumor regions for one patient.

    Returns (wsi_id, n_saved, log_lines).
    log_lines are collected in the worker and logged by the main process,
    avoiding logger pickling issues across multiprocessing workers.

    BUG FIX vs original cut_tumor_regions.py:
        Original builds mask_array with raw bbox (width, height) then applies it
        to tumour_img which may be smaller when the bbox overflows the WSI edge.
        Fix: clamp crop coordinates to WSI dimensions first, then build mask
        from the *actual* crop shape (crop_h, crop_w).
    """
    logs: List[str] = []
    wsi_path  = wsi_dir / f"{wsi_id}.jpg"
    json_path = wsi_dir / f"{wsi_id}.json"

    # ── validate inputs ───────────────────────────────────────────────────────
    for p, label in [(wsi_path, "WSI"), (json_path, "JSON")]:
        if not p.exists():
            logs.append(f"WARNING  [{wsi_id}] {label} not found: {p} — skipped")
            return wsi_id, 0, logs

    annotation_points, bboxes = get_annotation_points_and_bboxes(json_path)
    if not annotation_points:
        logs.append(f"WARNING  [{wsi_id}] No positive annotations — skipped")
        return wsi_id, 0, logs

    # ── load WSI ──────────────────────────────────────────────────────────────
    wsi_img = imread_wsi(wsi_path, jpeg_reader)
    if wsi_img is None:
        logs.append(f"ERROR    [{wsi_id}] Failed to load WSI — skipped")
        return wsi_id, 0, logs

    wsi_h, wsi_w = wsi_img.shape[:2]
    patient_dir = out_dir / wsi_id
    patient_dir.mkdir(parents=True, exist_ok=True)
    logs.append(f"DEBUG    [{wsi_id}] WSI {wsi_w}×{wsi_h} | "
                f"{len(annotation_points)} region(s)")

    saved = 0
    for i, (ann_points, bbox) in enumerate(zip(annotation_points, bboxes)):
        x, y, width, height = [int(v) for v in bbox]

        # ── FIX: clamp crop to WSI boundary before slicing ────────────────────
        # Original slices wsi_img[y:y+height, x:x+width] with no clamping.
        # When the bbox overflows (x+width > wsi_w or y+height > wsi_h), the
        # resulting crop is smaller than (height, width), so the mask built
        # with np.zeros((height, width)) does not match the crop — IndexError.
        x1 = max(0, x);        y1 = max(0, y)
        x2 = min(wsi_w, x + width)
        y2 = min(wsi_h, y + height)

        if x2 <= x1 or y2 <= y1:
            logs.append(f"DEBUG    [{wsi_id}] Region {i}: bbox OOB — skipped")
            continue

        tumour_img = wsi_img[y1:y2, x1:x2].copy()
        crop_h, crop_w = tumour_img.shape[:2]   # actual size after clamping

        # ── fill non-tumour areas (same logic as original) ────────────────────
        if fill_non_tumor:
            # Build mask at ACTUAL crop size, not raw bbox size
            mask_array = np.zeros((crop_h, crop_w), dtype=np.uint8)

            # Translate polygon to crop-local coords (same formula as original:
            #   ann_points - np.asarray([x, y])
            # but using clamped origin [x1, y1] so coords stay within crop)
            rel = (ann_points - np.asarray([x1, y1])).astype(np.int32)
            rel[:, 0] = np.clip(rel[:, 0], 0, crop_w - 1)
            rel[:, 1] = np.clip(rel[:, 1], 0, crop_h - 1)

            # expand_dims to (1, N, 2) — identical to original
            cv2.fillPoly(mask_array, np.expand_dims(rel, 0), color=255)
            tumour_img[mask_array != 255] = FILL_COLOR

        save_path = patient_dir / f"{wsi_id}_{i}.jpg"
        cv2.imwrite(str(save_path), tumour_img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        logs.append(f"DEBUG    [{wsi_id}] Saved region {i} → {save_path.name} "
                    f"({crop_w}×{crop_h})")
        saved += 1

    del wsi_img
    gc.collect()
    return wsi_id, saved, logs


# ── Pool initializer: create one TurboJPEG instance per worker ───────────────
_worker_jpeg = None

def _stage_a_worker_init(lib_path: Optional[str]) -> None:
    global _worker_jpeg
    if lib_path:
        try:
            from turbojpeg import TurboJPEG
            _worker_jpeg = TurboJPEG(lib_path)
        except Exception:
            _worker_jpeg = None
    else:
        _worker_jpeg = None


def _stage_a_worker(args: tuple) -> Tuple[str, int, List[str]]:
    wsi_id, wsi_dir_str, out_dir_str, fill = args
    return _stage_a_one_patient(
        wsi_id,
        Path(wsi_dir_str),
        Path(out_dir_str),
        fill,
        _worker_jpeg,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE A  — run all patients for one split
# ─────────────────────────────────────────────────────────────────────────────
def run_stage_a(
    split: str,
    patient_ids: List[str],
    wsi_dir: Path,
    tumor_root: Path,
    logger: logging.Logger,
    fill_non_tumor: bool = True,
    n_workers: int = 1,
    resume: bool = False,
) -> Dict[str, int]:
    out_dir = tumor_root / split
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[Stage A | {split}] {len(patient_ids)} patients → {out_dir} "
                f"| workers={n_workers} | resume={resume}")

    stats: Dict[str, int] = {}
    jobs = []
    for pid in patient_ids:
        pid_dir = out_dir / pid
        if resume and pid_dir.exists() and any(pid_dir.iterdir()):
            stats[pid] = len(list(pid_dir.glob("*.jpg")))
        else:
            jobs.append((pid, str(wsi_dir), str(out_dir), fill_non_tumor))

    if resume:
        logger.info(f"[{split}] Resuming: skipping {len(patient_ids) - len(jobs)} already processed patients.")

    if n_workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=n_workers,
            initializer=_stage_a_worker_init,
            initargs=(_TURBOJPEG_LIB,),
        ) as pool:
            for wsi_id, n_saved, logs in tqdm(
                pool.imap_unordered(_stage_a_worker, jobs),
                total=len(jobs),
                desc=f"Stage A [{split}]",
                unit="patient",
            ):
                _flush_worker_logs(logs, logger)
                stats[wsi_id] = n_saved
    else:
        # Single-process: create TurboJPEG once in the main process
        jpeg_reader = _make_jpeg_reader()
        for pid in tqdm(patient_ids, desc=f"Stage A [{split}]", unit="patient"):
            wsi_id, n_saved, logs = _stage_a_one_patient(
                pid, wsi_dir, out_dir, fill_non_tumor, jpeg_reader
            )
            _flush_worker_logs(logs, logger)
            stats[wsi_id] = n_saved

    total   = sum(stats.values())
    skipped = sum(1 for v in stats.values() if v == 0)
    logger.info(f"[Stage A | {split}] regions saved={total}  "
                f"skipped={skipped}/{len(patient_ids)}")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE B HELPERS  (faithful port of cut_patches.py)
# ─────────────────────────────────────────────────────────────────────────────
def _padding(image_array: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Pad image on right and bottom with white so H and W are multiples of
    patch_size.  Identical to original padding() in cut_patches.py.
    """
    height, width = image_array.shape[:2]
    right  = int(np.ceil(width  / patch_size) * patch_size - width)
    bottom = int(np.ceil(height / patch_size) * patch_size - height)
    if right == 0 and bottom == 0:
        return image_array
    return cv2.copyMakeBorder(image_array, 0, bottom, 0, right,
                              cv2.BORDER_CONSTANT, value=(255, 255, 255))


def _blank_ratio(patch: np.ndarray, h: int, w: int) -> float:
    """
    Fraction of pure-white pixels in a patch.

    Replicates the EXACT formula from cut_patches.py:
        blank_num = np.sum(image_array == (255, 255, 255)) / 3
    which counts channel-wise 255-hits then divides by 3.
    For patches filled with FILL_COLOR=(255,255,255) this is identical
    to a per-pixel all-channels check; the difference appears only for
    pixels that are 255 in some but not all channels (rare in H&E).
    """
    return (np.sum(patch == (255, 255, 255)) / 3) / (h * w)


def _tile_region(
    big_image: np.ndarray,
    patch_size: int,
) -> Tuple[np.ndarray, int, int]:
    """
    Return a zero-copy stride view of shape (rows, cols, SZ, SZ, 3).

    Using numpy stride tricks avoids materialising all patches in memory.
    Each tiles[r, c] is the (SZ, SZ, 3) patch at grid position (r, c);
    x = c * patch_size, y = r * patch_size  (matching original x/y naming).
    """
    H, W  = big_image.shape[:2]
    SZ    = patch_size
    rows  = H // SZ
    cols  = W // SZ
    s     = big_image.strides                      # (row_stride, col_stride, channel_stride)
    tiles = as_strided(
        big_image,
        shape=(rows, cols, SZ, SZ, 3),
        strides=(s[0] * SZ, s[1] * SZ, s[0], s[1], s[2]),
    )
    return tiles, rows, cols


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE B WORKER  — one patient, returns stats + log lines
# ─────────────────────────────────────────────────────────────────────────────
def _stage_b_one_patient(
    wsi_id: str,
    tumor_split_dir: Path,
    patch_split_dir: Path,
    patch_size: int,
    max_blank_ratio: float,
) -> Tuple[str, Dict, List[str]]:
    """
    Tile all tumor region images for one patient into patches.

    Speed improvements vs original cut_patches.py:
      - Stride-based zero-copy tiling (no per-patch array allocation)
      - Blank ratio computed over all patches of a region in one numpy call
        using the exact same formula as the original
      - imwrite called only for kept patches
    """
    logs: List[str] = []
    region_dir    = tumor_split_dir / wsi_id
    patient_out   = patch_split_dir / wsi_id

    if not region_dir.exists():
        logs.append(f"WARNING  [{wsi_id}] No tumor_regions dir — run Stage A first")
        return wsi_id, {"saved": 0, "discarded": 0, "regions": 0}, logs

    region_files = sorted(region_dir.glob("*.jpg"))
    if not region_files:
        logs.append(f"WARNING  [{wsi_id}] No region images in {region_dir}")
        return wsi_id, {"saved": 0, "discarded": 0, "regions": 0}, logs

    patient_out.mkdir(parents=True, exist_ok=True)
    tot_saved = tot_disc = 0

    for region_path in region_files:
        # original_image_name = stem of region file, e.g. "273_2"
        # patch name = "{original_image_name}_{x}_{y}.jpg"  (x before y, matching original)
        original_image_name = region_path.stem      # "{wsi_id}_{region_idx}"

        big_image_array = cv2.imread(str(region_path))
        if big_image_array is None:
            logs.append(f"ERROR    [{wsi_id}] cv2.imread failed: {region_path.name}")
            continue

        # pad — identical to original
        big_image_array = _padding(big_image_array, patch_size)
        height, width   = big_image_array.shape[:2]

        # ── stride tiling: zero-copy view (rows, cols, SZ, SZ, 3) ─────────
        tiles, rows, cols = _tile_region(big_image_array, patch_size)

        # ── vectorised blank detection over all patches in this region ─────
        # Flatten to (N, SZ, SZ, 3) — this is a VIEW, not a copy
        tiles_flat = tiles.reshape(-1, patch_size, patch_size, 3)
        # Exact original formula applied to the whole batch at once:
        #   blank_num = np.sum(patch == (255,255,255)) / 3 / (H*W)
        blank_ratios = (
            (tiles_flat == 255).sum(axis=(1, 2, 3)) / 3
        ) / (patch_size * patch_size)                  # shape: (rows*cols,)

        # ── write kept patches ─────────────────────────────────────────────
        for idx, ratio in enumerate(blank_ratios):
            r, c = divmod(idx, cols)
            x    = c * patch_size     # pixel x-offset (column → x, matching original)
            y    = r * patch_size     # pixel y-offset (row → y)
            patch_name = f"{original_image_name}_{x}_{y}.jpg"
            patch_path = str(patient_out / patch_name)

            if ratio >= max_blank_ratio:
                tot_disc += 1
                logs.append(f"DEBUG    [{wsi_id}] deprecated (blank={ratio:.2f}): "
                             f"{patch_name}")
            else:
                # .copy() needed: strides view is not contiguous — imwrite requires it
                cv2.imwrite(
                    patch_path,
                    tiles[r, c].copy(),
                    [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
                )
                tot_saved += 1
                logs.append(f"DEBUG    [{wsi_id}] saved {patch_name}")

        del big_image_array
        gc.collect()

    stats = {"saved": tot_saved, "discarded": tot_disc, "regions": len(region_files)}
    logs.append(f"DEBUG    [{wsi_id}] {len(region_files)} region(s) → "
                f"{tot_saved} saved, {tot_disc} discarded")
    return wsi_id, stats, logs


def _stage_b_worker(args: tuple):
    wsi_id, tumor_dir_str, patch_dir_str, patch_size, blank_ratio = args
    return _stage_b_one_patient(
        wsi_id,
        Path(tumor_dir_str),
        Path(patch_dir_str),
        patch_size,
        blank_ratio,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE B  — run all patients for one split
# ─────────────────────────────────────────────────────────────────────────────
def run_stage_b(
    split: str,
    patient_ids: List[str],
    tumor_root: Path,
    patches_root: Path,
    patch_size: int,
    max_blank_ratio: float,
    logger: logging.Logger,
    n_workers: int = 1,
    resume: bool = False,
) -> Dict[str, Dict]:
    tumor_split = tumor_root   / split
    patch_split = patches_root / split
    patch_split.mkdir(parents=True, exist_ok=True)
    logger.info(f"[Stage B | {split}] {patch_size}×{patch_size} patches  "
                f"blank<{max_blank_ratio}  workers={n_workers}  → {patch_split}")

    stats: Dict[str, Dict] = {}
    jobs = []
    for pid in patient_ids:
        pid_dir = patch_split / pid
        if resume and pid_dir.exists() and any(pid_dir.iterdir()):
            n_saved = len(list(pid_dir.glob("*.jpg")))
            stats[pid] = {"saved": n_saved, "discarded": 0, "regions": -1}
        else:
            jobs.append((pid, str(tumor_split), str(patch_split), patch_size, max_blank_ratio))

    if resume:
        logger.info(f"[{split}] Resuming Stage B: skipping {len(patient_ids) - len(jobs)} already processed patients.")

    if n_workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            for wsi_id, pstats, logs in tqdm(
                pool.imap_unordered(_stage_b_worker, jobs),
                total=len(jobs),
                desc=f"Stage B [{split}]",
                unit="patient",
            ):
                _flush_worker_logs(logs, logger)
                stats[wsi_id] = pstats
    else:
        for pid in tqdm(patient_ids, desc=f"Stage B [{split}]", unit="patient"):
            wsi_id, pstats, logs = _stage_b_one_patient(
                pid, tumor_split, patch_split, patch_size, max_blank_ratio
            )
            _flush_worker_logs(logs, logger)
            stats[wsi_id] = pstats

    n_saved   = sum(v["saved"]     for v in stats.values())
    n_disc    = sum(v["discarded"] for v in stats.values())
    n_empty   = sum(1 for v in stats.values() if v["saved"] == 0)
    logger.info(f"[Stage B | {split}] saved={n_saved}  discarded={n_disc}  "
                f"empty_patients={n_empty}/{len(patient_ids)}")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def _make_jpeg_reader():
    """Create a TurboJPEG instance if available, else None."""
    if _TURBOJPEG_LIB:
        try:
            from turbojpeg import TurboJPEG
            return TurboJPEG(_TURBOJPEG_LIB)
        except Exception:
            pass
    return None


def _flush_worker_logs(logs: List[str], logger: logging.Logger) -> None:
    """Replay log lines collected inside a worker into the main logger."""
    for line in logs:
        level, _, msg = line.partition("  ")
        lvl = getattr(logging, level.strip(), logging.DEBUG)
        logger.log(lvl, msg)


def _print_summary(
    split_ids:     Dict[str, List[str]],
    stage_a_stats: Dict[str, Dict],
    stage_b_stats: Dict[str, Dict],
    logger:        logging.Logger,
) -> None:
    logger.info("")
    logger.info("=" * 68)
    logger.info("  PREPROCESSING COMPLETE")
    logger.info("=" * 68)
    hdr = (f"  {'split':<6} | {'patients':>8} | {'regions':>8} | "
           f"{'patches':>10} | {'discarded':>10}")
    logger.info(hdr)
    logger.info("  " + "─" * 60)
    for split, ids in split_ids.items():
        a = stage_a_stats.get(split, {})
        b = stage_b_stats.get(split, {})
        n_regions = str(sum(a.values())) if a else "—"
        n_patches = str(sum(v["saved"]     for v in b.values())) if b else "—"
        n_disc    = str(sum(v["discarded"] for v in b.values())) if b else "—"
        logger.info(f"  {split:<6} | {len(ids):>8} | {n_regions:>8} | "
                    f"{n_patches:>10} | {n_disc:>10}")
    logger.info("=" * 68)
    logger.info(f"  Tumor regions : {TUMOR_ROOT}")
    logger.info(f"  Patches       : {PATCHES_ROOT}")
    logger.info(f"  Log file      : {LOG_DIR / '01_preprocessing.log'}")
    logger.info("  Patch naming  : {wsi_id}_{region_idx}_{x}_{y}.jpg")
    logger.info("=" * 68)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BCNB WSI Preprocessing | Stage A: tumor regions | Stage B: patches",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--splits", nargs="+", choices=["train", "val", "test"],
                   default=["train", "val", "test"])
    p.add_argument("--stage",  choices=["A", "B", "AB"], default="AB")
    p.add_argument("--patch_size",  type=int,   default=DEFAULT_PATCH_SIZE)
    p.add_argument("--blank_ratio", type=float, default=DEFAULT_BLANK_RATIO)
    p.add_argument("--no_fill",  action="store_true",
                   help="Skip white-filling non-tumour pixels in Stage A.")
    p.add_argument("--workers", type=int,
                   default=max(1, mp.cpu_count() // 2),
                   help="Parallel worker processes (default: cpu_count/2).")
    p.add_argument("--overwrite", action="store_true",
                   help="Remove and recreate existing output directories.")
    p.add_argument("--resume", action="store_true",
                   help="Skip already processed patients to resume after interruption.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = _parse_args()

    log_path = LOG_DIR / "01_preprocessing.log"
    logger   = setup_logging(log_path)

    logger.info("=" * 68)
    logger.info("  BCNB Preprocessing Pipeline")
    logger.info("=" * 68)
    logger.info(f"  Dataset root  : {DATASET_ROOT}")
    logger.info(f"  Splits        : {args.splits}")
    logger.info(f"  Stage(s)      : {args.stage}")
    logger.info(f"  Patch size    : {args.patch_size}×{args.patch_size}")
    logger.info(f"  Blank ratio   : {args.blank_ratio}")
    logger.info(f"  Fill non-tumor: {not args.no_fill}")
    logger.info(f"  Workers       : {args.workers}")
    turbo_status = f"TurboJPEG ({_TURBOJPEG_LIB})" if _TURBOJPEG_LIB else "PIL (TurboJPEG not found)"
    logger.info(f"  WSI loader    : {turbo_status}")

    for path, label in [(WSI_DIR, "WSI dir"), (SPLIT_DIR, "split dir")]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    # ── load & validate splits ─────────────────────────────────────────────
    all_split_ids: Dict[str, List[str]] = {}
    for split in ["train", "val", "test"]:
        all_split_ids[split] = load_split_ids(SPLIT_FILES[split])
        logger.info(f"  Loaded {len(all_split_ids[split]):>4} IDs  [{split}]")
    verify_no_leakage(all_split_ids, logger)

    selected = {s: all_split_ids[s] for s in args.splits}

    # ── handle existing outputs ────────────────────────────────────────────
    for split in args.splits:
        dirs = []
        if "A" in args.stage: dirs.append(TUMOR_ROOT   / split)
        if "B" in args.stage: dirs.append(PATCHES_ROOT / split)
        for d in dirs:
            if d.exists():
                if args.resume:
                    logger.info(f"--resume: keeping existing output directory {d}")
                elif not args.overwrite:
                    raise FileExistsError(
                        f"Output dir exists: {d}\n"
                        f"Pass --overwrite to replace it, or --resume to continue."
                    )
                else:
                    logger.warning(f"--overwrite: removing {d}")
                    shutil.rmtree(d)

    stage_a_stats: Dict[str, Dict] = {}
    stage_b_stats: Dict[str, Dict] = {}

    for split in args.splits:
        pids = selected[split]
        logger.info(f"\n{'─'*55}")
        logger.info(f"  Split: {split.upper()}  ({len(pids)} patients)")
        logger.info(f"{'─'*55}")

        if "A" in args.stage:
            stage_a_stats[split] = run_stage_a(
                split, pids, WSI_DIR, TUMOR_ROOT, logger,
                fill_non_tumor=not args.no_fill,
                n_workers=args.workers,
                resume=args.resume,
            )

        if "B" in args.stage:
            stage_b_stats[split] = run_stage_b(
                split, pids, TUMOR_ROOT, PATCHES_ROOT,
                args.patch_size, args.blank_ratio, logger,
                n_workers=args.workers,
                resume=args.resume,
            )

    _print_summary(selected, stage_a_stats, stage_b_stats, logger)


if __name__ == "__main__":
    main()