# -*- coding: utf-8 -*-
"""
04_evaluate_heatmaps.py
=======================
Phase 4 — Evaluation Metrics + Spatial Attention Heatmaps

Generates:
  1. Per-task ROC curves (single figure)
  2. Per-task confusion matrices
  3. Spatial attention heatmaps overlaid on WSI thumbnails for selected test patients

Usage:
    python 04_evaluate_heatmaps.py --model owkin_phikon
    python 04_evaluate_heatmaps.py --model owkin_phikon --n_heatmaps 20
    python 04_evaluate_heatmaps.py --model owkin_phikon --patients 984 737 793
"""

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # WSIs can exceed default limit
from sklearn.metrics import (auc, confusion_matrix, roc_auc_score, roc_curve,
                              ConfusionMatrixDisplay)

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_ROOT  = Path("/home/cgntlab5/bcnb/BCNB dataset")
FEATURES_ROOT = DATASET_ROOT / "outputs" / "features"
PATCHES_ROOT  = DATASET_ROOT / "outputs" / "patches"
CKPT_DIR      = DATASET_ROOT / "outputs" / "checkpoints"
RESULTS_DIR   = DATASET_ROOT / "outputs" / "results"
LOG_DIR       = DATASET_ROOT / "outputs" / "logs"
CLINICAL_XLSX = DATASET_ROOT / "patient-clinical-data.xlsx"
SPLIT_DIR     = DATASET_ROOT / "dataset-splitting"
WSI_DIR       = DATASET_ROOT / "WSIs"

BINARY_TASKS = ["ER", "PR", "HER2"]
GRADE_TASK   = "Grade"
ALL_TASKS    = BINARY_TASKS + [GRADE_TASK]
N_GRADE_CLS  = 3
PATCH_SIZE   = 256


# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("evaluate")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w"); fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
    ch = logging.StreamHandler(); ch.setLevel(logging.INFO); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger


# ── Model (must match 03_train_mil.py) ────────────────────────────────────────
class GatedABMIL(nn.Module):
    def __init__(self, embed_dim=768, attn_dim=256, attn_dropout=0.25, clf_dropout=0.0, use_layernorm=False):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(embed_dim, attn_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(embed_dim, attn_dim), nn.Sigmoid())
        self.attention_w = nn.Linear(attn_dim, 1, bias=False)
        self.attn_drop   = nn.Dropout(attn_dropout)
        
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.norm        = nn.LayerNorm(embed_dim)
            
        self.clf_dropout = clf_dropout
        if clf_dropout > 0.0:
            self.clf_drop    = nn.Dropout(clf_dropout)
            
        self.heads = nn.ModuleDict({
            "ER": nn.Linear(embed_dim, 1), "PR": nn.Linear(embed_dim, 1),
            "HER2": nn.Linear(embed_dim, 1), "Grade": nn.Linear(embed_dim, N_GRADE_CLS),
        })

    def forward(self, H):
        H = self.attn_drop(H)
        A = torch.softmax(self.attention_w(self.attention_V(H) * self.attention_U(H)), dim=0)
        z = (A * H).sum(dim=0, keepdim=True)
        if self.use_layernorm:
            z = self.norm(z)
        if self.clf_dropout > 0.0:
            z = self.clf_drop(z)
        return {t: head(z).squeeze(0) for t, head in self.heads.items()}, A


# ── Label loading ─────────────────────────────────────────────────────────────
def load_labels(excel_path: Path) -> pd.DataFrame:
    df = pd.read_excel(str(excel_path))
    df.columns = df.columns.str.strip()
    id_col = next((c for c in df.columns if c.lower().replace(" ", "_") == "patient_id"), None)
    if id_col is None:
        raise KeyError(f"Patient ID column not found. Available: {list(df.columns)}")
    df = df.set_index(id_col); df.index = df.index.astype(str)
    grade_col = next((c for c in df.columns if "grad" in c.lower()), None)
    if grade_col is None:
        raise KeyError("Grade column not found")
    df = df[BINARY_TASKS + [grade_col]].copy()
    df = df.rename(columns={grade_col: "Histological Grade"})
    for col in BINARY_TASKS:
        mapping = {"positive": 1, "negative": 0, "1": 1, "0": 0}
        df[col] = df[col].astype(str).str.lower().str.strip().map(mapping)
    df = df.dropna(subset=BINARY_TASKS + ["Histological Grade"])
    df["Grade_idx"] = df["Histological Grade"].astype(int) - 1
    df[BINARY_TASKS] = df[BINARY_TASKS].astype(int)
    return df


def load_split_ids(path: Path) -> List[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


# ── Inference on test set ─────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(
    model: GatedABMIL, feat_dir: Path, patient_ids: List[str],
    labels_df: pd.DataFrame, device: torch.device,
) -> Tuple[Dict, Dict, Dict, List[str]]:
    """Returns per-task logits, labels, attention weights, and valid IDs."""
    model.eval()
    all_logits = {t: [] for t in ALL_TASKS}
    all_labels = {k: [] for k in ALL_TASKS + ["Grade_idx"]}
    all_attns  = {}
    valid_ids  = []

    for pid in patient_ids:
        pt = feat_dir / f"{pid}.pt"
        if not pt.exists() or pid not in labels_df.index:
            continue
        feat = torch.load(str(pt), map_location="cpu", weights_only=True).float().to(device)
        preds, A = model(feat)
        row = labels_df.loc[pid]

        for t in BINARY_TASKS:
            all_logits[t].append(preds[t].cpu())
            all_labels[t].append(int(row[t]))
        all_logits["Grade"].append(preds["Grade"].cpu())
        all_labels["Grade_idx"].append(int(row["Grade_idx"]))
        all_attns[pid] = A.squeeze(-1).cpu().numpy()  # [N]
        valid_ids.append(pid)

    return all_logits, all_labels, all_attns, valid_ids


# ── Plot: ROC curves ─────────────────────────────────────────────────────────
def plot_roc_curves(all_logits, all_labels, out_path, logger):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Per-Task ROC Curves (Test Set)", fontsize=16, fontweight="bold", y=1.02)
    colors = ["#2563eb", "#7c3aed", "#db2777", "#059669"]

    for i, task in enumerate(ALL_TASKS):
        ax = axes[i]
        if task in BINARY_TASKS:
            logits = torch.stack(all_logits[task]).squeeze(-1).numpy()
            labels = np.array(all_labels[task])
            probs  = 1.0 / (1.0 + np.exp(-logits))
            fpr, tpr, _ = roc_curve(labels, probs)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i], lw=2.5, label=f"AUC = {roc_auc:.3f}")
            ax.fill_between(fpr, tpr, alpha=0.15, color=colors[i])
        else:
            logits_g = torch.stack(all_logits["Grade"]).numpy()
            labels_g = np.array(all_labels["Grade_idx"])
            exp_g = np.exp(logits_g - logits_g.max(axis=1, keepdims=True))
            probs_g = exp_g / exp_g.sum(axis=1, keepdims=True)
            grade_names = ["Grade 1", "Grade 2", "Grade 3"]
            grade_colors = ["#10b981", "#f59e0b", "#ef4444"]
            for g in range(N_GRADE_CLS):
                binary_labels = (labels_g == g).astype(int)
                if binary_labels.sum() == 0:
                    continue
                fpr, tpr, _ = roc_curve(binary_labels, probs_g[:, g])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=grade_colors[g], lw=2,
                        label=f"{grade_names[g]} AUC={roc_auc:.3f}")
                ax.fill_between(fpr, tpr, alpha=0.08, color=grade_colors[g])

        ax.plot([0,1], [0,1], "k--", lw=1, alpha=0.4)
        ax.set_xlabel("FPR", fontsize=11); ax.set_ylabel("TPR", fontsize=11)
        ax.set_title(task, fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"ROC curves saved → {out_path}")


# ── Plot: Confusion matrices ─────────────────────────────────────────────────
def plot_confusion_matrices(all_logits, all_labels, out_path, logger):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Confusion Matrices (Test Set)", fontsize=16, fontweight="bold", y=1.02)

    for i, task in enumerate(ALL_TASKS):
        ax = axes[i]
        if task in BINARY_TASKS:
            logits = torch.stack(all_logits[task]).squeeze(-1).numpy()
            labels = np.array(all_labels[task])
            preds = (1.0 / (1.0 + np.exp(-logits)) >= 0.5).astype(int)
            cm = confusion_matrix(labels, preds, labels=[0, 1])
            disp = ConfusionMatrixDisplay(cm, display_labels=["Neg", "Pos"])
        else:
            logits_g = torch.stack(all_logits["Grade"]).numpy()
            labels_g = np.array(all_labels["Grade_idx"])
            preds_g = logits_g.argmax(axis=1)
            cm = confusion_matrix(labels_g, preds_g, labels=[0, 1, 2])
            disp = ConfusionMatrixDisplay(cm, display_labels=["G1", "G2", "G3"])

        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(task, fontsize=13, fontweight="bold")

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrices saved → {out_path}")


# ── Spatial heatmap generation ────────────────────────────────────────────────
def parse_patch_name(fname: str) -> Tuple[str, int, int, int]:
    """Parse '{pid}_{region}_{x}_{y}.jpg' → (pid, region_idx, x_offset, y_offset)."""
    stem = Path(fname).stem
    parts = stem.split("_")
    pid = parts[0]
    region_idx = int(parts[1])
    x_off = int(parts[2])
    y_off = int(parts[3])
    return pid, region_idx, x_off, y_off


def get_region_bboxes(json_path: Path) -> Dict[int, Tuple[int, int, int, int]]:
    """Load annotation JSON → {region_idx: (x_min, y_min, width, height)}."""
    with open(str(json_path)) as f:
        data = json.load(f)
    bboxes = {}
    for i, ann in enumerate(data.get("positive", [])):
        verts = ann["vertices"]
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        x_min, y_min = int(min(xs)), int(min(ys))
        bboxes[i] = (x_min, y_min, int(max(xs)) - x_min, int(max(ys)) - y_min)
    return bboxes


def generate_spatial_heatmap(
    patient_id: str, attn_weights: np.ndarray, patches_dir: Path,
    wsi_dir: Path, out_path: Path, labels_df: pd.DataFrame,
    preds_dict: Dict, logger: logging.Logger,
):
    """Overlay attention weights on a canvas showing patch positions."""
    patch_paths = sorted(patches_dir.glob("*.jpg"))
    if len(patch_paths) == 0:
        logger.warning(f"  [{patient_id}] No patches found — skipping")
        return False

    if len(attn_weights) != len(patch_paths):
        logger.warning(
            f"  [{patient_id}] attn ({len(attn_weights)}) != patches ({len(patch_paths)}) — skipping"
        )
        return False

    json_path = wsi_dir / f"{patient_id}.json"
    if not json_path.exists():
        logger.warning(f"  [{patient_id}] JSON missing — skipping")
        return False

    bboxes = get_region_bboxes(json_path)

    # Map each patch to WSI-level coordinates
    patch_coords = []
    for idx, pp in enumerate(patch_paths):
        _, region_idx, x_off, y_off = parse_patch_name(pp.name)
        if region_idx not in bboxes:
            continue
        rx, ry, _, _ = bboxes[region_idx]
        wsi_x = rx + x_off
        wsi_y = ry + y_off
        patch_coords.append((wsi_x, wsi_y, attn_weights[idx], str(pp)))

    if len(patch_coords) == 0:
        logger.warning(f"  [{patient_id}] No valid patch coords — skipping")
        return False

    # Determine canvas bounds from patch coordinates
    all_x = [c[0] for c in patch_coords]
    all_y = [c[1] for c in patch_coords]
    x_min, x_max = min(all_x), max(all_x) + PATCH_SIZE
    y_min, y_max = min(all_y), max(all_y) + PATCH_SIZE
    w_range = x_max - x_min
    h_range = y_max - y_min

    # Scale to fit in a reasonable canvas
    max_canvas = 1600
    scale = min(max_canvas / max(w_range, 1), max_canvas / max(h_range, 1), 1.0)
    canvas_w = max(int(w_range * scale) + 10, 100)
    canvas_h = max(int(h_range * scale) + 10, 100)
    patch_scaled = max(1, int(PATCH_SIZE * scale))

    # Build attention heatmap on canvas
    attn_canvas = np.full((canvas_h, canvas_w), np.nan, dtype=np.float32)
    # Also build a tissue canvas from actual patch images
    tissue_canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 240  # light grey bg

    attn_vals = np.array([c[2] for c in patch_coords])
    if attn_vals.max() > attn_vals.min():
        attn_norm = (attn_vals - attn_vals.min()) / (attn_vals.max() - attn_vals.min())
    else:
        attn_norm = np.zeros_like(attn_vals)

    for i, (wx, wy, aw, pp) in enumerate(patch_coords):
        sx = int((wx - x_min) * scale)
        sy = int((wy - y_min) * scale)
        ex = min(sx + patch_scaled, canvas_w)
        ey = min(sy + patch_scaled, canvas_h)
        if sx >= canvas_w or sy >= canvas_h:
            continue
        attn_canvas[sy:ey, sx:ex] = attn_norm[i]

        # Load patch thumbnail for tissue context
        try:
            pimg = Image.open(pp).resize((ex - sx, ey - sy), Image.LANCZOS)
            tissue_canvas[sy:ey, sx:ex] = np.array(pimg)[:ey-sy, :ex-sx, :3]
        except Exception:
            pass

    # Custom colormap
    cmap = LinearSegmentedColormap.from_list("attn", [
        (0.0,  (0.0, 0.0, 0.5, 0.0)),
        (0.3,  (0.0, 0.3, 0.8, 0.3)),
        (0.5,  (0.0, 0.7, 0.9, 0.45)),
        (0.7,  (1.0, 0.9, 0.0, 0.6)),
        (0.85, (1.0, 0.4, 0.0, 0.75)),
        (1.0,  (0.8, 0.0, 0.0, 0.9)),
    ])

    # Labels
    row = labels_df.loc[patient_id] if patient_id in labels_df.index else None
    gt_str = ""
    pred_str = ""
    if row is not None:
        gt_str = (f"ER={int(row['ER'])}  PR={int(row['PR'])}  "
                  f"HER2={int(row['HER2'])}  Grade={int(row['Histological Grade'])}")
    if preds_dict:
        pred_str = "  |  ".join(f"{k}={v}" for k, v in preds_dict.items())

    # Plot: tissue + attention overlay
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Left: tissue patches
    axes[0].imshow(tissue_canvas)
    axes[0].set_title(f"Patient {patient_id} — Tissue Patches", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    # Right: attention heatmap overlay
    axes[1].imshow(tissue_canvas)
    masked = np.ma.masked_where(np.isnan(attn_canvas), attn_canvas)
    im = axes[1].imshow(masked, cmap=cmap, interpolation="nearest", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.035, pad=0.02, shrink=0.8)
    cbar.set_label("Attention Weight (normalised)", fontsize=11)

    title = f"Patient {patient_id} — Attention Heatmap\n"
    if gt_str:
        title += f"GT: {gt_str}\n"
    if pred_str:
        title += f"Pred: {pred_str}"
    axes[1].set_title(title, fontsize=12, fontweight="bold", pad=10)
    axes[1].axis("off")

    # Draw region outlines on both panels
    for ridx, (rx, ry, rw, rh) in bboxes.items():
        for ax in axes:
            rect = mpatches.Rectangle(
                ((rx - x_min) * scale, (ry - y_min) * scale), rw * scale, rh * scale,
                linewidth=1.5, edgecolor="#00ff88", facecolor="none", linestyle="--", alpha=0.6
            )
            ax.add_patch(rect)

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phase 4: Evaluation + Spatial Heatmaps")
    parser.add_argument("--model", type=str, default="owkin_phikon")
    parser.add_argument("--n_heatmaps", type=int, default=10,
                        help="Number of test patients to generate heatmaps for")
    parser.add_argument("--patients", nargs="*", default=None,
                        help="Specific patient IDs for heatmaps (overrides n_heatmaps)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    logger = setup_logging(LOG_DIR / "04_evaluate_heatmaps.log")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model_dir   = FEATURES_ROOT / args.model
    ckpt_path   = CKPT_DIR / args.model / "best_model.pth"
    results_dir = RESULTS_DIR / args.model
    heatmap_dir = results_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  Phase 4: Evaluation + Spatial Attention Heatmaps")
    logger.info("=" * 60)
    logger.info(f"  Model     : {args.model}")
    logger.info(f"  Device    : {device}")
    logger.info(f"  Checkpoint: {ckpt_path}")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    params = ckpt["params"]
    embed_dim = ckpt["embed_dim"]
    logger.info(f"  Embed dim : {embed_dim}")
    logger.info(f"  Params    : {params}")

    model = GatedABMIL(embed_dim, params["attn_dim"], params["attn_dropout"], params.get("clf_dropout", 0.0), params.get("use_layernorm", False)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    logger.info("  Model loaded ✓")

    # ── Load labels + test IDs ────────────────────────────────────────────────
    labels_df = load_labels(CLINICAL_XLSX)
    test_ids  = load_split_ids(SPLIT_DIR / "test_id.txt")
    feat_dir  = model_dir / "test"
    logger.info(f"  Test patients: {len(test_ids)}")

    # ── Run inference ─────────────────────────────────────────────────────────
    logger.info("\nRunning inference on test set …")
    all_logits, all_labels, all_attns, valid_ids = run_inference(
        model, feat_dir, test_ids, labels_df, device
    )
    logger.info(f"  Valid test patients: {len(valid_ids)}")

    # ── ROC Curves ────────────────────────────────────────────────────────────
    logger.info("\nGenerating ROC curves …")
    plot_roc_curves(all_logits, all_labels, results_dir / "roc_curves.png", logger)

    # ── Confusion Matrices ────────────────────────────────────────────────────
    logger.info("Generating confusion matrices …")
    plot_confusion_matrices(all_logits, all_labels, results_dir / "confusion_matrices.png", logger)

    # ── Build per-patient predictions for heatmap titles ──────────────────────
    patient_preds = {}
    for i, pid in enumerate(valid_ids):
        p = {}
        for t in BINARY_TASKS:
            logit = all_logits[t][i].item()
            prob = 1.0 / (1.0 + np.exp(-logit))
            p[t] = f"{prob:.2f}"
        grade_logits = all_logits["Grade"][i].numpy()
        exp_g = np.exp(grade_logits - grade_logits.max())
        grade_probs = exp_g / exp_g.sum()
        p["Grade"] = f"G{grade_probs.argmax()+1}({grade_probs.max():.2f})"
        patient_preds[pid] = p

    # ── Spatial Heatmaps ──────────────────────────────────────────────────────
    if args.patients:
        heatmap_ids = [p for p in args.patients if p in all_attns]
    else:
        # Pick patients with most patches (most informative heatmaps)
        patch_counts = []
        for pid in valid_ids:
            pdir = PATCHES_ROOT / "test" / pid
            if pdir.exists():
                n = len(list(pdir.glob("*.jpg")))
                patch_counts.append((pid, n))
        patch_counts.sort(key=lambda x: -x[1])
        heatmap_ids = [pid for pid, _ in patch_counts[:args.n_heatmaps]]

    logger.info(f"\nGenerating {len(heatmap_ids)} spatial attention heatmaps …")
    success = 0
    for pid in heatmap_ids:
        out_path = heatmap_dir / f"heatmap_{pid}.png"
        patches_dir = PATCHES_ROOT / "test" / pid
        ok = generate_spatial_heatmap(
            pid, all_attns[pid], patches_dir, WSI_DIR, out_path,
            labels_df, patient_preds.get(pid, {}), logger,
        )
        if ok:
            success += 1
            logger.info(f"  [{pid}] heatmap saved → {out_path}")

    logger.info(f"\n  Heatmaps generated: {success}/{len(heatmap_ids)}")
    logger.info(f"  All outputs in: {results_dir}")
    logger.info("\nDone.")


if __name__ == "__main__":
    main()
