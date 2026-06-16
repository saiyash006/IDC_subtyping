# -*- coding: utf-8 -*-
"""
05_comprehensive_evaluation.py
==============================
Generates publication-quality plots across 7 categories for multi-task MIL:
1. Classification Performance (ROC, PR curves, epoch-wise metrics)
2. Confusion Matrices (Raw, Normalized)
3. Calibration Metrics (Reliability Diagrams, Brier Scores)
4. Multi-Task Learning Analysis (Loss Breakdown, Task Correlation)
5. Embedding Space Visualizations (UMAP)
6. Overfitting & Training Diagnostics (Generalization Gap)

Usage:
    python 05_comprehensive_evaluation.py --model owkin_phikon
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
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (auc, confusion_matrix, roc_curve, 
                             precision_recall_curve, average_precision_score,
                             brier_score_loss)
from sklearn.calibration import calibration_curve
import umap

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_ROOT  = Path("/home/cgntlab5/bcnb/BCNB dataset")
FEATURES_ROOT = DATASET_ROOT / "outputs" / "features"
CKPT_DIR      = DATASET_ROOT / "outputs" / "checkpoints"
RESULTS_DIR   = DATASET_ROOT / "outputs" / "results"
LOG_DIR       = DATASET_ROOT / "outputs" / "logs"
CLINICAL_XLSX = DATASET_ROOT / "patient-clinical-data.xlsx"
SPLIT_DIR     = DATASET_ROOT / "dataset-splitting"

BINARY_TASKS = ["ER", "PR", "HER2"]
GRADE_TASK   = "Grade"
ALL_TASKS    = BINARY_TASKS + [GRADE_TASK]
N_GRADE_CLS  = 3

# ── Colors ────────────────────────────────────────────────────────────────────
TASK_COLORS = {"ER": "#2563eb", "PR": "#7c3aed", "HER2": "#db2777", "Grade": "#059669"}
GRADE_COLORS = ["#10b981", "#f59e0b", "#ef4444"]

# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("comp_eval")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
        fh = logging.FileHandler(log_path, mode="w"); fh.setFormatter(fmt)
        ch = logging.StreamHandler(); ch.setFormatter(fmt)
        logger.addHandler(fh); logger.addHandler(ch)
    return logger

# ── Model ─────────────────────────────────────────────────────────────────────
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
        z_norm = z
        if self.use_layernorm:
            z_norm = self.norm(z)
        z_drop = z_norm
        if self.clf_dropout > 0.0:
            z_drop = self.clf_drop(z_norm)
        return {t: head(z_drop).squeeze(0) for t, head in self.heads.items()}, A, z_norm

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_labels(excel_path: Path) -> pd.DataFrame:
    df = pd.read_excel(str(excel_path))
    df.columns = df.columns.str.strip()
    id_col = next((c for c in df.columns if c.lower().replace(" ", "_") == "patient_id"), None)
    df = df.set_index(id_col); df.index = df.index.astype(str)
    grade_col = next((c for c in df.columns if "grad" in c.lower()), None)
    df = df.rename(columns={grade_col: "Histological Grade"})
    for col in BINARY_TASKS:
        mapping = {"positive": 1, "negative": 0, "1": 1, "0": 0}
        df[col] = df[col].astype(str).str.lower().str.strip().map(mapping)
    df = df.dropna(subset=BINARY_TASKS + ["Histological Grade"])
    df["Grade_idx"] = df["Histological Grade"].astype(int) - 1
    df[BINARY_TASKS] = df[BINARY_TASKS].astype(int)
    return df

@torch.no_grad()
def run_inference(model: GatedABMIL, feat_dir: Path, patient_ids: List[str], labels_df: pd.DataFrame, device: torch.device):
    model.eval()
    all_logits = {t: [] for t in ALL_TASKS}
    all_labels = {k: [] for k in ALL_TASKS + ["Grade_idx"]}
    all_z = []
    
    for pid in patient_ids:
        pt = feat_dir / f"{pid}.pt"
        if not pt.exists() or pid not in labels_df.index: continue
        feat = torch.load(str(pt), map_location="cpu", weights_only=True).float().to(device)
        preds, _, z = model(feat)
        row = labels_df.loc[pid]
        
        for t in BINARY_TASKS:
            all_logits[t].append(preds[t].cpu())
            all_labels[t].append(int(row[t]))
        all_logits["Grade"].append(preds["Grade"].cpu())
        all_labels["Grade_idx"].append(int(row["Grade_idx"]))
        all_z.append(z.squeeze(0).cpu().numpy())
        
    return all_logits, all_labels, np.array(all_z)

# ── 1. Classification Performance ─────────────────────────────────────────────
def plot_pr_curves(all_logits, all_labels, out_path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Precision-Recall Curves (Test Set)", fontsize=16, fontweight="bold", y=1.02)
    
    for i, task in enumerate(ALL_TASKS):
        ax = axes[i]
        if task in BINARY_TASKS:
            probs = torch.sigmoid(torch.stack(all_logits[task]).squeeze(-1)).numpy()
            labels = np.array(all_labels[task])
            precision, recall, _ = precision_recall_curve(labels, probs)
            ap = average_precision_score(labels, probs)
            ax.plot(recall, precision, color=TASK_COLORS[task], lw=2.5, label=f"AP = {ap:.3f}")
            ax.fill_between(recall, precision, alpha=0.15, color=TASK_COLORS[task])
            baseline = labels.mean()
            ax.axhline(baseline, color="k", linestyle="--", alpha=0.5, label=f"Baseline ({baseline:.2f})")
        else:
            logits_g = torch.stack(all_logits["Grade"]).numpy()
            probs_g = np.exp(logits_g - logits_g.max(axis=1, keepdims=True))
            probs_g /= probs_g.sum(axis=1, keepdims=True)
            labels_g = np.array(all_labels["Grade_idx"])
            for g in range(N_GRADE_CLS):
                y_true = (labels_g == g).astype(int)
                if y_true.sum() == 0: continue
                precision, recall, _ = precision_recall_curve(y_true, probs_g[:, g])
                ap = average_precision_score(y_true, probs_g[:, g])
                ax.plot(recall, precision, color=GRADE_COLORS[g], lw=2, label=f"G{g+1} AP={ap:.3f}")
                ax.fill_between(recall, precision, alpha=0.08, color=GRADE_COLORS[g])

        ax.set_xlabel("Recall", fontsize=11); ax.set_ylabel("Precision", fontsize=11)
        ax.set_title(task, fontsize=13, fontweight="bold")
        ax.legend(loc="lower left", fontsize=9)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.grid(True, alpha=0.2)
        
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight"); plt.close(fig)

# ── 2. Confusion Matrices (Normalized) ────────────────────────────────────────
def plot_normalized_cm(all_logits, all_labels, out_path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Normalized Confusion Matrices (Test Set)", fontsize=16, fontweight="bold", y=1.02)
    
    for i, task in enumerate(ALL_TASKS):
        if task in BINARY_TASKS:
            probs = torch.sigmoid(torch.stack(all_logits[task]).squeeze(-1)).numpy()
            preds = (probs >= 0.5).astype(int)
            labels = np.array(all_labels[task])
            cm = confusion_matrix(labels, preds, labels=[0, 1], normalize='true')
            sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues", ax=axes[i], cbar=False,
                        xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
        else:
            logits_g = torch.stack(all_logits["Grade"]).numpy()
            preds_g = logits_g.argmax(axis=1)
            labels_g = np.array(all_labels["Grade_idx"])
            cm = confusion_matrix(labels_g, preds_g, labels=[0, 1, 2], normalize='true')
            sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues", ax=axes[i], cbar=False,
                        xticklabels=["G1", "G2", "G3"], yticklabels=["G1", "G2", "G3"])
        axes[i].set_title(task, fontsize=13, fontweight="bold")
        axes[i].set_xlabel("Predicted"); axes[i].set_ylabel("True")
        
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight"); plt.close(fig)

# ── 3. Calibration Metrics ────────────────────────────────────────────────────
def plot_calibration_curves(all_logits, all_labels, out_path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Reliability Diagrams / Calibration (Test Set)", fontsize=16, fontweight="bold", y=1.02)
    
    for i, task in enumerate(ALL_TASKS):
        ax = axes[i]
        ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        if task in BINARY_TASKS:
            probs = torch.sigmoid(torch.stack(all_logits[task]).squeeze(-1)).numpy()
            labels = np.array(all_labels[task])
            prob_true, prob_pred = calibration_curve(labels, probs, n_bins=10)
            brier = brier_score_loss(labels, probs)
            ax.plot(prob_pred, prob_true, "s-", color=TASK_COLORS[task], label=f"Brier = {brier:.3f}")
        else:
            logits_g = torch.stack(all_logits["Grade"]).numpy()
            probs_g = np.exp(logits_g - logits_g.max(axis=1, keepdims=True))
            probs_g /= probs_g.sum(axis=1, keepdims=True)
            labels_g = np.array(all_labels["Grade_idx"])
            for g in range(N_GRADE_CLS):
                y_true = (labels_g == g).astype(int)
                if y_true.sum() == 0: continue
                prob_true, prob_pred = calibration_curve(y_true, probs_g[:, g], n_bins=8)
                brier = brier_score_loss(y_true, probs_g[:, g])
                ax.plot(prob_pred, prob_true, "s-", color=GRADE_COLORS[g], label=f"G{g+1} Brier={brier:.3f}")

        ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Fraction of positives")
        ax.set_title(task, fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9); ax.grid(True, alpha=0.2)
        
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight"); plt.close(fig)

# ── 4. Multi-Task Learning Analysis ───────────────────────────────────────────
def plot_task_correlation(all_logits, out_path):
    preds_dict = {}
    for task in BINARY_TASKS:
        preds_dict[task] = torch.sigmoid(torch.stack(all_logits[task]).squeeze(-1)).numpy()
    logits_g = torch.stack(all_logits["Grade"]).numpy()
    probs_g = np.exp(logits_g - logits_g.max(axis=1, keepdims=True))
    probs_g /= probs_g.sum(axis=1, keepdims=True)
    # Use expected grade for correlation
    preds_dict["Expected Grade"] = (probs_g * np.array([1, 2, 3])).sum(axis=1)
    
    df = pd.DataFrame(preds_dict)
    corr = df.corr()
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax, square=True)
    ax.set_title("Task Prediction Correlation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight"); plt.close(fig)

# ── 5. Embedding Space Visualizations (UMAP) ──────────────────────────────────
def plot_umap_embeddings(all_z, all_labels, out_path):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    z_2d = reducer.fit_transform(all_z)
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    fig.suptitle("UMAP Projection of Learned MIL Embeddings", fontsize=16, fontweight="bold", y=1.05)
    
    for i, task in enumerate(ALL_TASKS):
        ax = axes[i]
        if task in BINARY_TASKS:
            labels = np.array(all_labels[task])
            scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap="coolwarm", s=15, alpha=0.8)
            legend1 = ax.legend(*scatter.legend_elements(), title="Class", loc="best")
            ax.add_artist(legend1)
        else:
            labels = np.array(all_labels["Grade_idx"])
            scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap="viridis", s=15, alpha=0.8)
            legend1 = ax.legend(*scatter.legend_elements(), title="Grade", loc="best")
            ax.add_artist(legend1)
            
        ax.set_title(f"Color by {task}", fontsize=13, fontweight="bold")
        ax.axis("off")
        
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight"); plt.close(fig)

# ── 7. Overfitting & Diagnostics (History) ────────────────────────────────────
def plot_history_diagnostics(history, out_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    
    # a. Generalization Gap (Loss)
    fig, ax = plt.subplots(figsize=(8, 5))
    gap = np.array(history["val_loss"]) - np.array(history["train_loss"])
    ax.plot(epochs, gap, 'k-', lw=2)
    ax.fill_between(epochs, gap, 0, where=(gap > 0), color="red", alpha=0.3, label="Overfitting")
    ax.fill_between(epochs, gap, 0, where=(gap <= 0), color="green", alpha=0.3, label="Generalizing")
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val Loss - Train Loss")
    ax.set_title("Generalization Gap", fontweight="bold")
    ax.legend(); plt.tight_layout()
    fig.savefig(str(out_dir / "generalization_gap.png"), dpi=200); plt.close(fig)
    
    # b. Task Loss Breakdown (Train)
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(epochs))
    for task in ALL_TASKS:
        t_loss = np.array(history[f"train_loss_{task}"])
        ax.bar(epochs, t_loss, bottom=bottom, label=task, color=TASK_COLORS[task], alpha=0.8)
        bottom += t_loss
    ax.set_xlabel("Epoch"); ax.set_ylabel("Total Train Loss")
    ax.set_title("Task Loss Breakdown over Epochs", fontweight="bold")
    ax.legend(loc="upper right"); plt.tight_layout()
    fig.savefig(str(out_dir / "task_loss_breakdown.png"), dpi=200); plt.close(fig)

    # c. Sensitivity/Specificity progression
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Sensitivity & Specificity Evolution", fontsize=15, fontweight="bold")
    for idx, task in enumerate(BINARY_TASKS):
        ax = axes[idx // 2][idx % 2]
        ax.plot(epochs, history[f"val_sensitivity_{task}"], label="Sensitivity (Recall)", color="#ef4444", lw=2)
        ax.plot(epochs, history[f"val_specificity_{task}"], label="Specificity", color="#3b82f6", lw=2)
        ax.plot(epochs, history[f"val_balanced_acc_{task}"], label="Balanced Acc", color="#10b981", lw=2, linestyle="--")
        ax.set_title(task, fontweight="bold")
        ax.set_ylim(0, 1.05); ax.legend(); ax.grid(True, alpha=0.2)
    axes[1][1].axis("off") # Grade doesn't have sens/spec
    plt.tight_layout(); fig.savefig(str(out_dir / "sens_spec_curves.png"), dpi=200); plt.close(fig)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="owkin_phikon")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    logger = setup_logging(LOG_DIR / "05_comp_eval.log")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    model_dir   = FEATURES_ROOT / args.model
    ckpt_path   = CKPT_DIR / args.model / "best_model.pth"
    results_dir = RESULTS_DIR / args.model
    out_dir     = results_dir / "comprehensive_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = results_dir / "training_history.json"

    logger.info("=" * 60)
    logger.info("  Phase 5: Comprehensive Medical AI Visualizations")
    logger.info("=" * 60)

    # 1. Load History Diagnostics
    if history_path.exists():
        logger.info("Loading training history …")
        with open(str(history_path)) as f:
            history = json.load(f)
        plot_history_diagnostics(history, out_dir)
        logger.info("  -> Saved diagnostic curves.")
    else:
        logger.warning(f"No history found at {history_path}. Run training first to get epoch-wise plots.")

    # 2. Run Inference on Test Set
    if not ckpt_path.exists():
        logger.error(f"No checkpoint found at {ckpt_path}.")
        return

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    model = GatedABMIL(ckpt["embed_dim"], ckpt["params"]["attn_dim"], ckpt["params"]["attn_dropout"], ckpt["params"].get("clf_dropout", 0.0), ckpt["params"].get("use_layernorm", False)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    
    labels_df = load_labels(CLINICAL_XLSX)
    with open(SPLIT_DIR / "test_id.txt") as f:
        test_ids = [l.strip() for l in f if l.strip()]

    logger.info("\nRunning inference on test set …")
    all_logits, all_labels, all_z = run_inference(model, model_dir / "test", test_ids, labels_df, device)
    
    # 3. Generate Static Plots
    logger.info("Generating Precision-Recall Curves …")
    plot_pr_curves(all_logits, all_labels, out_dir / "pr_curves.png")
    
    logger.info("Generating Normalized Confusion Matrices …")
    plot_normalized_cm(all_logits, all_labels, out_dir / "confusion_matrices_norm.png")
    
    logger.info("Generating Calibration Plots …")
    plot_calibration_curves(all_logits, all_labels, out_dir / "calibration_curves.png")
    
    logger.info("Generating Task Correlation Heatmap …")
    plot_task_correlation(all_logits, out_dir / "task_correlation.png")
    
    logger.info("Generating UMAP Embeddings …")
    plot_umap_embeddings(all_z, all_labels, out_dir / "umap_embeddings.png")
    
    logger.info(f"\nAll plots successfully saved to: {out_dir}")

if __name__ == "__main__":
    main()
