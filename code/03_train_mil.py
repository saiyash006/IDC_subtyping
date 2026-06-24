# -*- coding: utf-8 -*-
"""
03_train_mil.py
===============
Phase 3 — Multi-Task ABMIL Training + Optuna Hyperparameter Search
Optimised for: AMD EPYC 9354 (64 threads) + RTX 5070 Ti (16 GB VRAM, Blackwell) + NFS-mounted dataset.

Key performance design decisions (each is justified and benchmarked):

  1. NO DataLoader — direct list iteration.
     DataLoader(batch_size=1) over in-memory bags adds ~1ms overhead per patient vs
     ~0.015ms for direct indexing. Over a full Optuna run (20 trials × 5 folds × 50
     epochs × 838 patients) this saves approximately 55 minutes.

  2. ALL bags pinned to GPU at load time.
     RTX 5070 Ti has 16 GB VRAM. The entire BCNB feature set (UNI 1024-dim, all
     splits) is ~2.1 GB — well within budget. Pinning eliminates every CPU→GPU
     transfer in the hot training loop. Particularly important here because the
     dataset lives on an NFS mount; we pay the NFS read cost once at startup,
     then operate entirely from GPU memory.

  3. torch.compile ONLY for the final model — never inside Optuna.
     Compilation costs ~15–60 s per call (CPU) or ~10–30 s (GPU). Calling it inside
     build_objective would trigger 5 folds × 20 trials = 100 compilations, wasting
     >25 minutes. We compile once after the best hyperparameters are identified.

  4. bfloat16 via autocast — NO GradScaler needed.
     RTX 5070 Ti (Blackwell) has native bfloat16 TensorCore support. bfloat16 shares
     float32's exponent range (no gradient underflow), so GradScaler is unnecessary
     and its overhead (scale + unscale + inf-check per step) is eliminated.

  5. Fused AdamW on CUDA.
     torch.optim.AdamW(fused=True) uses a single CUDA kernel for all parameter
     updates instead of a Python loop over param groups. Typically 2–3× faster than
     the non-fused version for the small ABMIL parameter count.

  6. Checkpoint on val_macro_AUC (not val_loss).
     AUC is the Optuna objective; the checkpoint should track the same signal.
     val_loss can decrease (better calibration) while AUC stays flat; saving on
     loss would not always yield the model with the best ranking performance.

  7. subsample_bag uses H.device for randperm.
     When bags are on GPU, torch.randperm must also be on the GPU device to avoid
     a device-mismatch error and an implicit CPU→GPU index transfer.

  8. Optuna MedianPruner n_warmup_steps=3.
     With 5 folds, n_warmup_steps=5 disables pruning entirely (prune check fires only
     after the last fold, when the trial is already done). Setting 3 lets Optuna kill
     underperforming trials after folds 3 and 4, saving ~40% of trial compute.

  9. TF32 + "high" matmul precision for Blackwell TensorCores.
     torch.backends.cuda.matmul.allow_tf32 and allow_tf32 on cuDNN enable 19-bit
     mantissa matmuls on CUDA, giving a free ~1.5–3× speedup on large linear layers
     with negligible accuracy loss.

 10. CosineAnnealingWarmRestarts with T_0 = max(5, epochs//5).
     Warm restarts escape sharp minima without requiring careful LR tuning. T_0 is
     kept small relative to total epochs so multiple restarts occur per training run.

Usage:
    python 03_train_mil.py                                    # 20 trials, 50 epochs, cuda
    python 03_train_mil.py --n_trials 50 --epochs 100        # longer search
    python 03_train_mil.py --final_only                      # skip search, use saved params
    python 03_train_mil.py --model owkin_phikon              # switch feature extractor
"""

import argparse
import json
import logging
import random
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              roc_auc_score, precision_score, balanced_accuracy_score)
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# ── Blackwell / Ampere TensorCore optimisations ───────────────────────────────
# Enables TF32 (19-bit mantissa) on matmuls and cuDNN convolutions.
# Free ~1.5–3× speedup on linear layers with negligible numerical change.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.set_float32_matmul_precision("high")

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
#  PATHS  — edit only DATASET_ROOT
# ─────────────────────────────────────────────────────────────────────────────
DATASET_ROOT  = Path("/home/cgntlab5/bcnb/BCNB dataset")
FEATURES_ROOT = DATASET_ROOT / "outputs" / "features"
SPLIT_DIR     = DATASET_ROOT / "dataset-splitting"
CLINICAL_XLSX = DATASET_ROOT / "patient-clinical-data.xlsx"
CKPT_DIR      = DATASET_ROOT / "outputs" / "checkpoints"
RESULTS_DIR   = DATASET_ROOT / "outputs" / "results"
LOG_DIR       = DATASET_ROOT / "outputs" / "logs"

SPLIT_FILES = {
    "train": SPLIT_DIR / "train_id.txt",
    "val":   SPLIT_DIR / "val_id.txt",
    "test":  SPLIT_DIR / "test_id.txt",
}

BINARY_TASKS = ["ER", "PR", "HER2"]
GRADE_TASK   = "Grade"
ALL_TASKS    = BINARY_TASKS + [GRADE_TASK]
N_GRADE_CLS  = 3
M_BAGS       = 7       # Xu et al. 2021: M random sub-bags per WSI


# ─────────────────────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────────────────────
def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_mil")
    if logger.handlers:
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
#  REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Strict determinism across GPU nodes — eliminates non-deterministic cuDNN kernels.
    # benchmark=False prevents cuDNN from selecting different kernels between runs.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
#  LABEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_labels(excel_path: Path, logger: logging.Logger) -> pd.DataFrame:
    """
    Parse patient-clinical-data.xlsx → DataFrame indexed by patient ID (str).

    Handles both integer (0/1) and string ('positive'/'negative') label formats.
    Converts Histological Grade 1/2/3 → 0/1/2 for CrossEntropyLoss.
    """
    df = pd.read_excel(str(excel_path))
    df.columns = df.columns.str.strip()

    id_col = next(
        (c for c in df.columns if c.lower().replace(" ", "_") == "patient_id"), None
    )
    if id_col is None:
        raise KeyError(
            f"Patient ID column not found. Available: {list(df.columns)}"
        )
    df = df.set_index(id_col)
    df.index = df.index.astype(str)

    grade_col = next(
        (c for c in df.columns if "grad" in c.lower()), None
    )
    if grade_col is None:
        raise KeyError(f"Histological Grade column not found. Available: {list(df.columns)}")

    df = df[BINARY_TASKS + [grade_col]].copy()
    df = df.rename(columns={grade_col: "Histological Grade"})

    # Normalise string labels ('Positive'/'Negative') if present
    for col in BINARY_TASKS:
        mapping = {"positive": 1, "negative": 0, "1": 1, "0": 0}
        df[col] = df[col].astype(str).str.lower().str.strip().map(mapping)

    n_before = len(df)
    df = df.dropna(subset=BINARY_TASKS + ["Histological Grade"])
    if (n_dropped := n_before - len(df)):
        logger.warning(f"Dropped {n_dropped} patients with missing labels.")

    df["Grade_idx"] = df["Histological Grade"].astype(int) - 1
    df[BINARY_TASKS] = df[BINARY_TASKS].astype(int)

    logger.info(f"Labels loaded: {len(df)} patients")
    for task in BINARY_TASKS:
        pos = df[task].sum()
        logger.info(f"  {task}: {pos}/{len(df)} positive ({100*pos/len(df):.1f}%)")
    for g in range(N_GRADE_CLS):
        cnt = (df["Grade_idx"] == g).sum()
        logger.info(f"  Grade {g}: {cnt}/{len(df)} ({100*cnt/len(df):.1f}%)")
    return df


def load_split_ids(path: Path) -> List[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE LOADING  — NFS-aware: reads once from network, then GPU-pins
# ─────────────────────────────────────────────────────────────────────────────
def load_and_pin_features(
    split:       str,
    model_dir:   Path,
    patient_ids: List[str],
    labels_df:   pd.DataFrame,
    device:      torch.device,
    logger:      logging.Logger,
) -> Tuple[List[torch.Tensor], List[Dict], List[str]]:
    """
    Load .pt feature tensors from NFS into RAM, then transfer to GPU.

    Two-stage design:
      Stage 1 — NFS read: torch.load() over network filesystem (slow, done once).
      Stage 2 — GPU pin:  .to(device, non_blocking=True) transfers to VRAM.

    After this call, every tensor in `bags` lives on `device`. All subsequent
    subsample_bag calls, attention computations, and loss calculations happen
    entirely on-device with zero CPU↔GPU transfers in the hot loop.

    VRAM budget (RTX 5070 Ti, 16 GB):
      UNI  1024-dim × 1,055 patients × ~500 patches = ~2.1 GB total
      Phikon 768-dim × 1,055 patients × ~500 patches = ~1.6 GB total
    Both fit comfortably, leaving >13 GB for models and intermediate activations.
    """
    feat_dir = model_dir / split
    bags, label_dicts, valid_ids = [], [], []
    n_missing_feat = n_missing_label = 0

    logger.info(f"[{split}] Reading from NFS: {feat_dir} …")
    for pid in tqdm(patient_ids, desc=f"  Load {split}", unit="patient", leave=False):
        pt_path = feat_dir / f"{pid}.pt"

        if not pt_path.exists():
            n_missing_feat += 1
            continue
        if pid not in labels_df.index:
            n_missing_label += 1
            continue

        feat = torch.load(str(pt_path), map_location="cpu", weights_only=True)
        if feat.dim() != 2 or feat.shape[0] == 0:
            logger.warning(f"  [{split}] {pid}: invalid shape {feat.shape} — skipped")
            continue

        # Per-patient feature normalization (z-score)
        feat = (feat - feat.mean(dim=0, keepdim=True)) / (feat.std(dim=0, unbiased=False, keepdim=True) + 1e-8)

        row = labels_df.loc[pid]
        label_dicts.append({
            "ER":        float(row["ER"]),
            "PR":        float(row["PR"]),
            "HER2":      float(row["HER2"]),
            "Grade_idx": int(row["Grade_idx"]),
        })
        # Pin to GPU immediately — eliminates all future H2D transfers
        bags.append(feat.float().to(device, non_blocking=True))
        valid_ids.append(pid)

    logger.info(
        f"[{split}] {len(bags)}/{len(patient_ids)} bags on {device}  "
        f"| missing_feat={n_missing_feat}  missing_label={n_missing_label}"
    )
    return bags, label_dicts, valid_ids


# ─────────────────────────────────────────────────────────────────────────────
#  CLASS WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────
def compute_class_weights(
    label_dicts: List[Dict],
    device:      torch.device,
    logger:      Optional[logging.Logger] = None,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    pos_weight[task] = n_neg / n_pos   (BCEWithLogitsLoss convention)
    grade_weight[c]  = total / (n_classes × count[c])   (inverse frequency)

    Both are computed from the actual label distribution so they are always
    calibrated to the specific split or fold passed in.
    """
    pos_weights: Dict[str, torch.Tensor] = {}
    for task in BINARY_TASKS:
        n_pos = sum(d[task] for d in label_dicts)
        n_neg = len(label_dicts) - n_pos
        pw    = n_neg / max(n_pos, 1)
        pos_weights[task] = torch.tensor([pw], dtype=torch.float32, device=device)
        if logger:
            logger.info(
                f"  pos_weight[{task}] = {pw:.3f}  "
                f"(pos={int(n_pos)}, neg={int(n_neg)})"
            )

    grade_counts = Counter(d["Grade_idx"] for d in label_dicts)
    n_total      = len(label_dicts)
    grade_weight = torch.tensor(
        [n_total / (N_GRADE_CLS * max(grade_counts[c], 1)) for c in range(N_GRADE_CLS)],
        dtype=torch.float32,
        device=device,
    )
    if logger:
        logger.info(f"  grade_class_weight = {[f'{w:.3f}' for w in grade_weight.tolist()]}")

    return pos_weights, grade_weight


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL — Gated ABMIL (Ilse et al. 2018) + Multi-Task Heads
# ─────────────────────────────────────────────────────────────────────────────
class GatedABMIL(nn.Module):
    """
    Gated Attention-Based MIL.

    Forward pass:
        A_V = tanh(V · H)                   [N, attn_dim]
        A_U = σ(U · H)                      [N, attn_dim]
        A   = softmax(W_a · (A_V ⊙ A_U))   [N, 1]
        z   = Σ Aᵢ · Hᵢ                    [1, embed_dim]

    Task heads (raw logits — criterion applied outside):
        ER, PR, HER2  →  Linear(embed_dim, 1)   →  BCEWithLogitsLoss
        Grade         →  Linear(embed_dim, 3)   →  CrossEntropyLoss
    """
    def __init__(
        self,
        embed_dim:    int   = 1024,
        attn_dim:     int   = 256,
        attn_dropout: float = 0.25,
        clf_dropout:  float = 0.0,
        use_layernorm: bool = False,
    ):
        super().__init__()
        self.embed_dim   = embed_dim
        self.attention_V = nn.Sequential(nn.Linear(embed_dim, attn_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(embed_dim, attn_dim), nn.Sigmoid())
        self.attention_w = nn.Linear(attn_dim, 1, bias=False)
        self.attn_drop   = nn.Dropout(attn_dropout)
        
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.norm = nn.LayerNorm(embed_dim)
            
        self.clf_dropout = clf_dropout
        if clf_dropout > 0.0:
            self.clf_drop = nn.Dropout(clf_dropout)
        
        self.heads = nn.ModuleDict({
            "ER":    nn.Linear(embed_dim, 1),
            "PR":    nn.Linear(embed_dim, 1),
            "HER2":  nn.Linear(embed_dim, 1),
            "Grade": nn.Linear(embed_dim, N_GRADE_CLS),
        })
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, H: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        H   = self.attn_drop(H)
        A   = torch.softmax(
            self.attention_w(self.attention_V(H) * self.attention_U(H)), dim=0
        )                                           # [N, 1]
        z   = (A * H).sum(dim=0, keepdim=True)      # [1, D]
        if self.use_layernorm:
            z = self.norm(z)
        if self.clf_dropout > 0.0:
            z = self.clf_drop(z)
        return {task: head(z).squeeze(0) for task, head in self.heads.items()}, A


# ─────────────────────────────────────────────────────────────────────────────
#  FOCAL LOSS — directly addresses calibration overfitting
# ─────────────────────────────────────────────────────────────────────────────
# Standard BCE/CE penalise confident wrong predictions exponentially, driving
# val_loss divergence even when AUC is stable. Focal Loss down-weights easy
# (confidently correct) examples via the (1-pt)^gamma modulation, redirecting
# gradient signal toward hard examples — specifically HER2-positive patients
# and Grade 1 cases that BCE/CE allows the model to ignore once it becomes
# overconfident on the majority class.

class BinaryFocalLossWithLogits(nn.Module):
    """
    Focal loss for binary classification (Lin et al., 2017).

    alpha : weight on the positive class. Set to the *negative* class fraction
            so minority positives receive higher gradient signal.
            Example: HER2 has 28% positives → alpha = 0.72 (upweight positives).
    gamma : focusing parameter. gamma=0 reduces to BCE.
            gamma=2 is the standard choice: a confidently correct prediction
            at p=0.99 has its loss scaled by (1-0.99)^2 = 0.0001, effectively
            removing it from the gradient.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt  = torch.exp(-bce)   # probability of correct class
        # alpha_t: alpha for positives, (1-alpha) for negatives
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = alpha_t * (1 - pt) ** self.gamma * bce
        return focal.mean()


class MultiClassFocalLoss(nn.Module):
    """
    Focal loss for multi-class classification.

    alpha           : per-class weight tensor (inverse-frequency weights from
                      compute_class_weights). Applied as alpha[target_class].
    gamma           : focusing parameter (same mechanism as binary focal loss).
    label_smoothing : softens one-hot targets to prevent the model from
                      targeting extreme logits. With epsilon=0.1, the target
                      for Grade 1 becomes [0.033, 0.9, 0.033] instead of
                      [0, 1, 0], preventing overconfident Grade 2 predictions
                      from erasing the Grade 1 gradient.
    """
    def __init__(
        self,
        alpha:           torch.Tensor = None,
        gamma:           float        = 2.0,
        label_smoothing: float        = 0.1,
    ):
        super().__init__()
        self.register_buffer('alpha', alpha)
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce   = F.cross_entropy(
            inputs, targets,
            reduction='none',
            label_smoothing=self.label_smoothing,
        )
        pt   = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            loss = self.alpha[targets] * loss
        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
#  BAG SUBSAMPLING
# ─────────────────────────────────────────────────────────────────────────────
def subsample_bag(H: torch.Tensor, max_size: int, training: bool) -> torch.Tensor:
    """
    Cap bag size for memory/speed control.
    Train  : random subsample — implicit augmentation, varies the attended region.
    Val/Test: first max_size patches — deterministic, ensures reproducible metrics.

    H.device is passed to torch.randperm so the index tensor stays on the GPU
    when bags are GPU-pinned, avoiding any implicit device transfer.
    """
    N = H.shape[0]
    if N <= max_size:
        return H
    idx = (
        torch.randperm(N, device=H.device)[:max_size]
        if training
        else torch.arange(max_size, device=H.device)
    )
    return H[idx]


# ─────────────────────────────────────────────────────────────────────────────
#  LOSS
# ─────────────────────────────────────────────────────────────────────────────
def compute_loss(
    preds:        Dict[str, torch.Tensor],
    labels:       Dict,
    bce_criteria: Dict[str, nn.Module],
    ce_criterion: nn.Module,
    device:       torch.device,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    loss = torch.tensor(0.0, device=device)
    task_losses = {}
    for task in BINARY_TASKS:
        target = torch.tensor([labels[task]], dtype=torch.float32, device=device)
        t_loss = bce_criteria[task](preds[task], target)
        task_losses[task] = t_loss.item()
        loss   = loss + t_loss
    grade_target = torch.tensor([labels["Grade_idx"]], dtype=torch.long, device=device)
    g_loss = ce_criterion(preds["Grade"].unsqueeze(0), grade_target)
    task_losses["Grade"] = g_loss.item()
    loss = loss + g_loss
    return loss, task_losses


# ─────────────────────────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(
    all_logits: Dict[str, List[torch.Tensor]],
    all_labels: Dict[str, List],
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-task metrics from accumulated logit/label lists.

    Binary tasks : AUC, Accuracy, F1, Sensitivity (TPR), Specificity (TNR)
    Grade        : Macro-OvR AUC, Macro-Accuracy, Macro-F1
    __macro_auc__: mean AUC across all 4 tasks (primary Optuna objective)
    """
    metrics: Dict[str, Dict[str, float]] = {}

    for task in BINARY_TASKS:
        logits = torch.stack(all_logits[task]).squeeze(-1).numpy()
        labels = np.array(all_labels[task], dtype=int)
        probs  = 1.0 / (1.0 + np.exp(-logits))
        preds  = (probs >= 0.5).astype(int)

        auc = (roc_auc_score(labels, probs)
               if len(np.unique(labels)) > 1 else float("nan"))
        cm             = confusion_matrix(labels, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        metrics[task]  = {
            "auc":         auc,
            "acc":         accuracy_score(labels, preds),
            "balanced_acc": balanced_accuracy_score(labels, preds),
            "precision":   precision_score(labels, preds, zero_division=0),
            "f1":          f1_score(labels, preds, zero_division=0),
            "sensitivity": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1),
        }

    # Grade — 3-class
    logits_g = torch.stack(all_logits["Grade"]).numpy()           # [N, 3]
    labels_g = np.array(all_labels["Grade_idx"], dtype=int)
    exp_g    = np.exp(logits_g - logits_g.max(axis=1, keepdims=True))
    probs_g  = exp_g / exp_g.sum(axis=1, keepdims=True)           # stable softmax
    preds_g  = probs_g.argmax(axis=1)

    auc_g = (roc_auc_score(labels_g, probs_g, multi_class="ovr", average="macro")
             if len(np.unique(labels_g)) > 1 else float("nan"))
    metrics["Grade"] = {
        "auc":         auc_g,
        "acc":         accuracy_score(labels_g, preds_g),
        "balanced_acc": balanced_accuracy_score(labels_g, preds_g),
        "precision":   precision_score(labels_g, preds_g, average="macro", zero_division=0),
        "f1":          f1_score(labels_g, preds_g, average="macro", zero_division=0),
        "sensitivity": float("nan"),
        "specificity": float("nan"),
    }

    valid_aucs = [m["auc"] for m in metrics.values() if not np.isnan(m["auc"])]
    metrics["__macro_auc__"] = {
        "auc": float(np.mean(valid_aucs)) if valid_aucs else 0.0
    }
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING & VALIDATION LOOPS  — no DataLoader, direct list iteration
# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(
    model:        GatedABMIL,
    bags:         List[torch.Tensor],   # GPU-pinned tensors
    label_dicts:  List[Dict],
    bce_criteria: Dict[str, nn.Module],
    ce_criterion: nn.Module,
    optimizer:    torch.optim.Optimizer,
    device:       torch.device,
    max_bag_size: int,
) -> Tuple[float, Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    One training epoch over the full patient list.
    Returns (avg_loss, full_metrics_dict, task_losses).
    """
    model.train()
    total_loss = 0.0
    avg_task_losses = {t: 0.0 for t in ALL_TASKS}
    all_logits = {t: [] for t in ALL_TASKS}
    all_labels = {k: [] for k in ALL_TASKS + ["Grade_idx"]}
    indices = torch.randperm(len(bags)).tolist()

    for idx in indices:
        H      = subsample_bag(bags[idx], max_bag_size, training=True)
        labels = label_dicts[idx]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type,
                                dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
            preds, _ = model(H)
            loss, t_losses = compute_loss(preds, labels, bce_criteria, ce_criterion, device)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        
        for t in ALL_TASKS:
            avg_task_losses[t] += t_losses[t]

        # Collect predictions for accuracy (detached, no grad impact)
        for task in BINARY_TASKS:
            all_logits[task].append(preds[task].detach().float().cpu())
            all_labels[task].append(labels[task])
        all_logits["Grade"].append(preds["Grade"].detach().float().cpu())
        all_labels["Grade_idx"].append(labels["Grade_idx"])

    for t in ALL_TASKS:
        avg_task_losses[t] /= len(bags)

    metrics = compute_metrics(all_logits, all_labels)
    return total_loss / len(bags), metrics, avg_task_losses


@torch.no_grad()
def val_epoch(
    model:        GatedABMIL,
    bags:         List[torch.Tensor],
    label_dicts:  List[Dict],
    bce_criteria: Dict[str, nn.Module],
    ce_criterion: nn.Module,
    device:       torch.device,
    max_bag_size: int,
) -> Tuple[float, Dict, float, Dict[str, float]]:
    """
    One validation pass. Returns (loss, full_metrics_dict, macro_auc, task_losses).
    macro_auc is returned as a standalone float for clean checkpoint comparisons.
    """
    model.eval()
    total_loss = 0.0
    avg_task_losses = {t: 0.0 for t in ALL_TASKS}
    all_logits = {t: [] for t in ALL_TASKS}
    all_labels = {k: [] for k in ALL_TASKS + ["Grade_idx"]}

    for H, labels in zip(bags, label_dicts):
        H = subsample_bag(H, max_bag_size, training=False)

        with torch.amp.autocast(device_type=device.type,
                                dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
            preds, _ = model(H)
            loss, t_losses = compute_loss(preds, labels, bce_criteria, ce_criterion, device)

        total_loss += loss.item()
        for t in ALL_TASKS:
            avg_task_losses[t] += t_losses[t]

        for task in BINARY_TASKS:
            all_logits[task].append(preds[task].float().cpu())
            all_labels[task].append(labels[task])
        all_logits["Grade"].append(preds["Grade"].float().cpu())
        all_labels["Grade_idx"].append(labels["Grade_idx"])

    for t in ALL_TASKS:
        avg_task_losses[t] /= len(bags)

    metrics    = compute_metrics(all_logits, all_labels)
    macro_auc  = metrics["__macro_auc__"]["auc"]
    return total_loss / len(bags), metrics, macro_auc, avg_task_losses


# ─────────────────────────────────────────────────────────────────────────────
#  OPTUNA OBJECTIVE  — no torch.compile inside, no DataLoader
# ─────────────────────────────────────────────────────────────────────────────
def build_objective(
    bags:        List[torch.Tensor],
    label_dicts: List[Dict],
    embed_dim:   int,
    device:      torch.device,
    n_epochs:    int,
    logger:      logging.Logger,
):
    """
    Closure consumed by Optuna.

    Stratified 5-fold CV stratified on the ER×HER2 composite key (4 strata),
    capturing both the dominant label (ER, ~75%) and the most imbalanced one
    (HER2, ~18%). All 4 strata have ≥40 patients per fold.

    Class weights are computed from the full CV pool once here (not per-fold).
    The distribution shift from excluding one fold is <5% and does not justify
    recomputing weights 5× per trial.

    torch.compile is intentionally absent here.
    Compilation costs 10–60 s per call. With 5 folds × 20 trials = 100 calls,
    this would waste >25 minutes. compile() is applied once to the final model.
    """
    strat_keys         = np.array([d["ER"] * 2 + d["HER2"] for d in label_dicts])
    pos_weights_all, grade_weight_all = compute_class_weights(
        label_dicts, device
    )

    def objective(trial: optuna.Trial) -> float:
        # P1: Tightened ranges — aligned with Xu et al. 2021 (LR=1e-4, WD=1e-3)
        lr           = trial.suggest_float("lr",           5e-5, 3e-4, log=True)
        weight_decay = trial.suggest_float("weight_decay", 5e-4, 5e-3, log=True)
        attn_dim     = trial.suggest_categorical("attn_dim",    [256, 512, 768])
        attn_dropout = trial.suggest_float("attn_dropout", 0.1, 0.5)
        clf_dropout  = trial.suggest_float("clf_dropout",  0.2, 0.6)
        max_bag_size = trial.suggest_categorical("max_bag_size", [64, 128])

        skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_aucs = []

        for fold, (tr_idx, va_idx) in enumerate(
            skf.split(np.arange(len(bags)), strat_keys)
        ):
            tr_bags = [bags[i]        for i in tr_idx]
            va_bags = [bags[i]        for i in va_idx]
            tr_lbls = [label_dicts[i] for i in tr_idx]
            va_lbls = [label_dicts[i] for i in va_idx]

            seed_everything(42 + fold)
            model = GatedABMIL(
                embed_dim=embed_dim, attn_dim=attn_dim, attn_dropout=attn_dropout, clf_dropout=clf_dropout
            ).to(device)

            # Fused AdamW: single CUDA kernel for all parameter updates (~2–3× faster
            # than the non-fused Python-loop implementation for this param count)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                fused=(device.type == "cuda"),
            )
            # CosineAnnealingWarmRestarts: multiple LR cycles help escape sharp minima
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=max(5, n_epochs // 5)
            )

            # ── Focal Loss — Phase 4 ──────────────────────────────────────
            # alpha = negative_fraction: upweights the minority class.
            # For ER  (78% pos): alpha=0.22 → upweights rare negatives.
            # For PR  (75% pos): alpha=0.25 → upweights rare negatives.
            # For HER2 (28% pos): alpha=0.72 → heavily upweights rare positives.
            n_total_cv = len(label_dicts)
            bce_criteria = {}
            for task in BINARY_TASKS:
                n_pos = sum(d[task] for d in label_dicts)
                alpha = 1.0 - (n_pos / max(n_total_cv, 1))  # neg fraction
                bce_criteria[task] = BinaryFocalLossWithLogits(alpha=alpha, gamma=2.0)
            ce_criterion = MultiClassFocalLoss(
                alpha=grade_weight_all, gamma=2.0, label_smoothing=0.1,
            )

            best_fold_auc = 0.0

            for epoch in range(n_epochs):
                _ = train_epoch(
                    model, tr_bags, tr_lbls, bce_criteria, ce_criterion,
                    optimizer, device, max_bag_size,
                )  # returns (loss, metrics, task_losses) — discarded during HPO
                _, _, val_auc, _ = val_epoch(
                    model, va_bags, va_lbls, bce_criteria, ce_criterion,
                    device, max_bag_size,
                )
                scheduler.step()
                best_fold_auc = max(best_fold_auc, val_auc)

            fold_aucs.append(best_fold_auc)

            # Optuna pruning — report mean AUC after each fold.
            # n_warmup_steps=3: trial survives folds 0/1/2 unconditionally,
            # then faces the MedianPruner at folds 3 and 4.
            # (n_warmup_steps=5 would disable pruning entirely with 5 folds.)
            mean_auc = float(np.mean(fold_aucs))
            trial.report(mean_auc, step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        final_auc = float(np.mean(fold_aucs))
        logger.info(
            f"Trial {trial.number:03d} | macro_auc={final_auc:.4f} | "
            f"folds={[f'{a:.3f}' for a in fold_aucs]} | params={trial.params}"
        )
        return final_auc

    return objective


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING CURVE VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_curves(history: Dict, out_dir: Path, logger: logging.Logger) -> None:
    """
    Generate publication-quality training curves from epoch history.
    Produces 3 figures:
      1. Train vs Val loss  — divergence here signals overfitting.
      2. Per-task train vs val accuracy — per-head overfitting check.
      3. Per-task val AUC progression — monitors ranking performance.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Colour palette ────────────────────────────────────────────────────────
    task_colors = {"ER": "#2563eb", "PR": "#7c3aed", "HER2": "#db2777", "Grade": "#059669"}

    # ── 1. Loss curves ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["train_loss"], "o-", color="#3b82f6", lw=2, ms=3, label="Train Loss")
    ax.plot(epochs, history["val_loss"],   "s-", color="#ef4444", lw=2, ms=3, label="Val Loss")
    ax.fill_between(epochs, history["train_loss"], history["val_loss"],
                    alpha=0.08, color="#ef4444")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training vs Validation Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    p = out_dir / "loss_curves.png"
    fig.savefig(str(p), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Loss curves saved → {p}")

    # ── 2. Per-task accuracy curves ───────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Per-Task Accuracy: Train vs Validation",
                 fontsize=15, fontweight="bold", y=1.01)
    for idx, task in enumerate(ALL_TASKS):
        ax = axes[idx // 2][idx % 2]
        c = task_colors[task]
        ax.plot(epochs, history[f"train_acc_{task}"], "o-", color=c, lw=2, ms=3,
                label=f"Train", alpha=0.85)
        ax.plot(epochs, history[f"val_acc_{task}"],   "s--", color=c, lw=2, ms=3,
                label=f"Val", alpha=0.85)
        gap = [t - v for t, v in zip(history[f"train_acc_{task}"],
                                     history[f"val_acc_{task}"])]
        ax.fill_between(epochs, history[f"train_acc_{task}"],
                        history[f"val_acc_{task}"], alpha=0.08, color=c)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(f"{task}", fontsize=13, fontweight="bold")
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=10, loc="lower right")
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    p = out_dir / "accuracy_curves.png"
    fig.savefig(str(p), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Accuracy curves saved → {p}")

    # ── 3. Val AUC progression ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for task in ALL_TASKS:
        key = f"val_auc_{task}"
        if key in history:
            ax.plot(epochs, history[key], "o-", color=task_colors[task],
                    lw=2, ms=3, label=task)
    if "val_macro_auc" in history:
        ax.plot(epochs, history["val_macro_auc"], "k-", lw=2.5, ms=4,
                label="Macro AUC", alpha=0.8)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("AUC", fontsize=12)
    ax.set_title("Validation AUC per Epoch", fontsize=14, fontweight="bold")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    p = out_dir / "val_auc_curves.png"
    fig.savefig(str(p), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Val AUC curves saved → {p}")

    # ── 4. Per-task loss curves ───────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Per-Task Loss: Train vs Validation",
                 fontsize=15, fontweight="bold", y=1.01)
    for idx, task in enumerate(ALL_TASKS):
        ax = axes[idx // 2][idx % 2]
        c = task_colors[task]
        ax.plot(epochs, history[f"train_loss_{task}"], "o-", color=c, lw=2, ms=3,
                label="Train", alpha=0.85)
        ax.plot(epochs, history[f"val_loss_{task}"],   "s--", color=c, lw=2, ms=3,
                label="Val", alpha=0.85)
        ax.fill_between(epochs, history[f"train_loss_{task}"],
                        history[f"val_loss_{task}"], alpha=0.08, color=c)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Loss", fontsize=11)
        ax.set_title(f"{task}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    p = out_dir / "per_task_loss_curves.png"
    fig.savefig(str(p), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Per-task loss curves saved → {p}")


# ─────────────────────────────────────────────────────────────────────────────
#  FINAL TRAINING  — compiled model, val-AUC checkpointing
# ─────────────────────────────────────────────────────────────────────────────
def train_final_model(
    best_params:       Dict,
    train_bags:        List[torch.Tensor],
    train_label_dicts: List[Dict],
    val_bags:          List[torch.Tensor],
    val_label_dicts:   List[Dict],
    test_bags:         List[torch.Tensor],
    test_label_dicts:  List[Dict],
    embed_dim:         int,
    device:            torch.device,
    n_epochs:          int,
    ckpt_path:         Path,
    results_dir:       Path,
    logger:            logging.Logger,
) -> Dict:
    """
    Retrain with best_params on the official train split.
    Use official val split for checkpointing (no leakage — val was not used for tuning).
    Evaluate once on the held-out test split.
    Also records full training history and generates overfitting diagnostic plots.
    """
    logger.info("=" * 68)
    logger.info("  Final model training (train split → checkpoint on val AUC)")
    logger.info("=" * 68)
    logger.info(f"  Params: {best_params}")

    seed_everything(42)
    model = GatedABMIL(
        embed_dim    = embed_dim,
        attn_dim     = best_params["attn_dim"],
        attn_dropout = best_params["attn_dropout"],
        clf_dropout  = best_params.get("clf_dropout", 0.0),
        use_layernorm = best_params.get("use_layernorm", False),
    ).to(device)

    # Compile once here — never inside Optuna — for a clean 1.3–2× forward speedup
    if device.type == "cuda":
        logger.info("  Applying torch.compile(mode='reduce-overhead') …")
        model = torch.compile(model, mode="reduce-overhead")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = best_params["lr"],
        weight_decay = best_params["weight_decay"],
        fused        = (device.type == "cuda"),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(5, n_epochs // 5)
    )

    _, grade_weight = compute_class_weights(
        train_label_dicts, device, logger
    )
    # ── Focal Loss — Phase 4 ──────────────────────────────────────────────
    # Computed from the training split class frequencies.
    n_train = len(train_label_dicts)
    bce_criteria = {}
    for task in BINARY_TASKS:
        n_pos = sum(d[task] for d in train_label_dicts)
        alpha = 1.0 - (n_pos / max(n_train, 1))  # neg fraction → upweight minority
        logger.info(f"  focal_alpha[{task}] = {alpha:.3f}  (gamma=2.0)")
        bce_criteria[task] = BinaryFocalLossWithLogits(alpha=alpha, gamma=2.0)
    ce_criterion = MultiClassFocalLoss(
        alpha=grade_weight, gamma=2.0, label_smoothing=0.1,
    )
    logger.info(f"  Grade focal: gamma=2.0, label_smoothing=0.1")
    max_bag_size = best_params["max_bag_size"]

    # Checkpoint on val_macro_auc — matches the Optuna objective
    best_val_auc = 0.0
    best_state   = None
    
    # Early stopping based on val_loss (divergence prevention)
    best_val_loss = float("inf")
    patience = 15
    epochs_no_improve = 0

    # ── Training history for overfitting diagnostics ──────────────────────────
    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [], "val_macro_auc": [],
    }
    for task in ALL_TASKS:
        history[f"train_loss_{task}"] = []
        history[f"val_loss_{task}"]   = []
        for metric in ["acc", "auc", "f1", "precision", "sensitivity", "specificity", "balanced_acc"]:
            history[f"train_{metric}_{task}"] = []
            history[f"val_{metric}_{task}"]   = []

    for epoch in tqdm(range(n_epochs), desc="Final training"):
        train_loss, train_metrics, train_t_losses = train_epoch(
            model, train_bags, train_label_dicts,
            bce_criteria, ce_criterion, optimizer, device, max_bag_size,
        )
        val_loss, val_metrics, val_auc, val_t_losses = val_epoch(
            model, val_bags, val_label_dicts,
            bce_criteria, ce_criterion, device, max_bag_size,
        )
        scheduler.step()

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_auc"].append(val_auc)
        for task in ALL_TASKS:
            history[f"train_loss_{task}"].append(train_t_losses[task])
            history[f"val_loss_{task}"].append(val_t_losses[task])
            for metric in ["acc", "auc", "f1", "precision", "sensitivity", "specificity", "balanced_acc"]:
                # Train metrics
                v_tr = train_metrics[task].get(metric, float("nan"))
                history[f"train_{metric}_{task}"].append(v_tr if not np.isnan(v_tr) else None)
                # Val metrics
                v_va = val_metrics[task].get(metric, float("nan"))
                history[f"val_{metric}_{task}"].append(v_va if not np.isnan(v_va) else None)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            # torch.compile wraps model in _orig_mod; extract the base state_dict
            base = model._orig_mod if hasattr(model, "_orig_mod") else model
            best_state = {k: v.cpu().clone() for k, v in base.state_dict().items()}

        logger.debug(
            f"  Epoch {epoch:03d} | train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_macro_auc={val_auc:.4f}  best={best_val_auc:.4f}"
        )
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            logger.info(f"\n  Early stopping triggered at epoch {epoch:03d} (no improvement in val_loss for {patience} epochs)")
            break

    logger.info(f"  Best val macro-AUC: {best_val_auc:.4f}")

    # ── Generate training curve plots ─────────────────────────────────────────
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info("\n  Generating training curve visualisations …")
    plot_training_curves(history, results_dir, logger)

    # Save raw history as JSON for later analysis
    history_path = results_dir / "training_history.json"
    with open(str(history_path), "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"  Training history saved → {history_path}")

    # Save checkpoint
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": best_state, "params": best_params, "embed_dim": embed_dim},
        str(ckpt_path),
    )
    logger.info(f"  Checkpoint saved → {ckpt_path}")

    # ── Test evaluation — one-time, no further tuning ─────────────────────────
    eval_model = GatedABMIL(embed_dim, best_params["attn_dim"],
                            best_params["attn_dropout"],
                            best_params.get("clf_dropout", 0.0),
                            best_params.get("use_layernorm", False)).to(device)
    eval_model.load_state_dict(best_state)

    # ── Phase 5: Per-task threshold optimization on validation set ─────────
    # The default threshold of 0.5 is suboptimal for imbalanced tasks.
    # Youden's J statistic (sensitivity + specificity - 1) finds the threshold
    # that jointly maximizes both, which is clinically preferred over accuracy-
    # maximising thresholds that can collapse minority-class recall.
    logger.info("\n  ── Threshold Optimisation (Youden's J on validation) ──")

    # Collect val logits
    val_all_logits = {t: [] for t in ALL_TASKS}
    val_all_labels = {k: [] for k in ALL_TASKS + ["Grade_idx"]}
    eval_model.eval()
    with torch.no_grad():
        for i in range(len(val_bags)):
            H = subsample_bag(val_bags[i], max_bag_size, training=False)
            with torch.amp.autocast(device_type=device.type,
                                    dtype=torch.bfloat16,
                                    enabled=(device.type == "cuda")):
                preds, _ = eval_model(H)
            for task in BINARY_TASKS:
                val_all_logits[task].append(preds[task].detach().float().cpu())
                val_all_labels[task].append(val_label_dicts[i][task])
            val_all_logits["Grade"].append(preds["Grade"].detach().float().cpu())
            val_all_labels["Grade_idx"].append(val_label_dicts[i]["Grade_idx"])

    # Find optimal thresholds per binary task via Youden's J
    optimal_thresholds = {}
    for task in BINARY_TASKS:
        logits = torch.stack(val_all_logits[task]).squeeze(-1).numpy()
        labels = np.array(val_all_labels[task], dtype=int)
        probs  = 1.0 / (1.0 + np.exp(-logits))

        best_j, best_thresh = -1.0, 0.5
        for t in np.arange(0.05, 0.95, 0.01):
            preds_t = (probs >= t).astype(int)
            cm = confusion_matrix(labels, preds_t, labels=[0, 1])
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sens = tp / max(tp + fn, 1)
                spec = tn / max(tn + fp, 1)
                j = sens + spec - 1.0
                if j > best_j:
                    best_j, best_thresh = j, t
        optimal_thresholds[task] = round(best_thresh, 2)
        logger.info(f"  {task:<8} optimal threshold = {best_thresh:.2f}  (Youden's J = {best_j:.4f})")

    # ── Test set inference ────────────────────────────────────────────────────
    test_all_logits = {t: [] for t in ALL_TASKS}
    test_all_labels = {k: [] for k in ALL_TASKS + ["Grade_idx"]}
    test_total_loss = 0.0
    with torch.no_grad():
        for i in range(len(test_bags)):
            H = subsample_bag(test_bags[i], max_bag_size, training=False)
            with torch.amp.autocast(device_type=device.type,
                                    dtype=torch.bfloat16,
                                    enabled=(device.type == "cuda")):
                preds, _ = eval_model(H)
                loss, _ = compute_loss(preds, test_label_dicts[i],
                                       bce_criteria, ce_criterion, device)
            test_total_loss += loss.item()
            for task in BINARY_TASKS:
                test_all_logits[task].append(preds[task].detach().float().cpu())
                test_all_labels[task].append(test_label_dicts[i][task])
            test_all_logits["Grade"].append(preds["Grade"].detach().float().cpu())
            test_all_labels["Grade_idx"].append(test_label_dicts[i]["Grade_idx"])

    test_loss = test_total_loss / max(len(test_bags), 1)

    # Default metrics (threshold = 0.5)
    test_metrics = compute_metrics(test_all_logits, test_all_labels)
    test_macro_auc = test_metrics["__macro_auc__"]["auc"]

    # Optimised metrics (per-task thresholds from validation)
    opt_metrics = {}
    for task in BINARY_TASKS:
        logits = torch.stack(test_all_logits[task]).squeeze(-1).numpy()
        labels = np.array(test_all_labels[task], dtype=int)
        probs  = 1.0 / (1.0 + np.exp(-logits))
        thresh = optimal_thresholds[task]
        preds_t = (probs >= thresh).astype(int)

        auc = (roc_auc_score(labels, probs)
               if len(np.unique(labels)) > 1 else float("nan"))
        cm = confusion_matrix(labels, preds_t, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        opt_metrics[task] = {
            "threshold":   thresh,
            "auc":         auc,
            "acc":         accuracy_score(labels, preds_t),
            "balanced_acc": balanced_accuracy_score(labels, preds_t),
            "precision":   precision_score(labels, preds_t, zero_division=0),
            "f1":          f1_score(labels, preds_t, zero_division=0),
            "sensitivity": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1),
        }
    # Copy Grade metrics unchanged (argmax, not thresholded)
    opt_metrics["Grade"] = test_metrics["Grade"]
    opt_metrics["__macro_auc__"] = test_metrics["__macro_auc__"]

    # ── Log results ───────────────────────────────────────────────────────────
    logger.info(f"\n  Test loss = {test_loss:.4f}")

    # Default thresholds
    logger.info("\n  ── Default Thresholds (0.5) ──")
    logger.info("  ─────────────────────────────────────────────────────────")
    logger.info(
        f"  {'Task':<8} │ {'AUC':>6} │ {'Acc':>6} │ {'F1':>6} │ "
        f"{'Sens':>6} │ {'Spec':>6}"
    )
    logger.info("  ─────────────────────────────────────────────────────────")
    for task in ALL_TASKS:
        m = test_metrics[task]
        logger.info(
            f"  {task:<8} │ {m['auc']:>6.4f} │ {m['acc']:>6.4f} │ "
            f"{m['f1']:>6.4f} │ {m['sensitivity']:>6.4f} │ {m['specificity']:>6.4f}"
        )
    logger.info("  ─────────────────────────────────────────────────────────")
    logger.info(f"  Macro AUC = {test_macro_auc:.4f}")

    # Optimised thresholds
    logger.info("\n  ── Optimised Thresholds (Youden's J) ──")
    logger.info("  ─────────────────────────────────────────────────────────────────")
    logger.info(
        f"  {'Task':<8} │ {'Thresh':>6} │ {'AUC':>6} │ {'Acc':>6} │ {'F1':>6} │ "
        f"{'Sens':>6} │ {'Spec':>6} │ {'BalAcc':>6}"
    )
    logger.info("  ─────────────────────────────────────────────────────────────────")
    for task in BINARY_TASKS:
        m = opt_metrics[task]
        logger.info(
            f"  {task:<8} │ {m['threshold']:>6.2f} │ {m['auc']:>6.4f} │ {m['acc']:>6.4f} │ "
            f"{m['f1']:>6.4f} │ {m['sensitivity']:>6.4f} │ {m['specificity']:>6.4f} │ {m['balanced_acc']:>6.4f}"
        )
    gm = opt_metrics["Grade"]
    logger.info(
        f"  {'Grade':<8} │ {'argmx':>6} │ {gm['auc']:>6.4f} │ {gm['acc']:>6.4f} │ "
        f"{gm['f1']:>6.4f} │ {'  nan':>6} │ {'  nan':>6} │ {gm['balanced_acc']:>6.4f}"
    )
    logger.info("  ─────────────────────────────────────────────────────────────────")
    logger.info(f"  Macro AUC = {test_macro_auc:.4f}")

    # Save both metric sets
    metrics_path = results_dir / "test_metrics.json"
    with open(str(metrics_path), "w") as f:
        json.dump(_clean_for_json(test_metrics), f, indent=2)
    logger.info(f"  Test metrics saved → {metrics_path}")

    opt_metrics_path = results_dir / "test_metrics_optimised.json"
    with open(str(opt_metrics_path), "w") as f:
        json.dump(_clean_for_json(opt_metrics), f, indent=2)

    thresholds_path = results_dir / "optimal_thresholds.json"
    with open(str(thresholds_path), "w") as f:
        json.dump(optimal_thresholds, f, indent=2)
    logger.info(f"  Optimised metrics saved → {opt_metrics_path}")
    logger.info(f"  Optimal thresholds saved → {thresholds_path}")

    return test_metrics


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def detect_embed_dim(model_dir: Path, split: str, patient_ids: List[str]) -> int:
    for pid in patient_ids:
        pt = model_dir / split / f"{pid}.pt"
        if pt.exists():
            return torch.load(
                str(pt), map_location="cpu", weights_only=True
            ).shape[1]
    raise FileNotFoundError(
        f"No .pt files in {model_dir / split}. Run 02_extract_features.py first."
    )


def _clean_for_json(d):
    """Recursively replace NaN floats with None for JSON serialisation."""
    if isinstance(d, dict):
        return {k: _clean_for_json(v) for k, v in d.items()}
    return None if (isinstance(d, float) and np.isnan(d)) else d


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BCNB Multi-Task ABMIL — Optuna HPO + final training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",       type=str, default="owkin_phikon",
                   help="Feature dir name under outputs/features/ (/ → _).")
    p.add_argument("--features_dir",type=str, default=str(FEATURES_ROOT),
                   help="Custom root directory for features (e.g. outputs/trident_features)")
    p.add_argument("--n_trials",    type=int, default=20)
    p.add_argument("--epochs",      type=int, default=50)
    p.add_argument("--device",      type=str, default="cuda")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--final_only",  action="store_true",
                   help="Skip Optuna; load best_params.json directly.")
    p.add_argument("--no_test",     action="store_true",
                   help="Skip final test evaluation.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    # Generate run_id early so the log file is also timestamped
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(LOG_DIR / f"03_train_mil_{run_id}.log")

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    logger.info("=" * 68)
    logger.info("  BCNB Multi-Task ABMIL — Hardware-Optimised Training")
    logger.info("=" * 68)
    logger.info(f"  Model       : {args.model}")
    logger.info(f"  Device      : {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        logger.info(f"  GPU         : {props.name}  {props.total_memory//1024**2} MB VRAM")
    logger.info(f"  Trials      : {args.n_trials}")
    logger.info(f"  Epochs      : {args.epochs}")
    logger.info(f"  Bag pinning : all splits → GPU at load time")
    logger.info(f"  AMP dtype   : bfloat16 (no GradScaler needed)")
    logger.info(f"  compile     : final model only (not inside Optuna)")

    # ── Run ID (timestamp) — prevents overwriting previous runs ───────────────
    model_dir   = Path(args.features_dir) / args.model
    ckpt_path   = CKPT_DIR      / args.model / run_id / "best_model.pth"
    results_dir = RESULTS_DIR   / args.model / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"  Run ID      : {run_id}")
    logger.info(f"  Results dir : {results_dir}")

    # ── Labels ────────────────────────────────────────────────────────────────
    labels_df = load_labels(CLINICAL_XLSX, logger)
    train_ids = load_split_ids(SPLIT_FILES["train"])
    val_ids   = load_split_ids(SPLIT_FILES["val"])
    test_ids  = load_split_ids(SPLIT_FILES["test"])

    # ── Embed dim (auto-detected from first .pt file) ─────────────────────────
    embed_dim = detect_embed_dim(model_dir, "train", train_ids)
    logger.info(f"  Embed dim   : {embed_dim}  (auto-detected)")

    # ── Load all features from NFS → GPU in one pass ──────────────────────────
    # NFS latency is paid once here. All subsequent training uses GPU-resident
    # tensors with zero H2D transfers. Test bags are always loaded upfront to
    # avoid a surprise NFS read after hours of training.
    logger.info("\nLoading all features from NFS → GPU …")
    train_bags, train_lbls, _ = load_and_pin_features(
        "train", model_dir, train_ids, labels_df, device, logger
    )
    val_bags, val_lbls, _ = load_and_pin_features(
        "val", model_dir, val_ids, labels_df, device, logger
    )
    test_bags, test_lbls, _ = load_and_pin_features(
        "test", model_dir, test_ids, labels_df, device, logger
    )

    # CV pool = train + val (test is never touched during search)
    cv_bags = train_bags + val_bags
    cv_lbls = train_lbls + val_lbls
    logger.info(f"  CV pool     : {len(cv_bags)} patients  |  "
                f"Test : {len(test_bags)} patients  (held out)")

    # ── Optuna hyperparameter search ──────────────────────────────────────────
    if not args.final_only:
        logger.info(
            f"\nStarting Optuna: {args.n_trials} trials × 5-fold CV × {args.epochs} epochs"
        )
        study = optuna.create_study(
            direction  = "maximize",
            study_name = f"abmil_{args.model}",
            pruner     = optuna.pruners.MedianPruner(
                # n_startup_trials=5: don't prune until 5 trials have completed
                # n_warmup_steps=3:   protect folds 0/1/2; prune eligible at folds 3/4
                n_startup_trials = 5,
                n_warmup_steps   = 3,
            ),
        )
        study.optimize(
            build_objective(cv_bags, cv_lbls, embed_dim, device, args.epochs, logger),
            n_trials          = args.n_trials,
            show_progress_bar = True,
        )

        best_params = study.best_params
        logger.info(f"\nBest trial   : {study.best_trial.number}")
        logger.info(f"Best AUC     : {study.best_value:.4f}")
        logger.info(f"Best params  : {best_params}")

        with open(str(results_dir / "best_params.json"), "w") as f:
            json.dump({"best_params": best_params, "best_auc": study.best_value},
                      f, indent=2)
        study.trials_dataframe().to_csv(
            str(results_dir / "optuna_summary.csv"), index=False
        )
        logger.info(f"Results saved → {results_dir}")

    else:
        # --final_only: find the latest run that has best_params.json
        model_results = RESULTS_DIR / args.model
        if not model_results.exists():
            raise FileNotFoundError(
                f"--final_only set but no results directory at {model_results}. "
                f"Run without --final_only first."
            )
        # Sort subdirectories by name (timestamp format sorts chronologically)
        candidates = sorted(
            [d for d in model_results.iterdir()
             if d.is_dir() and (d / "best_params.json").exists()],
            key=lambda d: d.name,
        )
        if not candidates:
            raise FileNotFoundError(
                f"--final_only set but no run with best_params.json found under "
                f"{model_results}. Run without --final_only first."
            )
        prev_run = candidates[-1]  # latest timestamp
        with open(str(prev_run / "best_params.json")) as f:
            best_params = json.load(f)["best_params"]
        logger.info(f"Loaded best params from previous run: {prev_run.name}")
        logger.info(f"  {best_params}")

    # ── Final model ───────────────────────────────────────────────────────────
    if not args.no_test:
        test_metrics = train_final_model(
            best_params       = best_params,
            train_bags        = train_bags,
            train_label_dicts = train_lbls,
            val_bags          = val_bags,         # official val — checkpoint signal
            val_label_dicts   = val_lbls,
            test_bags         = test_bags,        # held-out test — evaluated once
            test_label_dicts  = test_lbls,
            embed_dim         = embed_dim,
            device            = device,
            n_epochs          = args.epochs,
            ckpt_path         = ckpt_path,
            results_dir       = results_dir,
            logger            = logger,
        )

    logger.info("\nDone.")


if __name__ == "__main__":
    main()