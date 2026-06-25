#!/usr/bin/env python3
"""
05_train_single_task.py
=======================
Train one independent Gated ABMIL model per biomarker task (ER, PR, HER2, Grade).
Uses 5-Fold Stratified CV with moderate regularisation.
After training, runs patient-level ensemble evaluation on the held-out test set.
"""
import argparse, json, logging, random, warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (roc_auc_score, roc_curve, auc, accuracy_score,
                              f1_score, balanced_accuracy_score, confusion_matrix)
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
warnings.filterwarnings("ignore", category=UserWarning)

DATASET_ROOT  = Path("/home/cgntlab5/bcnb/BCNB dataset")
SPLIT_DIR     = DATASET_ROOT / "dataset-splitting"
CLINICAL_XLSX = DATASET_ROOT / "patient-clinical-data.xlsx"
RESULTS_DIR   = DATASET_ROOT / "outputs" / "results"
SPLIT_FILES   = {s: SPLIT_DIR / f"{s}_id.txt" for s in ["train", "val", "test"]}
BINARY_TASKS  = ["ER", "PR", "HER2"]
ALL_TASKS     = BINARY_TASKS + ["Grade"]
N_GRADE_CLS   = 3

# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("single_task")
    if logger.handlers: return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
    for h in [logging.FileHandler(log_path, mode="w"), logging.StreamHandler()]:
        h.setLevel(logging.INFO); h.setFormatter(fmt); logger.addHandler(h)
    return logger

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

# ── Data Loading ──────────────────────────────────────────────────────────────
def load_labels(xlsx, logger):
    df = pd.read_excel(str(xlsx)); df.columns = df.columns.str.strip()
    id_col = next(c for c in df.columns if c.lower().replace(" ", "_") == "patient_id")
    df = df.set_index(id_col); df.index = df.index.astype(str)
    grade_col = next(c for c in df.columns if "grad" in c.lower())
    df = df[BINARY_TASKS + [grade_col]].copy().rename(columns={grade_col: "Histological Grade"})
    for col in BINARY_TASKS:
        df[col] = df[col].astype(str).str.lower().str.strip().map({"positive":1,"negative":0,"1":1,"0":0})
    n0 = len(df); df = df.dropna(subset=BINARY_TASKS + ["Histological Grade"])
    if n0 - len(df): logger.warning(f"Dropped {n0-len(df)} patients with missing labels.")
    df["Grade_idx"] = df["Histological Grade"].astype(int) - 1
    df[BINARY_TASKS] = df[BINARY_TASKS].astype(int)
    logger.info(f"Labels loaded: {len(df)} patients"); return df

def load_split_ids(path):
    with open(path) as f: return [l.strip() for l in f if l.strip()]

def load_and_pin_features(split, model_dir, pids, labels_df, device, logger):
    feat_dir = model_dir / split; bags, lbls, vids = [], [], []
    mf = ml = 0
    for pid in tqdm(pids, desc=f"  Load {split}", leave=False):
        pt = feat_dir / f"{pid}.pt"
        if not pt.exists(): mf += 1; continue
        if pid not in labels_df.index: ml += 1; continue
        feat = torch.load(str(pt), map_location="cpu", weights_only=True)
        if feat.dim() != 2 or feat.shape[0] == 0: continue
        feat = (feat - feat.mean(0, keepdim=True)) / (feat.std(0, unbiased=False, keepdim=True) + 1e-8)
        row = labels_df.loc[pid]
        lbls.append({"ER": float(row["ER"]), "PR": float(row["PR"]),
                      "HER2": float(row["HER2"]), "Grade_idx": int(row["Grade_idx"])})
        bags.append(feat.float().to(device, non_blocking=True)); vids.append(pid)
    logger.info(f"[{split}] {len(bags)}/{len(pids)} bags | miss_feat={mf} miss_lbl={ml}")
    return bags, lbls, vids

# ── Single-Task Model ────────────────────────────────────────────────────────
class SingleTaskABMIL(nn.Module):
    def __init__(self, embed_dim, attn_dim=256, attn_dropout=0.3, clf_dropout=0.3,
                 noise_std=0.05, n_classes=1):
        super().__init__()
        self.noise_std = noise_std; self.n_classes = n_classes
        self.attention_V = nn.Sequential(nn.Linear(embed_dim, attn_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(embed_dim, attn_dim), nn.Sigmoid())
        self.attention_w = nn.Linear(attn_dim, 1, bias=False)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.clf_drop = nn.Dropout(clf_dropout) if clf_dropout > 0 else nn.Identity()
        self.head = nn.Linear(embed_dim, n_classes)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, H):
        if self.training and self.noise_std > 0: H = H + torch.randn_like(H) * self.noise_std
        H = self.attn_drop(H)
        A = torch.softmax(self.attention_w(self.attention_V(H) * self.attention_U(H)), dim=0)
        z = (A * H).sum(dim=0, keepdim=True)
        z = self.norm(z); z = self.clf_drop(z)
        return self.head(z).squeeze(0), A

# ── Focal Losses ──────────────────────────────────────────────────────────────
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__(); self.alpha = alpha; self.gamma = gamma
    def forward(self, x, t):
        bce = F.binary_cross_entropy_with_logits(x, t, reduction='none')
        pt = torch.exp(-bce)
        return ((self.alpha * t + (1-self.alpha)*(1-t)) * (1-pt)**self.gamma * bce).mean()

class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1):
        super().__init__(); self.register_buffer('alpha', alpha)
        self.gamma = gamma; self.ls = label_smoothing
    def forward(self, x, t):
        ce = F.cross_entropy(x, t, reduction='none', label_smoothing=self.ls)
        pt = torch.exp(-ce); loss = (1-pt)**self.gamma * ce
        if self.alpha is not None: loss = self.alpha[t] * loss
        return loss.mean()

def subsample_bag(H, max_size, training):
    N = H.shape[0]
    if N <= max_size: return H
    idx = torch.randperm(N, device=H.device)[:max_size] if training else torch.arange(max_size, device=H.device)
    return H[idx]

# ── Training Loop for One Task ────────────────────────────────────────────────
def train_single_task(task, train_bags, train_lbls, val_bags, val_lbls,
                      embed_dim, device, params, n_epochs, ckpt_path, logger):
    is_binary = task in BINARY_TASKS
    n_classes = 1 if is_binary else N_GRADE_CLS

    model = SingleTaskABMIL(embed_dim, attn_dim=params["attn_dim"],
                            attn_dropout=params["attn_dropout"],
                            clf_dropout=params["clf_dropout"],
                            noise_std=params["noise_std"], n_classes=n_classes).to(device)
    model = torch.compile(model, mode="reduce-overhead")

    if is_binary:
        n_pos = sum(d[task] for d in train_lbls); n_neg = len(train_lbls) - n_pos
        alpha = n_neg / max(n_pos + n_neg, 1)
        criterion = BinaryFocalLoss(alpha=alpha, gamma=2.0)
        logger.info(f"  Focal alpha={alpha:.3f}  pos={int(n_pos)} neg={int(n_neg)}")
    else:
        counts = Counter(d["Grade_idx"] for d in train_lbls); n = len(train_lbls)
        weights = torch.tensor([n/(N_GRADE_CLS*max(counts[c],1)) for c in range(N_GRADE_CLS)],
                               dtype=torch.float32, device=device)
        criterion = MultiClassFocalLoss(alpha=weights, gamma=2.0, label_smoothing=0.1)

    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"],
                            weight_decay=params["weight_decay"], fused=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=max(5, n_epochs//5))
    label_key = task if is_binary else "Grade_idx"

    best_auc = -1.0; patience_cnt = 0; patience = 15
    history = {"train_loss": [], "val_loss": [], "val_auc": [], "train_auc": []}

    for epoch in range(1, n_epochs + 1):
        # Train
        model.train(); t_loss = 0.0; t_logits = []; t_labels = []
        for H, lbl in zip(train_bags, train_lbls):
            H_sub = subsample_bag(H, params["max_bag_size"], training=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                pred, _ = model(H_sub)
                if is_binary:
                    target = torch.tensor([lbl[task]], dtype=torch.float32, device=device)
                else:
                    target = torch.tensor([lbl["Grade_idx"]], dtype=torch.long, device=device)
                    pred = pred.unsqueeze(0)
                loss = criterion(pred, target)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); t_loss += loss.item()
            t_logits.append(pred.detach().float().cpu()); t_labels.append(lbl[label_key])
        sched.step()
        t_loss /= len(train_bags)

        # Val
        model.eval(); v_loss = 0.0; v_logits = []; v_labels = []
        with torch.no_grad():
            for H, lbl in zip(val_bags, val_lbls):
                H_sub = subsample_bag(H, params["max_bag_size"], training=False)
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                    pred, _ = model(H_sub)
                    if is_binary:
                        target = torch.tensor([lbl[task]], dtype=torch.float32, device=device)
                    else:
                        target = torch.tensor([lbl["Grade_idx"]], dtype=torch.long, device=device)
                        pred = pred.unsqueeze(0)
                    loss = criterion(pred, target)
                v_loss += loss.item()
                v_logits.append(pred.detach().float().cpu()); v_labels.append(lbl[label_key])
        v_loss /= len(val_bags)

        # Compute AUC
        def calc_auc(logits_list, labels_list):
            labels_np = np.array(labels_list, dtype=int)
            if len(np.unique(labels_np)) < 2: return 0.5
            if is_binary:
                scores = torch.sigmoid(torch.stack(logits_list).squeeze(-1)).numpy()
                return roc_auc_score(labels_np, scores)
            else:
                logits_np = torch.stack(logits_list).squeeze(1).numpy()
                probs = np.exp(logits_np - logits_np.max(1, keepdims=True))
                probs = probs / probs.sum(1, keepdims=True)
                return roc_auc_score(labels_np, probs, multi_class="ovr", average="macro")

        t_auc = calc_auc(t_logits, t_labels); v_auc = calc_auc(v_logits, v_labels)
        history["train_loss"].append(t_loss); history["val_loss"].append(v_loss)
        history["train_auc"].append(t_auc); history["val_auc"].append(v_auc)

        if v_auc > best_auc:
            best_auc = v_auc; patience_cnt = 0
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "params": params,
                        "embed_dim": embed_dim, "task": task, "n_classes": n_classes,
                        "best_val_auc": best_auc}, ckpt_path)
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                logger.info(f"  Early stop at epoch {epoch} (best AUC={best_auc:.4f})")
                break

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"  E{epoch:03d}  tL={t_loss:.4f} vL={v_loss:.4f} tAUC={t_auc:.4f} vAUC={v_auc:.4f}")

    logger.info(f"  Best val AUC: {best_auc:.4f}")
    return history, best_auc

# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_loss_curves(history, save_path, task):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    e = np.arange(1, len(history["train_loss"]) + 1)
    ax1.plot(e, history["train_loss"], "o-", color="#3b82f6", ms=2, label="Train")
    ax1.plot(e, history["val_loss"], "s-", color="#ef4444", ms=2, label="Val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title(f"{task} — Loss")
    ax1.legend(); ax1.grid(True, alpha=0.2)
    ax2.plot(e, history["train_auc"], "o-", color="#3b82f6", ms=2, label="Train AUC")
    ax2.plot(e, history["val_auc"], "s-", color="#ef4444", ms=2, label="Val AUC")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("AUC"); ax2.set_title(f"{task} — AUC")
    ax2.legend(); ax2.grid(True, alpha=0.2)
    plt.tight_layout(); plt.savefig(save_path, dpi=200); plt.close()

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="phikon_patch256_mag20")
    parser.add_argument("--features_dir", default="outputs/trident_features")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = RESULTS_DIR / f"{args.model}_single_task" / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(results_dir / "train.log")

    logger.info("=" * 60)
    logger.info("  Single-Task ABMIL Training")
    logger.info("=" * 60)

    model_dir = DATASET_ROOT / args.features_dir / args.model
    labels_df = load_labels(CLINICAL_XLSX, logger)
    train_ids = load_split_ids(SPLIT_FILES["train"])
    val_ids = load_split_ids(SPLIT_FILES["val"])
    test_ids = load_split_ids(SPLIT_FILES["test"])

    embed_dim = torch.load(str(next((model_dir / "train").glob("*.pt"))),
                           map_location="cpu", weights_only=True).shape[1]
    logger.info(f"  Embed dim: {embed_dim}")

    logger.info("\nLoading features → GPU …")
    tr_bags, tr_lbls, _ = load_and_pin_features("train", model_dir, train_ids, labels_df, device, logger)
    va_bags, va_lbls, _ = load_and_pin_features("val", model_dir, val_ids, labels_df, device, logger)
    te_bags, te_lbls, _ = load_and_pin_features("test", model_dir, test_ids, labels_df, device, logger)

    cv_bags = tr_bags + va_bags; cv_lbls = tr_lbls + va_lbls
    logger.info(f"  CV pool: {len(cv_bags)} | Test: {len(te_bags)}")

    params = {"lr": 2e-4, "weight_decay": 5e-3, "attn_dim": 256,
              "attn_dropout": 0.3, "clf_dropout": 0.3, "max_bag_size": 64, "noise_std": 0.05}

    final_results = {}

    for task in ALL_TASKS:
        logger.info(f"\n{'='*60}\n  TASK: {task}\n{'='*60}")
        task_dir = results_dir / task; task_dir.mkdir(exist_ok=True)

        if task in BINARY_TASKS:
            strat_keys = np.array([int(d[task]) for d in cv_lbls])
        else:
            strat_keys = np.array([d["Grade_idx"] for d in cv_lbls])

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        fold_aucs = []

        for fold, (tr_idx, va_idx) in enumerate(skf.split(np.arange(len(cv_bags)), strat_keys)):
            logger.info(f"\n  ── Fold {fold+1}/5 ──")
            f_tr_bags = [cv_bags[i] for i in tr_idx]; f_tr_lbls = [cv_lbls[i] for i in tr_idx]
            f_va_bags = [cv_bags[i] for i in va_idx]; f_va_lbls = [cv_lbls[i] for i in va_idx]

            hist, best_auc = train_single_task(
                task, f_tr_bags, f_tr_lbls, f_va_bags, f_va_lbls,
                embed_dim, device, params, args.epochs,
                task_dir / f"fold_{fold+1}_best.pth", logger
            )
            fold_aucs.append(best_auc)
            plot_loss_curves(hist, task_dir / f"fold_{fold+1}_curves.png", f"{task} Fold {fold+1}")
            with open(task_dir / f"fold_{fold+1}_history.json", "w") as f:
                json.dump(hist, f)

        logger.info(f"\n  {task} CV AUCs: {[f'{a:.4f}' for a in fold_aucs]}")
        logger.info(f"  {task} Mean CV AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

        # Patient-level ensemble eval on test set
        logger.info(f"\n  Ensemble eval on test set ({task})...")
        models = []
        for fold in range(1, 6):
            ckpt = torch.load(task_dir / f"fold_{fold}_best.pth", map_location="cpu", weights_only=True)
            n_cls = ckpt["n_classes"]
            m = SingleTaskABMIL(embed_dim, attn_dim=params["attn_dim"], attn_dropout=0.0,
                                clf_dropout=0.0, noise_std=0.0, n_classes=n_cls).to(device)
            m.load_state_dict(ckpt["state_dict"]); m.eval(); models.append(m)

        is_binary = task in BINARY_TASKS
        y_true, y_score = [], []
        with torch.no_grad():
            for H, lbl in te_bags, te_lbls:
                pass  # placeholder
        # Proper loop
        y_true, y_score = [], []
        with torch.no_grad():
            for H, lbl in zip(te_bags, te_lbls):
                preds_all = []
                for m in models:
                    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                        pred, _ = m(H)
                    preds_all.append(pred.float().cpu())
                avg_pred = torch.stack(preds_all).mean(0)
                if is_binary:
                    y_score.append(torch.sigmoid(avg_pred).item())
                    y_true.append(int(lbl[task]))
                else:
                    probs = torch.softmax(avg_pred, dim=-1).numpy()
                    y_score.append(probs)
                    y_true.append(lbl["Grade_idx"])

        y_true_np = np.array(y_true, dtype=int)
        if is_binary:
            y_score_np = np.array(y_score)
            test_auc = roc_auc_score(y_true_np, y_score_np) if len(np.unique(y_true_np)) > 1 else 0.5
        else:
            y_score_np = np.array(y_score)
            test_auc = roc_auc_score(y_true_np, y_score_np, multi_class="ovr", average="macro") \
                if len(np.unique(y_true_np)) > 1 else 0.5

        logger.info(f"  ★ {task} Patient-Level Test AUC = {test_auc:.4f}")
        final_results[task] = {"cv_aucs": fold_aucs, "cv_mean": float(np.mean(fold_aucs)),
                               "test_auc": float(test_auc)}

        # Save ROC for binary tasks
        if is_binary:
            fpr, tpr, _ = roc_curve(y_true_np, y_score_np)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(fpr, tpr, lw=2.5, label=f"AUC = {test_auc:.3f}")
            ax.plot([0,1],[0,1],"k--",lw=1); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
            ax.set_title(f"{task} — Patient-Level ROC"); ax.legend(); ax.grid(True, alpha=0.2)
            plt.tight_layout(); plt.savefig(task_dir / "test_roc.png", dpi=200); plt.close()

    # ── Final Summary ─────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"  FINAL SINGLE-TASK RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"  {'Task':<8} │ {'CV AUC':>10} │ {'Test AUC':>10}")
    logger.info(f"  {'─'*38}")
    all_test_aucs = []
    for task in ALL_TASKS:
        r = final_results[task]
        logger.info(f"  {task:<8} │ {r['cv_mean']:>10.4f} │ {r['test_auc']:>10.4f}")
        all_test_aucs.append(r["test_auc"])
    logger.info(f"  {'─'*38}")
    logger.info(f"  {'Macro':<8} │ {'':>10} │ {np.mean(all_test_aucs):>10.4f}")

    with open(results_dir / "final_results.json", "w") as f:
        json.dump(final_results, f, indent=2)
    logger.info(f"\n  Results → {results_dir}")
    logger.info(f"  Done!")

if __name__ == "__main__":
    main()
