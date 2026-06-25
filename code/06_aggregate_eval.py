"""
06_aggregate_eval.py
====================
Patient-level ensemble evaluation using ALL patches per patient (no sub-bag splitting).
Loads 5-fold CV models, passes each patient's FULL feature tensor through all 5 models,
averages the predictions, and generates final ROC curves + box plots.
"""
import json
import logging
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, f1_score
from sklearn.metrics import precision_score, balanced_accuracy_score, confusion_matrix
from tqdm import tqdm
import importlib.util

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()

def load_train_cv():
    spec = importlib.util.spec_from_file_location(
        "train_cv", "/home/cgntlab5/bcnb/BCNB dataset/code/04_train_cv.py"
    )
    train_cv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_cv)
    return train_cv

train_cv = load_train_cv()
GatedABMIL = train_cv.GatedABMIL

BINARY_TASKS = ["ER", "PR", "HER2"]
ALL_TASKS    = BINARY_TASKS + ["Grade"]
N_GRADE_CLS  = 3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Path to run dir. If omitted, uses the latest timestamp.")
    parser.add_argument("--model", type=str, default="phikon_patch256_mag20")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        results_root = Path("/home/cgntlab5/bcnb/BCNB dataset/outputs/results") / args.model
        candidates = sorted([d for d in results_root.iterdir() if d.is_dir() and (d / "fold_1").exists()])
        if not candidates:
            raise FileNotFoundError(f"No completed runs found under {results_root}")
        run_dir = candidates[-1]
        logger.info(f"Auto-detected latest run: {run_dir.name}")
    device = torch.device(args.device)

    # ── 1. Aggregate Training History across Folds ────────────────────────────
    logger.info("=" * 68)
    logger.info("  Patient-Level Ensemble Evaluation")
    logger.info("=" * 68)
    logger.info("\nAggregating training history...")
    all_histories = {}
    for fold in range(1, 6):
        hist_file = run_dir / f"fold_{fold}" / "training_history.json"
        if hist_file.exists():
            with open(hist_file) as f:
                all_histories[fold] = json.load(f)

    if all_histories:
        min_len = min(len(h["train_loss"]) for h in all_histories.values())
        t_loss = np.array([h["train_loss"][:min_len] for h in all_histories.values()])
        v_loss = np.array([h["val_loss"][:min_len] for h in all_histories.values()])
        epochs = np.arange(1, min_len + 1)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, t_loss.mean(0), "o-", color="#3b82f6", lw=2, ms=3, label="Train Loss (mean)")
        ax.fill_between(epochs, t_loss.mean(0) - t_loss.std(0),
                        t_loss.mean(0) + t_loss.std(0), alpha=0.15, color="#3b82f6")
        ax.plot(epochs, v_loss.mean(0), "s-", color="#ef4444", lw=2, ms=3, label="Val Loss (mean)")
        ax.fill_between(epochs, v_loss.mean(0) - v_loss.std(0),
                        v_loss.mean(0) + v_loss.std(0), alpha=0.15, color="#ef4444")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("5-Fold CV: Training vs Validation Loss (Mean ± Std)", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(run_dir / "aggregated_loss_curves.png", dpi=200)
        plt.close()
        logger.info(f"  Saved → {run_dir / 'aggregated_loss_curves.png'}")

    # ── 2. Load Test Data ─────────────────────────────────────────────────────
    logger.info("\nLoading test data...")
    labels_df = train_cv.load_labels(train_cv.CLINICAL_XLSX, logger)
    test_ids  = train_cv.load_split_ids(train_cv.SPLIT_FILES["test"])
    model_dir = Path("/home/cgntlab5/bcnb/BCNB dataset/outputs/trident_features/phikon_patch256_mag20")

    test_bags, test_lbls, valid_test_ids = train_cv.load_and_pin_features(
        "test", model_dir, test_ids, labels_df, device, logger
    )

    # ── 3. Load All 5 Fold Models ─────────────────────────────────────────────
    logger.info("\nLoading fold models...")
    models = []
    best_params = None
    for fold in range(1, 6):
        ckpt_path = run_dir / f"fold_{fold}" / "best_model.pth"
        if not ckpt_path.exists():
            logger.warning(f"  Missing: {ckpt_path}")
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        best_params = ckpt["params"]
        embed_dim = ckpt["embed_dim"]

        # Inference mode: no dropout, no noise
        model = GatedABMIL(
            embed_dim=embed_dim,
            attn_dim=best_params["attn_dim"],
            attn_dropout=0.0,
            clf_dropout=0.0,
            use_layernorm=best_params.get("use_layernorm", False),
            noise_std=0.0,
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        models.append(model)

    logger.info(f"  Loaded {len(models)} models for ensembling.")
    if not models:
        logger.error("No models found!")
        return

    # ── 4. Patient-Level Prediction (ALL patches, no sub-bag splitting) ───────
    logger.info("\nRunning patient-level ensemble inference (full bags)...")
    patient_preds = {t: [] for t in ALL_TASKS}
    patient_labels = {t: [] for t in ALL_TASKS}

    with torch.no_grad():
        for H_full, labels in tqdm(zip(test_bags, test_lbls), total=len(test_bags), desc="Evaluating"):
            # Pass the ENTIRE patient bag through each model
            model_probs = {t: [] for t in ALL_TASKS}

            for model in models:
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                    preds, _ = model(H_full)

                for task in BINARY_TASKS:
                    prob = torch.sigmoid(preds[task]).float().item()
                    model_probs[task].append(prob)

                logits_g = preds["Grade"].float().cpu()
                probs_g = torch.softmax(logits_g, dim=-1).numpy().flatten()
                model_probs["Grade"].append(probs_g)

            # Average across 5 models → single patient prediction
            for task in BINARY_TASKS:
                patient_preds[task].append(np.mean(model_probs[task]))
                patient_labels[task].append(labels[task])

            patient_preds["Grade"].append(np.mean(model_probs["Grade"], axis=0))
            patient_labels["Grade"].append(labels["Grade_idx"])

    # ── 5. Compute & Log Final Metrics ────────────────────────────────────────
    logger.info("\n" + "=" * 68)
    logger.info("  FINAL PATIENT-LEVEL RESULTS (5-Model Ensemble)")
    logger.info("=" * 68)

    all_aucs = {}
    logger.info(f"\n  {'Task':<8} │ {'AUC':>6} │ {'Acc':>6} │ {'F1':>6} │ {'Sens':>6} │ {'Spec':>6} │ {'BalAcc':>6}")
    logger.info("  " + "─" * 62)

    for task in BINARY_TASKS:
        y_true = np.array(patient_labels[task], dtype=int)
        y_score = np.array(patient_preds[task])
        y_pred = (y_score >= 0.5).astype(int)

        task_auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")
        all_aucs[task] = task_auc
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)

        logger.info(
            f"  {task:<8} │ {task_auc:>6.4f} │ {accuracy_score(y_true, y_pred):>6.4f} │ "
            f"{f1_score(y_true, y_pred, zero_division=0):>6.4f} │ {sens:>6.4f} │ {spec:>6.4f} │ "
            f"{balanced_accuracy_score(y_true, y_pred):>6.4f}"
        )

    # Grade
    grade_true = np.array(patient_labels["Grade"], dtype=int)
    grade_probs = np.array([p for p in patient_preds["Grade"]])
    grade_pred = grade_probs.argmax(axis=1)
    grade_auc = roc_auc_score(grade_true, grade_probs, multi_class="ovr", average="macro") \
        if len(np.unique(grade_true)) > 1 else float("nan")
    all_aucs["Grade"] = grade_auc
    logger.info(
        f"  {'Grade':<8} │ {grade_auc:>6.4f} │ {accuracy_score(grade_true, grade_pred):>6.4f} │ "
        f"{f1_score(grade_true, grade_pred, average='macro', zero_division=0):>6.4f} │ "
        f"{'  nan':>6} │ {'  nan':>6} │ {balanced_accuracy_score(grade_true, grade_pred):>6.4f}"
    )
    logger.info("  " + "─" * 62)

    valid_aucs = [v for v in all_aucs.values() if not np.isnan(v)]
    macro_auc = np.mean(valid_aucs) if valid_aucs else 0.0
    logger.info(f"  Macro AUC = {macro_auc:.4f}")

    # Save metrics
    metrics_out = {
        task: {"auc": float(all_aucs[task])} for task in ALL_TASKS
    }
    metrics_out["macro_auc"] = float(macro_auc)
    with open(run_dir / "patient_level_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    logger.info(f"\n  Metrics saved → {run_dir / 'patient_level_metrics.json'}")

    # ── 6. ROC Curves ─────────────────────────────────────────────────────────
    colors = {"ER": "#2563eb", "PR": "#7c3aed", "HER2": "#db2777"}
    fig, ax = plt.subplots(figsize=(8, 8))
    for task in BINARY_TASKS:
        y_true = np.array(patient_labels[task], dtype=int)
        y_score = np.array(patient_preds[task])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[task], lw=2.5, label=f"{task} (AUC = {roc_auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("Patient-Level ROC Curves (5-Model Ensemble)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(run_dir / "patient_roc_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  ROC curves → {run_dir / 'patient_roc_curves.png'}")

    # ── 7. Box Plots ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, task in enumerate(BINARY_TASKS):
        ax = axes[i]
        y_true = np.array(patient_labels[task], dtype=int)
        y_score = np.array(patient_preds[task])

        data_neg = y_score[y_true == 0]
        data_pos = y_score[y_true == 1]

        bp = ax.boxplot(
            [data_neg, data_pos],
            labels=["Negative", "Positive"],
            patch_artist=True,
            widths=0.5,
            showfliers=False,
        )
        palette = ["#93c5fd", "#fca5a5"]
        for patch, color in zip(bp["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Overlay individual points
        for j, (data, xpos) in enumerate([(data_neg, 1), (data_pos, 2)]):
            jitter = np.random.normal(0, 0.04, size=len(data))
            ax.scatter(xpos + jitter, data, alpha=0.35, s=15, color="black", zorder=3)

        ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.5)
        ax.set_title(f"{task}  (AUC={all_aucs[task]:.3f})", fontsize=13, fontweight="bold")
        ax.set_ylabel("Predicted Probability", fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.15, axis="y")

    plt.suptitle("Predicted Probability by True Class", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(run_dir / "patient_box_plots.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Box plots → {run_dir / 'patient_box_plots.png'}")

    # ── 8. Per-Task AUC Box Plot across Folds ─────────────────────────────────
    fold_aucs = {t: [] for t in ALL_TASKS}
    for fold in range(1, 6):
        metrics_file = run_dir / f"fold_{fold}" / "test_metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                fm = json.load(f)
            for task in ALL_TASKS:
                if task in fm and fm[task].get("auc") is not None:
                    fold_aucs[task].append(fm[task]["auc"])

    if any(fold_aucs[t] for t in ALL_TASKS):
        fig, ax = plt.subplots(figsize=(8, 5))
        task_colors_list = ["#2563eb", "#7c3aed", "#db2777", "#059669"]
        positions = list(range(len(ALL_TASKS)))
        for i, task in enumerate(ALL_TASKS):
            if fold_aucs[task]:
                bp = ax.boxplot(
                    fold_aucs[task], positions=[i], widths=0.4,
                    patch_artist=True, showfliers=True,
                )
                bp["boxes"][0].set_facecolor(task_colors_list[i])
                bp["boxes"][0].set_alpha(0.6)
                jitter = np.random.normal(0, 0.03, size=len(fold_aucs[task]))
                ax.scatter([i] + jitter, fold_aucs[task], color="black", s=30, zorder=3, alpha=0.7)

        ax.set_xticks(positions)
        ax.set_xticklabels(ALL_TASKS, fontsize=12)
        ax.set_ylabel("AUC", fontsize=12)
        ax.set_title("Per-Fold AUC Distribution by Task", fontsize=14, fontweight="bold")
        ax.set_ylim(0.4, 1.0)
        ax.grid(True, alpha=0.2, axis="y")
        plt.tight_layout()
        fig.savefig(run_dir / "fold_auc_boxplot.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Fold AUC box plot → {run_dir / 'fold_auc_boxplot.png'}")

    logger.info("\n" + "=" * 68)
    logger.info("  All evaluation outputs saved!")
    logger.info("=" * 68)


if __name__ == "__main__":
    main()
