# BCNB Dataset — Agent Reference Document

> **Project Goal:** Repurpose the BCNB Early Breast Cancer Core-Needle Biopsy dataset — originally built for Axillary Lymph Node (ALN) metastasis prediction — to build a **multi-target prognostic AI for Invasive Ductal Carcinoma (IDC)** that predicts molecular biomarkers (ER, PR, HER2) and Histological Grade directly from H&E-stained biopsy slides using a Foundation Model + MIL architecture.

---

## 1. Dataset Overview

| Property | Details |
|---|---|
| **Full Name** | BCNB: Early Breast Cancer Core-Needle Biopsy WSI Dataset |
| **Original Paper** | *"Predicting Axillary Lymph Node Metastasis in Early Breast Cancer Using Deep Learning on Primary Tumor Biopsy Slides"* (Xu et al., Frontiers in Oncology, 2021) |
| **Dataset Portal** | https://bupt-ai-cz.github.io/BCNB/ |
| **Baseline Code Repo** | https://github.com/bupt-ai-cz/BALNMP |
| **Total Patients** | **1,058** (WSIs numbered 1–1058, not all contiguous) |
| **Image Format** | High-resolution `.jpg` WSIs (ranging from ~1.5 MB to ~300 MB per slide) |
| **License** | Academic & non-commercial use only. Must cite the original paper. Redistribution prohibited. |

### Citation (Required)
```bibtex
@article{xu2021predicting,
  title={Predicting Axillary Lymph Node Metastasis in Early Breast Cancer Using Deep Learning on Primary Tumor Biopsy Slides},
  author={Xu, Feng and Zhu, Chuang and Tang, Wenqi and Wang, Ying and Zhang, Yu and Li, Jie and Jiang, Hongchuan and Shi, Zhongyue and Liu, Jun and Jin, Mulan},
  journal={Frontiers in Oncology},
  pages={4133},
  year={2021},
  publisher={Frontiers}
}
```

---

## 2. Directory Structure

```
BCNB dataset/                            ← Dataset root
├── README.txt
├── patient-clinical-data.xlsx           # Ground-truth labels (ER, PR, HER2, Grade, ALN, ...)
├── WSIs/                                # Raw WSIs — primary inputs to preprocessing
│   ├── {patient_id}.jpg
│   └── {patient_id}.json               # Polygon annotations (positive tumor regions)
├── dataset-splitting/
│   ├── train_id.txt                     # 629 patient IDs
│   ├── val_id.txt                       # 209 patient IDs
│   └── test_id.txt                      # 217 patient IDs
└── paper_patches/                       # ⚠️ NOT USED — original paper artefact only
```

```
project/                                 ← Working directory (separate from dataset root)
├── preprocessing/                       # BALNMP scripts (cloned from repo)
│   ├── cut_tumor_regions.py
│   └── cut_patches.py
│
├── tumor_regions/                       # Intermediate: annotated bounding-box crops
│   ├── train/
│   │   └── {patient_id}/
│   │       └── {patient_id}_{region_idx}.jpg
│   ├── val/
│   │   └── {patient_id}/
│   │       └── {patient_id}_{region_idx}.jpg
│   └── test/
│       └── {patient_id}/
│           └── {patient_id}_{region_idx}.jpg
│
├── patches/                             # Final 256×256 tiles — model input
│   ├── train/
│   │   └── {patient_id}/
│   │       └── {patient_id}_{region_idx}_{x}_{y}.jpg
│   ├── val/
│   │   └── {patient_id}/
│   │       └── {patient_id}_{region_idx}_{x}_{y}.jpg
│   └── test/
│       └── {patient_id}/
│           └── {patient_id}_{region_idx}_{x}_{y}.jpg
│
├── features/                            # Extracted FM embeddings
│   ├── train/{patient_id}.pt            # shape: [N_patches, embed_dim]
│   ├── val/{patient_id}.pt
│   └── test/{patient_id}.pt
│
├── checkpoints/
│   └── best_model.pth
└── results/
    └── metrics_{split}.json
```

### Dataset Splits Summary

| Split | Patients | File |
|---|---|---|
| Train | 629 | `dataset-splitting/train_id.txt` |
| Validation | 209 | `dataset-splitting/val_id.txt` |
| Test | 217 | `dataset-splitting/test_id.txt` |
| **Total** | **1,055** | *(3 IDs may be absent from official splits)* |

---

## 3. Data Formats

### 3.1 WSI Annotations — `WSIs/{id}.json`

```json
{
  "positive": [
    {
      "name": "Annotation 0",
      "vertices": [[x1, y1], [x2, y2], ...]
    }
  ],
  "negative": []
}
```

- `positive` → tumor-annotated polygon regions (only these are used by BALNMP scripts)
- `negative` → non-tumor regions (typically empty, ignored)
- Coordinates are in native pixel space of the paired `.jpg`

### 3.2 Patch Naming Convention

Produced by the BALNMP pipeline. Each patch name encodes its full provenance:

```
{patient_id}_{region_idx}_{x}_{y}.jpg
```

| Field | Meaning |
|---|---|
| `patient_id` | Integer patient identifier (matches WSI filename and clinical data row) |
| `region_idx` | Index of the annotated positive tumor region within the patient's `.json` |
| `x` | Pixel x-offset of the top-left corner of the patch within the tumor region crop |
| `y` | Pixel y-offset of the top-left corner of the patch within the tumor region crop |

**Example:** `273_2_512_256.jpg` → Patient 273, 3rd tumor region (0-indexed), patch at pixel (512, 256).

The intermediate tumor region crops (Step 1 output) are named:

```
{patient_id}_{region_idx}.jpg
```

### 3.3 Clinical Labels — `patient-clinical-data.xlsx`

Key columns (load with `pandas.read_excel()`):

| Column | Type | Description | Our Target |
|---|---|---|---|
| `ER` | Binary (0/1) | Estrogen Receptor expression | ✅ Biomarker head |
| `PR` | Binary (0/1) | Progesterone Receptor expression | ✅ Biomarker head |
| `HER2` | Binary (0/1) | HER2/neu receptor expression | ✅ Biomarker head |
| `Histological Grade` | Multi-class (1/2/3) | Nottingham Grading System | ✅ Grading head |
| `ALN status` | Multi-class | Axillary LN metastasis (original task) | ℹ️ Reference only |
| `Patient ID` | Integer | Links to WSI filename | Key column |

> **Grade label offset:** Excel stores Grade as 1, 2, 3. Convert to 0, 1, 2 before `CrossEntropyLoss`.

---

## 4. Data Leakage Prevention — CRITICAL

> **Rule: Split patients FIRST, then preprocess per-split into isolated directories.**

The official train/val/test ID lists define the patient-level split. Because all patches for a single patient come from a single WSI, processing by patient ID is inherently leak-free — **as long as each split's outputs live in separate directories**.

### What would cause leakage
- Running preprocessing over all WSIs into one shared directory then splitting patches afterward
- Normalisation statistics (mean/std) computed on val or test patches and applied to training
- Any augmentation decision that mixes information across patients

### Correct workflow

```
Step 0: Read split files
    train_ids = read("dataset-splitting/train_id.txt")   # 629 IDs
    val_ids   = read("dataset-splitting/val_id.txt")     # 209 IDs
    test_ids  = read("dataset-splitting/test_id.txt")    # 217 IDs

Step 1: Partition WSI directory per split
    → filter WSIs/{id}.jpg and WSIs/{id}.json by split membership
    → do NOT mix patients across splits at any stage

Step 2: Run BALNMP Step 1 (cut_tumor_regions.py) separately per split
    → output: tumor_regions/train/, tumor_regions/val/, tumor_regions/test/

Step 3: Run BALNMP Step 2 (cut_patches.py) separately per split
    → output: patches/train/, patches/val/, patches/test/

Step 4: Compute normalisation stats ONLY on patches/train/
    → apply same stats to val/ and test/ (no recalculation)
```

---

## 5. Preprocessing Pipeline — BALNMP Scripts

> **Install:** `pip install numpy opencv-python`
> **Clone:** `git clone https://github.com/bupt-ai-cz/BALNMP.git`

The pipeline has **two sequential stages** per split.

### Stage 1: `cut_tumor_regions.py` — Extract annotated tumor bounding boxes

**What it does:**
1. Reads `{wsi_id}.jpg` and paired `{wsi_id}.json`
2. For each polygon in `positive`, computes its axis-aligned bounding box
3. Crops that rectangle from the WSI
4. Fills pixels outside the polygon with white (255, 255, 255) — masking non-tumor tissue
5. Saves as `{wsi_id}/{wsi_id}_{region_idx}.jpg`

**Source code key logic:**
```python
# From cut_tumor_regions.py (BALNMP)
for i, (ann_points, bbox) in enumerate(zip(annotation_points, bboxes)):
    x, y, width, height = bbox
    tumour_img = wsi_img[y: y + height, x: x + width]
    # Polygon mask: fill outside-polygon pixels with white
    mask_array = np.zeros((height, width), dtype=np.uint8)
    ann_points = ann_points - np.asarray([x, y])   # relative coords
    cv2.fillPoly(mask_array, ann_points, color=255)
    tumour_img[mask_array != 255] = (255, 255, 255) # fill non-tumor
    cv2.imwrite(tumour_save_path, tumour_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
```

**Run per split (example for train):**
```bash
python preprocessing/cut_tumor_regions.py \
    --wsi_dir_path   ./WSIs_train \
    --output_dir_path ./tumor_regions/train
```

> Since `cut_tumor_regions.py` processes all `.jpg` in a flat directory, you need to either:
> - Symlink only the relevant split's WSIs into a temporary directory, or
> - Use the wrapper script below

**Wrapper to run per-split without copying files:**
```python
# run_preprocessing.py
import os, shutil, subprocess

DATASET_ROOT = "/home/cgntlab5/bcnb/BCNB dataset"
WSI_DIR      = os.path.join(DATASET_ROOT, "WSIs")
SPLIT_DIR    = os.path.join(DATASET_ROOT, "dataset-splitting")
OUT_ROOT     = "./project"   # adjust as needed

SPLITS = {
    "train": "train_id.txt",
    "val":   "val_id.txt",
    "test":  "test_id.txt",
}

for split, id_file in SPLITS.items():
    # Read patient IDs for this split
    with open(os.path.join(SPLIT_DIR, id_file)) as f:
        ids = {line.strip() for line in f if line.strip()}

    # Create a temporary symlink directory containing only this split's WSIs
    tmp_wsi = os.path.join(OUT_ROOT, f"_tmp_wsi_{split}")
    os.makedirs(tmp_wsi, exist_ok=True)
    for pid in ids:
        for ext in [".jpg", ".json"]:
            src = os.path.join(WSI_DIR, pid + ext)
            dst = os.path.join(tmp_wsi, pid + ext)
            if os.path.exists(src) and not os.path.exists(dst):
                os.symlink(src, dst)

    # Stage 1: extract tumor regions
    tumor_out = os.path.join(OUT_ROOT, "tumor_regions", split)
    subprocess.run([
        "python", "preprocessing/cut_tumor_regions.py",
        "--wsi_dir_path",    tmp_wsi,
        "--output_dir_path", tumor_out,
    ], check=True)

    # Stage 2: cut patches
    patch_out = os.path.join(OUT_ROOT, "patches", split)
    subprocess.run([
        "python", "preprocessing/cut_patches.py",
        "--tumour_region_dir_path", tumor_out,
        "--size",            "256",
        "--max_blank_ratio", "0.3",
        "--output_dir_path", patch_out,
    ], check=True)

    # Clean up temp symlink dir
    shutil.rmtree(tmp_wsi)
    print(f"[{split}] Done. Patches at: {patch_out}")
```

### Stage 2: `cut_patches.py` — Tile tumor crops into 256×256 patches

**What it does:**
1. Pads each tumor region image to be divisible by `size` (right + bottom, white padding)
2. Slides a non-overlapping `size × size` window
3. Discards patches where `white_pixels / total_pixels >= max_blank_ratio`
4. Saves kept patches as `{patient_id}_{region_idx}_{x}_{y}.jpg`

**Source code key logic:**
```python
# From cut_patches.py (BALNMP)
def is_deprecated(image_array, blank_rate):
    blank_num = np.sum(image_array == (255, 255, 255)) / 3
    height, width = image_array.shape[:2]
    return blank_num / (height * width) >= blank_rate

for y in np.arange(0, height, args.size):
    for x in np.arange(0, width, args.size):
        patch = big_image_array[y:y+args.size, x:x+args.size]
        save_name = '{}_{}_{}.jpg'.format(original_image_name, x, y)
        # original_image_name = "{patient_id}_{region_idx}"
        # → final patch name:   "{patient_id}_{region_idx}_{x}_{y}.jpg"
        if not is_deprecated(patch, args.max_blank_ratio):
            cv2.imwrite(save_path, patch, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
```

**Run per split:**
```bash
python preprocessing/cut_patches.py \
    --tumour_region_dir_path ./tumor_regions/train \
    --size           256 \
    --max_blank_ratio 0.3 \
    --output_dir_path ./patches/train
```

### Recommended Parameters

| Parameter | Value | Rationale |
|---|---|---|
| `--size` | `256` | Matches foundation model input expectations; BALNMP default |
| `--max_blank_ratio` | `0.3` | Discard patches with >30% white — removes padding artefacts |
| JPEG quality | `100` | Lossless save from BALNMP scripts |

---

## 6. Full Pipeline Architecture

```
WSIs/{id}.jpg + WSIs/{id}.json   (raw input)
        │
        │  [SPLIT FIRST — by official train/val/test IDs]
        │
        ▼
[PHASE 1-A] cut_tumor_regions.py  (per split)
        │  → crop annotated positive regions, mask non-tumor with white
        │  → tumor_regions/{split}/{patient_id}/{patient_id}_{region_idx}.jpg
        ▼
[PHASE 1-B] cut_patches.py  (per split)
        │  → tile each region into 256×256, discard >30% white patches
        │  → patches/{split}/{patient_id}/{patient_id}_{region_idx}_{x}_{y}.jpg
        ▼
[PHASE 2] Feature Extraction  (frozen Foundation Model)
        │  → embed each 256×256 patch → dense vector (e.g. 1024-dim)
        │  → features/{split}/{patient_id}.pt   shape: [N_patches, embed_dim]
        │  → stats computed ONLY on features/train/ then applied to all splits
        ▼
[PHASE 3] MIL Aggregation  (ABMIL / TransMIL)
        │  → aggregate bag [N_patches, embed_dim] → slide repr [embed_dim]
        ▼
[PHASE 4] Multi-Task Classification Heads
        ├── ER Head    → Linear → Sigmoid  → BCELoss   (binary)
        ├── PR Head    → Linear → Sigmoid  → BCELoss   (binary)
        ├── HER2 Head  → Linear → Sigmoid  → BCELoss   (binary)
        └── Grade Head → Linear → Softmax  → CELoss    (3-class: 0/1/2)
```

---

## 7. Phase 2: Feature Extraction

**Recommended Foundation Models:**

| Model | Source | Embed Dim | Notes |
|---|---|---|---|
| **UNI** | MahmoodLab/UNI (HuggingFace) | 1024 | Strong general pathology FM |
| **CONCH** | MahmoodLab/CONCH | 512 | Vision-language, breast focus |
| **Phikon** | owkin/phikon | 768 | ViT-B, open-weight |
| **Virchow** | paige-ai/Virchow | 1280 | Large-scale FM |

```python
import torch, os
from torchvision import transforms
from PIL import Image

def extract_and_save(patient_id, patch_dir, out_dir, model, transform, device):
    """Extract FM embeddings for all patches of one patient."""
    patch_paths = sorted([
        os.path.join(patch_dir, f)
        for f in os.listdir(patch_dir) if f.endswith(".jpg")
    ])
    embeddings = []
    model.eval()
    with torch.no_grad():
        for pp in patch_paths:
            img = Image.open(pp).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            feat = model(x).squeeze(0).cpu()   # [embed_dim]
            embeddings.append(feat)
    features = torch.stack(embeddings)          # [N_patches, embed_dim]
    os.makedirs(out_dir, exist_ok=True)
    torch.save(features, os.path.join(out_dir, f"{patient_id}.pt"))

# Run per split: train → val → test
# Compute normalisation only from training features AFTER extraction
```

---

## 8. Phase 3 & 4: MIL + Multi-Task Heads

```python
import torch
import torch.nn as nn

class ABMIL(nn.Module):
    def __init__(self, embed_dim=1024, hidden_dim=256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, H):
        # H: [N_patches, embed_dim]
        A = torch.softmax(self.attention(H), dim=0)  # [N_patches, 1]
        return (A * H).sum(dim=0), A                  # [embed_dim], [N_patches, 1]


class BCNBMultiTaskModel(nn.Module):
    def __init__(self, embed_dim=1024):
        super().__init__()
        self.mil       = ABMIL(embed_dim)
        self.er_head   = nn.Linear(embed_dim, 1)
        self.pr_head   = nn.Linear(embed_dim, 1)
        self.her2_head = nn.Linear(embed_dim, 1)
        self.grade_head = nn.Linear(embed_dim, 3)  # Grade 0/1/2 (converted from 1/2/3)

    def forward(self, H):
        z, attn = self.mil(H)
        return {
            "ER":    torch.sigmoid(self.er_head(z)),
            "PR":    torch.sigmoid(self.pr_head(z)),
            "HER2":  torch.sigmoid(self.her2_head(z)),
            "Grade": self.grade_head(z),          # raw logits for CrossEntropy
        }, attn


# Loss
bce = nn.BCELoss()
ce  = nn.CrossEntropyLoss()

def compute_loss(pred, labels):
    return (
        bce(pred["ER"],   labels["ER"])   +
        bce(pred["PR"],   labels["PR"])   +
        bce(pred["HER2"], labels["HER2"]) +
        ce(pred["Grade"], labels["Grade"])  # labels["Grade"] must be 0-indexed LongTensor
    )
```

---

## 9. Label Loading

```python
import pandas as pd

def load_labels(excel_path, split_id_file):
    df = pd.read_excel(excel_path)
    # Identify patient ID column (adjust column name if needed after inspecting the file)
    df = df.set_index("Patient ID")
    with open(split_id_file) as f:
        ids = [int(line.strip()) for line in f if line.strip()]
    subset = df.loc[df.index.isin(ids)][["ER", "PR", "HER2", "Histological Grade"]].copy()
    # Convert Grade 1/2/3 → 0/1/2 for CrossEntropyLoss
    subset["Grade_idx"] = subset["Histological Grade"] - 1
    return subset

train_labels = load_labels("patient-clinical-data.xlsx", "dataset-splitting/train_id.txt")
val_labels   = load_labels("patient-clinical-data.xlsx", "dataset-splitting/val_id.txt")
test_labels  = load_labels("patient-clinical-data.xlsx", "dataset-splitting/test_id.txt")
```

---

## 10. Training Strategy

| Aspect | Recommendation |
|---|---|
| **Optimizer** | AdamW, lr = 1e-4 |
| **Scheduler** | CosineAnnealingLR or ReduceLROnPlateau |
| **Batch size** | 1 patient per iteration (variable N_patches per bag) |
| **Epochs** | 50–100 with early stopping on val mean-AUROC |
| **Imbalance** | Weighted BCE for ER/PR/HER2; class_weight for Grade |
| **Primary metric** | AUROC per head; macro-averaged across all 4 tasks |
| **FM backbone** | **Frozen** — only MIL + heads are trainable parameters |

---

## 11. Key Practical Notes

1. **Raw WSIs only** — `paper_patches/` is the original paper's artefact and is intentionally ignored. We tile fresh from `WSIs/` using the BALNMP scripts to have full control over parameters.

2. **Leakage prevention** — Patches are produced per-split into isolated `patches/train/`, `patches/val/`, `patches/test/` directories. No patient appears in more than one split (guaranteed by the official split files). Normalisation statistics are computed only from training data.

3. **Variable bag sizes** — Each patient yields a different number of patches N. Use `batch_size=1` in the DataLoader, or a custom `collate_fn` that pads/lists bags.

4. **Feature extraction is expensive but one-time** — Save `.pt` tensors to `features/{split}/`, then only the MIL + heads are iterated during training.

5. **WSI memory footprint** — Slides range from ~1.5 MB to ~300 MB. `cv2.imread` loads the full image into RAM. On machines with <32 GB RAM, process one WSI at a time and explicitly `del wsi_img` + `gc.collect()` after each patient.

6. **JPEG quality 100** — Both BALNMP scripts write patches at JPEG quality 100 (minimal compression). Treat them as lossless for practical purposes.

7. **Polygon coordinate space** — The `.json` vertices are in the native pixel space of the paired `.jpg`. BALNMP's `cut_tumor_regions.py` correctly handles the coordinate remapping when cropping and masking.

8. **License compliance** — Non-commercial only. Do not distribute raw images or patch files. Model weights and aggregated results are permissible derivative works.

---

## 12. References

- **Dataset Paper:** Xu et al. (2021). *Predicting ALN Metastasis Using Deep Learning*. Frontiers in Oncology. https://doi.org/10.3389/fonc.2021.752428
- **BALNMP Repo + Preprocessing Scripts:** https://github.com/bupt-ai-cz/BALNMP/tree/main/preprocessing
  - `cut_tumor_regions.py` — polygon-aware tumor region extraction
  - `cut_patches.py` — blank-filtered 256×256 tiling
- **Dataset Site:** https://bupt-ai-cz.github.io/BCNB/
- **ABMIL:** Ilse et al. (2018). *Attention-based Deep MIL*. ICML.
- **TransMIL:** Shao et al. (2021). *TransMIL*. NeurIPS.
