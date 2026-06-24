# BCNB Multi-Target Prognostic Pipeline

This repository contains a modular, production-grade PyTorch pipeline designed to repurpose the **BCNB Early Breast Cancer Core-Needle Biopsy Dataset**. While the original dataset was designed for Axillary Lymph Node (ALN) metastasis prediction, this pipeline builds a **multi-target prognostic AI** for Invasive Ductal Carcinoma (IDC), predicting molecular biomarkers (ER, PR, HER2) and Histological Grade directly from H&E-stained biopsy slides.

The architecture leverages a pre-trained **Pathology Foundation Model** (e.g., Phikon, UNI) for feature extraction, followed by an **Attention-Based Multiple Instance Learning (ABMIL)** framework for patient-level classification.

---

## 🛠️ Pipeline Architecture

The pipeline is organized into clear, sequential stages to ensure efficient processing and prevent data leakage:

### 1. Preprocessing & Tiling (`01_preprocessing.py`)
- **Stage A:** Extracts annotated tumor regions from extremely high-resolution Whole Slide Images (WSIs) using provided `.json` polygon coordinates. Masks out non-tumor tissue.
- **Stage B:** Tiles the extracted tumor crops into `256x256` patches. Implements memory-efficient zero-copy tiling and PyTorch-accelerated blank-patch filtering to discard empty white background.

### 2. Feature Extraction (`02_extract_features.py`)
- Uses a pre-trained, frozen Vision Transformer (ViT) pathology foundation model (e.g., `owkin/phikon` or `MahmoodLab/UNI`).
- Efficiently processes patient patch directories in batches via a PyTorch `DataLoader`.
- Saves extracted features for each patient as a highly compressed `.pt` tensor of shape `[N_patches, embed_dim]`.

### 3. ABMIL Training (Next Step)
- An Attention-Based Multiple Instance Learning model that takes bags of extracted features `[N, embed_dim]` and outputs binary biomarker predictions (ER, PR, HER2) and a 3-class Grade prediction.

---

## 🚀 Setup & Installation

The code requires a specific Conda environment, particularly optimized for large WSIs and PyTorch GPU acceleration.

```bash
# 1. Create and activate the conda environment
conda create -n bcnb_subtyping python=3.10 -y
conda activate bcnb_subtyping

# 2. Install PyTorch (Adjust CUDA version as needed)
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# 3. Install pipeline requirements
pip install -r code/requirements.txt
```

---

## 📊 Dataset & Citation

The dataset utilized is the **BCNB: Early Breast Cancer Core-Needle Biopsy WSI Dataset**. 

If you use this dataset, you must cite the original authors:

> Xu, Feng and Zhu, Chuang and Tang, Wenqi and Wang, Ying and Zhang, Yu and Li, Jie and Jiang, Hongchuan and Shi, Zhongyue and Liu, Jun and Jin, Mulan. **"Predicting Axillary Lymph Node Metastasis in Early Breast Cancer Using Deep Learning on Primary Tumor Biopsy Slides"**, *Frontiers in Oncology* (2021).

### License Terms
This dataset is freely available to academic and non-academic entities strictly for **non-commercial purposes** (academic research, teaching, personal experimentation).
1. You must include the reference to the BCNB Dataset in any work.
2. You may not distribute this dataset or modified WSI versions (derivative works like trained models or patch features are permissible).
3. No commercial usage or licensing of the data.
