"""
02_extract_features.py
======================
Phase 2 of the BCNB pipeline.

Extracts dense feature embeddings from the generated 256x256 patches using a
pre-trained Pathology Foundation Model (e.g., Phikon, UNI, CONCH).

Inputs:
    outputs/patches/{split}/{patient_id}/*.jpg
Outputs:
    outputs/features/{model_name}/{split}/{patient_id}.pt
    (shape: [N_patches, embed_dim])

Usage:
    python code/02_extract_features.py --splits train val test --model owkin/phikon
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────────────────────
DATASET_ROOT = Path("/home/cgntlab5/bcnb/BCNB dataset")
PATCHES_ROOT = DATASET_ROOT / "outputs" / "patches"
OUT_ROOT     = DATASET_ROOT / "outputs" / "features"
LOG_DIR      = DATASET_ROOT / "outputs" / "logs"

# ─────────────────────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────────────────────
def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("extract_features")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

# ─────────────────────────────────────────────────────────────────────────────
#  DATASETDCIS
# ─────────────────────────────────────────────────────────────────────────────
class PatientPatchDataset(Dataset):
    """Loads all patches for a single patient."""
    def __init__(self, patch_paths: List[Path], transform):
        self.patch_paths = patch_paths
        self.transform = transform

    def __len__(self):
        return len(self.patch_paths)

    def __getitem__(self, idx):
        path = self.patch_paths[idx]
        img = Image.open(str(path)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, str(path)

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_foundation_model(model_name: str, device: torch.device):
    logger = logging.getLogger("extract_features")
    logger.info(f"Loading Foundation Model: {model_name}")

    if model_name == "owkin/phikon":
        from transformers import ViTModel
        model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
        model = model.to(device)
        model.eval()
        
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        def forward_fn(x):
            out = model(x)
            return out.last_hidden_state[:, 0, :]
        
        embed_dim = 768

    elif model_name == "MahmoodLab/UNI":
        import timm
        model = timm.create_model(
            "hf-hub:MahmoodLab/UNI", pretrained=True, init_values=1e-5, dynamic_img_size=True
        )
        model = model.to(device)
        model.eval()

        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        def forward_fn(x):
            return model(x)
        
        embed_dim = 1024

    elif model_name == "prov-gigapath/prov-gigapath":
        import timm
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        model = model.to(device)
        model.eval()

        transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
        def forward_fn(x):
            return model(x)
            
        embed_dim = 1528

    else:
        raise ValueError(f"Model {model_name} not implemented.")

    return forward_fn, transform, embed_dim

# ─────────────────────────────────────────────────────────────────────────────
#  EXTRACTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def extract_split(
    split: str,
    model_name: str,
    forward_fn,
    transform,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    overwrite: bool,
    logger: logging.Logger
):
    safe_model_name = model_name.replace("/", "_")
    in_dir = PATCHES_ROOT / split
    out_dir = OUT_ROOT / safe_model_name / split

    if not in_dir.exists():
        logger.warning(f"[{split}] Patch directory missing: {in_dir} - Skipping")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    
    patient_dirs = sorted([d for d in in_dir.iterdir() if d.is_dir()])
    logger.info(f"[{split}] Found {len(patient_dirs)} patients in {in_dir}")

    for pdir in tqdm(patient_dirs, desc=f"Extracting {split}", unit="patient"):
        patient_id = pdir.name
        out_pt = out_dir / f"{patient_id}.pt"
        
        if out_pt.exists() and not overwrite:
            logger.debug(f"[{split}] {patient_id} already extracted, skipping...")
            continue

        patch_paths = list(pdir.glob("*.jpg"))
        if len(patch_paths) == 0:
            logger.warning(f"[{split}] {patient_id} has 0 patches!")
            continue

        dataset = PatientPatchDataset(patch_paths, transform)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False
        )

        sorted_patch_paths = sorted([str(p) for p in patch_paths])
        feat_dict = {}

        with torch.no_grad():
            for batch_imgs, batch_paths in loader:
                batch_imgs = batch_imgs.to(device, non_blocking=True)
                feats = forward_fn(batch_imgs)
                feats = feats.cpu()
                
                for i, p in enumerate(batch_paths):
                    feat_dict[p] = feats[i]

        ordered_feats = [feat_dict[p] for p in sorted_patch_paths]
        final_tensor = torch.stack(ordered_feats)
        
        torch.save(final_tensor, out_pt)
        logger.debug(f"[{split}] {patient_id} saved tensor shape: {final_tensor.shape}")

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Extract pathology foundation model embeddings")
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val", "test"])
    parser.add_argument("--model", type=str, default="owkin/phikon")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging(LOG_DIR / "02_extract_features.log")

    logger.info("=" * 70)
    logger.info("  Phase 2: Foundation Model Feature Extraction")
    logger.info("=" * 70)
    logger.info(f"  Model       : {args.model}")
    logger.info(f"  Batch size  : {args.batch_size}")
    logger.info(f"  Workers     : {args.workers}")
    logger.info(f"  Splits      : {args.splits}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  Device      : {device}")

    forward_fn, transform, embed_dim = load_foundation_model(args.model, device)
    logger.info(f"  Embed Dim   : {embed_dim}")

    for split in args.splits:
        extract_split(
            split=split,
            model_name=args.model,
            forward_fn=forward_fn,
            transform=transform,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.workers,
            overwrite=args.overwrite,
            logger=logger
        )

    logger.info("=" * 70)
    logger.info("  Feature Extraction Complete!")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
