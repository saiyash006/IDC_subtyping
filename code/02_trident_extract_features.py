"""
02_trident_extract_features.py
==============================
Phase 2 (Alternative) using Mahmood Lab TRIDENT toolkit.

Extracts dense feature embeddings from the generated 256x256 patches using 
TRIDENT's `encoder_factory`, which automatically handles proper preprocessing
and normalization for each supported foundation model (UNI, CONCH, Phikon, etc).

Inputs:
    outputs/patches/{split}/{patient_id}/*.jpg
Outputs:
    outputs/trident_features/{model_name}/{split}/{patient_id}.pt
    (shape: [N_patches, embed_dim])

Usage:
    python code/02_trident_extract_features.py --splits train val test --model phikon
"""

import argparse
import logging
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

try:
    from trident.patch_encoder_models.load import encoder_factory
except ImportError:
    raise ImportError(
        "TRIDENT not found. Please install it via:\n"
        "git clone https://github.com/mahmoodlab/trident.git\n"
        "cd trident && pip install -e ."
    )

# ─────────────────────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────────────────────
DATASET_ROOT = Path("/home/cgntlab5/bcnb/BCNB dataset")
PATCHES_ROOT = DATASET_ROOT / "outputs" / "patches"
OUT_ROOT     = DATASET_ROOT / "outputs" / "trident_features"
LOG_DIR      = DATASET_ROOT / "outputs" / "logs"

# ─────────────────────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────────────────────
def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("trident_extract_features")
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
#  DATASET
# ─────────────────────────────────────────────────────────────────────────────
class PatientPatchDataset(Dataset):
    """Loads all patches for a single patient."""
    def __init__(self, patch_paths: List[Path]):
        self.patch_paths = patch_paths

    def __len__(self):
        return len(self.patch_paths)

    def __getitem__(self, idx):
        path = self.patch_paths[idx]
        img = Image.open(str(path)).convert("RGB")
        return img, str(path)

# Custom collate_fn because TRIDENT's encoder_factory models might expect a list of PIL Images
# rather than a stacked tensor, but typical PyTorch expects tensors. However, the encoder's 
# built-in transform requires either PIL images or already transformed tensors.
# If we apply transform here, we need the transform from the encoder.
# Fortunately, we can extract the transform from the encoder.

# ─────────────────────────────────────────────────────────────────────────────
#  EXTRACTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def extract_split(
    split: str,
    model_name: str,
    encoder,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    overwrite: bool,
    logger: logging.Logger,
    patch_size: int
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

        sorted_patch_paths = sorted([str(p) for p in patch_paths])
        feat_dict = {}

        # Since TRIDENT's encoder often comes with an internal transform (or handles it inside),
        # we can pass PIL Images directly if the encoder's __call__ supports it, or use the 
        # encoder's `transform` or `eval_transforms` attribute. For robustness with standard TRIDENT workflow:
        transform = getattr(encoder, "transform", None) or getattr(encoder, "eval_transforms", None)

        if transform:
            class TransformedDataset(Dataset):
                def __init__(self, paths, transform, target_size):
                    self.paths = paths
                    self.transform = transform
                    self.target_size = target_size
                def __len__(self): return len(self.paths)
                def __getitem__(self, idx):
                    img = Image.open(self.paths[idx]).convert("RGB")
                    img = img.resize((self.target_size, self.target_size))
                    return self.transform(img), self.paths[idx]
            
            dataset = TransformedDataset(sorted_patch_paths, transform, patch_size)
        else:
            class ResizeDataset(Dataset):
                def __init__(self, paths, target_size):
                    self.paths = paths
                    self.target_size = target_size
                def __len__(self): return len(self.paths)
                def __getitem__(self, idx):
                    img = Image.open(self.paths[idx]).convert("RGB")
                    img = img.resize((self.target_size, self.target_size))
                    return img, self.paths[idx]
            
            dataset = ResizeDataset(sorted_patch_paths, patch_size)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
            collate_fn=None if transform else lambda batch: (
                [item[0] for item in batch],
                [item[1] for item in batch]
            )
        )

        with torch.no_grad():
            for batch_imgs, batch_paths in loader:
                if transform:
                    batch_imgs = batch_imgs.to(device, non_blocking=True)
                
                # TRIDENT encoders return embeddings directly
                feats = encoder(batch_imgs)
                
                if isinstance(feats, dict) and "embedding" in feats:
                    feats = feats["embedding"]
                
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
    parser = argparse.ArgumentParser(description="Extract embeddings using TRIDENT")
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val", "test"])
    parser.add_argument("--patch_encoder", type=str, default="uni_v2", help="TRIDENT supported model name (e.g., uni_v2, phikon_v2)")
    parser.add_argument("--patch_size", type=int, default=256, help="Patch size to resize to if needed")
    parser.add_argument("--mag", type=float, default=20, help="Magnification (for naming/logging)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging(LOG_DIR / "02_trident_extract_features.log")

    logger.info("=" * 70)
    logger.info("  Phase 2: Feature Extraction (TRIDENT)")
    logger.info("=" * 70)
    logger.info(f"  Patch Encoder: {args.patch_encoder}")
    logger.info(f"  Patch size  : {args.patch_size}")
    logger.info(f"  Magnification: {args.mag}x")
    logger.info(f"  Batch size  : {args.batch_size}")
    logger.info(f"  Workers     : {args.workers}")
    logger.info(f"  Splits      : {args.splits}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  Device      : {device}")

    # Load TRIDENT encoder
    logger.info(f"Loading TRIDENT encoder: {args.patch_encoder}")
    encoder = encoder_factory(args.patch_encoder)
    encoder = encoder.to(device)
    encoder.eval()

    for split in args.splits:
        extract_split(
            split=split,
            model_name=f"{args.patch_encoder}_patch{args.patch_size}_mag{int(args.mag)}",
            encoder=encoder,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.workers,
            overwrite=args.overwrite,
            logger=logger,
            patch_size=args.patch_size
        )

    logger.info("=" * 70)
    logger.info("  Feature Extraction Complete!")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
