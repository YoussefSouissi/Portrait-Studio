import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)

IMAGE_SIZE = 1024


def get_processed_root():
    from config import get_processed_root as _get
    return _get()

def get_celeba_root():
    from config import get_celeba_root as _get
    return _get()


class Text2ImageDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        data_root: Path = None,
        validate: bool = False,
    ):
        self.data_root = data_root or get_processed_root()
        self.celeba_root = get_celeba_root()
        self.celeba_img_dir = self.celeba_root / "img_align_celeba"
        self.split = split
        self.samples = []

        # Resize + center-crop to SDXL native resolution
        self._transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(IMAGE_SIZE),
        ])

        meta_path = self.data_root / split / "metadata.csv"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append({
                    "image_path": row["image_path"],
                    "prompt": row["prompt"],
                    "domain": row.get("domain", "celeba"),
                    "class_name": row.get("class_name", "visage_humain"),
                })

        if validate:
            self.samples = self._validate_samples()

        logger.info(f"Loaded {len(self.samples)} samples from {split} split")

    def _validate_samples(self) -> List[Dict]:
        from config import get_celeba_img_dir
        celeba_img_dir = get_celeba_img_dir()
        valid_samples = []
        for sample in self.samples:
            image_filename = sample["image_path"].split("/")[-1]
            img_path = celeba_img_dir / image_filename
            if not img_path.exists():
                logger.warning(f"Missing image: {sample['image_path']}")
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                if img.size[0] >= 64 and img.size[1] >= 64:
                    valid_samples.append(sample)
            except Exception as e:
                logger.warning(f"Corrupted image {sample['image_path']}: {e}")
        return valid_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_filename = sample["image_path"].split("/")[-1]
        img_path = self.celeba_img_dir / image_filename

        img = Image.open(img_path).convert("RGB")
        img = self._transform(img)

        img_array = np.array(img)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
        img_tensor = (img_tensor - 0.5) / 0.5

        return {
            "pixel_values": img_tensor,
            "prompt": sample["prompt"],
            "domain": sample["domain"],
            "class_name": sample["class_name"],
            "image_path": sample["image_path"],
        }


def load_sampling_weights(data_root: Path = None) -> Tuple[Dict, Dict]:
    data_root = data_root or get_processed_root()
    stats_path = data_root / "statistics.json"
    if not stats_path.exists():
        logger.warning(f"Statistics file not found: {stats_path}")
        return None, None
    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        return (
            stats.get("suggested_sampling_weights_class"),
            stats.get("suggested_sampling_weights_domain"),
        )
    except Exception as e:
        logger.warning(f"Error loading sampling weights: {e}")
        return None, None
