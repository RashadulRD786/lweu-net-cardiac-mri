"""
PyTorch Dataset for ACDC cardiac MRI segmentation.

Loads preprocessed .npy pairs from data/processed/{split}/images/ and masks/.

Preprocessing already completed upstream (pipeline.py):
  - 224×224 resize
  - 1.5mm isotropic resampling
  - Z-score normalization (foreground, 0.5–99.5 percentile clipping)
  - Label encoding: {0=BG, 1=RV, 2=MYO, 3=LV}

Split counts (seed=42, patient-level stratified):
  - Train : 1506 slices
  - Val   :  396 slices
  - Test  : 1076 slices
"""

"""
    Dataset for ACDC preprocessed 2D slices.

    Parameters
    ----------
    data_dir : str
        Root of processed data, e.g. 'data/processed'.
        Expected structure:
            {data_dir}/{split}/images/<name>.npy   float32, shape (224, 224)
            {data_dir}/{split}/masks/<name>.npy    uint8,   shape (224, 224)
    split : str
        One of 'train', 'val', 'test'.
    augment : bool, optional
        If True, applies training augmentation pipeline. Default follows split
        (True for train, False otherwise). Pass explicitly to override.

    Returns (per __getitem__)
    -------
    image : torch.Tensor  shape (1, H, W), float32
    mask  : torch.Tensor  shape (H, W),    int64 (long) — required by nn.CrossEntropyLoss
    """

import os
import numpy as np
import torch
from torch.utils.data import Dataset
 
from src.data.augmentation import get_train_augmentation, get_val_augmentation
 
 
class ACDCDataset(Dataset):
    NUM_CLASSES = 4
    IMAGE_SIZE  = (224, 224)
 
    def __init__(self, data_dir: str, split: str, augment: bool = None):
        assert split in ("train", "val", "test"), \
            f"split must be 'train', 'val', or 'test', got '{split}'"
 
        self.split     = split
        self.augment   = augment if augment is not None else (split == "train")
        self.split_dir = os.path.join(data_dir, split)
 
        if not os.path.isdir(self.split_dir):
            raise FileNotFoundError(
                f"Split directory not found: {self.split_dir}\n"
                "Check that data_dir points to 'data/preprocessed'."
            )
 
        # stems: 'patient003_ED_slice00'  (without _img.npy suffix)
        self.stems = sorted([
            f.replace("_img.npy", "")
            for f in os.listdir(self.split_dir)
            if f.endswith("_img.npy")
        ])
 
        if len(self.stems) == 0:
            raise RuntimeError(
                f"No *_img.npy files found in {self.split_dir}."
            )
 
        missing = [
            s for s in self.stems
            if not os.path.exists(os.path.join(self.split_dir, f"{s}_msk.npy"))
        ]
        if missing:
            raise RuntimeError(
                f"{len(missing)} image(s) have no matching mask.\n"
                f"First missing: {missing[0]}"
            )
 
        self.transform = (
            get_train_augmentation() if self.augment
            else get_val_augmentation()
        )
 
    def __len__(self) -> int:
        return len(self.stems)
 
    def __getitem__(self, idx: int):
        stem  = self.stems[idx]
        image = np.load(os.path.join(self.split_dir, f"{stem}_img.npy"))
        mask  = np.load(os.path.join(self.split_dir, f"{stem}_msk.npy"))
 
        assert image.shape == self.IMAGE_SIZE, f"Bad image shape {image.shape}"
        assert mask.shape  == self.IMAGE_SIZE, f"Bad mask shape {mask.shape}"
        assert mask.max()  <= 3,               f"Invalid label {mask.max()}"
 
        augmented = self.transform(image=image, mask=mask.astype(np.uint8))
        image = torch.from_numpy(augmented["image"]).unsqueeze(0).float()
        mask  = torch.from_numpy(augmented["mask"].astype(np.int64)).long()
 
        return image, mask
 
    def __repr__(self) -> str:
        return (f"ACDCDataset(split='{self.split}', "
                f"n_slices={len(self)}, augment={self.augment})")
 