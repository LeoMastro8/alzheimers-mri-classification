"""
DataLoader pipeline with 3D augmentation
"""

import random
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from adni_dataset import ADNIDataset

import monai.transforms as mt


# ── Augmentation ───────────────────────────────────────────────────────────────

def get_train_transforms():
    """
    Agmentations for multi-channel 3D brain MRI.
    Avoids flipping (to preserve anatomical asymmetry) and limits rotations
    to prevent aggressive edge clipping.
    """
    # 5 degrees is much safer than 10 to avoid clipping the cortex at the edges
    radians_5 = 5 * math.pi / 180.0

    return mt.Compose([
        # Small random affine transformations (rotation and translation)
        # padding_mode="zeros" is safe here because the background is 0.0
        mt.RandAffine(
            prob=0.5,
            # sample a random angle between -5 degrees and +5 degrees for the X, Y, and Z axes independently
            rotate_range=(radians_5, radians_5, radians_5),
            translate_range=(5, 5, 5), # Shift by max 5 voxels
            padding_mode="zeros"
        ),

        # Subtle intensity scaling (simulates different scanner calibrations)
        mt.RandScaleIntensity(factors=0.1, prob=0.5),

        # Subtle Gaussian noise (simulates sensor noise)
        mt.RandGaussianNoise(prob=0.3, mean=0.0, std=0.02)
    ])


# ── Seeds ─────────────────────────────────────────────────────────────────

def set_global_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


# ── Class weights ─────────────────────────────────────────────────────────

def compute_class_weights(csv_path, split="train_sessions"):
    """
    Calculates balanced class weights for CrossEntropyLoss.
    Formula: N_total / (N_classes * N_c)
    """
    df = pd.read_csv(csv_path)
    counts = df[df["split"] == split]["label_num"].value_counts().sort_index()
    
    # Calculate weights and convert to float32 tensor
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights.values, dtype=torch.float32)


# ── DataLoader ────────────────────────────────────────────────────────────

def get_dataloader(split, csv_path, data_dir, batch_size=4, augment=False, num_workers=4):
    """
    Creates a PyTorch DataLoader for the given split.
    
    Args:
        split (str): 'train_sessions', 'eval_sessions', or 'test_sessions'
        csv_path (str): Path to the split_labels.csv
        data_dir (str): Root directory of preprocessed .npy files
        augment (bool): Whether to apply training augmentations
        num_workers (int): Number of sub-processes for data loading
    """
    # Only apply transforms if we are training AND augment is explicitly True
    transform = get_train_transforms() if augment and split == "train_sessions" else None
    
    dataset = ADNIDataset(
        csv_path=csv_path, 
        split_name=split, 
        data_dir=data_dir, 
        transform=transform,
        use_preprocessed=True # Enforce using the optimized .npy pipeline
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train_sessions"),
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        worker_init_fn=worker_init_fn,
        persistent_workers=(num_workers > 0),
    )