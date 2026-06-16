import os
import pandas as pd
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class ADNIDataset(Dataset):
    """PyTorch Dataset for ADNI multi-modal MRI data.

    Supports two modes via the `use_preprocessed` flag:
    - True: Returns stacked early-fusion tensor (x, y, session_id)
    - False: Returns unpacked tuple (t1, t2, flair, label, session_id)
    """

    # Fixed order used when returning the three MRI volumes.
    modalities = ("T1", "T2", "FLAIR")

    def __init__(self, csv_path, split_name, data_dir, use_preprocessed=True, transform=None):
        """
        csv_path         : path to split_labels.csv
        split_name       : 'train_sessions', 'eval_sessions', or 'test_sessions'
        data_dir         : path to the folder that contains all session subfolders
        use_preprocessed : boolean flag to toggle between raw .nii.gz and preprocessed .npy
        transform        : optional function applied to the combined volume
        """

        df = pd.read_csv(csv_path)
        self.records = df[df["split"] == split_name].reset_index(drop=True)

        self.data_dir = data_dir
        self.transform = transform
        self.use_preprocessed = use_preprocessed

    def __len__(self):
        return len(self.records)

    def _volume_path(self, session_id, modality):
        if self.use_preprocessed:
            # Load preprocessed .npy
            path = os.path.join(self.data_dir, session_id,
                                f"{session_id}_{modality}.npy")
            if os.path.exists(path):
                return path
        else:
            # Load raw compressed or uncompressed NIfTI files
            base = os.path.join(self.data_dir, session_id,
                                f"{session_id}_{modality}")
            for ext in (".nii.gz", ".nii"):
                path = base + ext
                if os.path.exists(path):
                    return path

        raise FileNotFoundError(
            f"Missing {modality} volume for {session_id} (preprocessed={self.use_preprocessed})")

    def _load_volume(self, session_id, modality):
        path = self._volume_path(session_id, modality)

        if self.use_preprocessed:
            # Preprocessed arrays are already saved as [1, D, H, W]
            array = np.load(path).astype(np.float32)
            tensor = torch.from_numpy(array)
        else:
            # Raw arrays need the channel dimension added
            image = nib.load(path)
            array = np.asarray(image.dataobj, dtype=np.float32)
            tensor = torch.from_numpy(array).unsqueeze(
                0)  # Shape: (1, D, H, W)

        return tensor

    def __getitem__(self, idx):
        # Read one row from the split CSV.
        row = self.records.iloc[idx]
        session_id = row["session_id"]
        label = int(row["label_num"]) if "label_num" in row else -1
        y = torch.tensor(label, dtype=torch.long)

        # Load t1, t2, flair for this subject/session.
        volumes = [self._load_volume(session_id, modality)
                   for modality in self.modalities]

        if self.use_preprocessed:
            # --- TRAINING MODE ---
            # Early fusion: [3, D, H, W]
            x = torch.cat(volumes, dim=0)

            if self.transform is not None:
                x = self.transform(x)

            return x, y, session_id

        else:
            # # Stack temporarily just in case a transform needs to be applied uniformly
            # x = torch.cat(volumes, dim=0)

            # if self.transform is not None:
            #     x = self.transform(x)

            # # Unpack back to [1, D, H, W] to match older utility scripts
            # t1 = x[0].unsqueeze(0)
            # t2 = x[1].unsqueeze(0)
            # flair = x[2].unsqueeze(0)

            return volumes[0], volumes[1], volumes[2], y, session_id
