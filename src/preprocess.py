"""
MRI preprocessing for ADNI-style data.

Uses:
- HD-BET for deep learning-based skull stripping (T1)
- nibabel for loading NIfTI files and canonical reorientation
- ANTsPy (antspyx) for Rigid intra-subject co-registration to T1
- MONAI for trilinear resizing
- Custom NumPy logic for Percentile Min-Max intensity normalization

Pipeline per session:
1) Process T1: Run HD-BET -> Canonical -> Resize -> Normalize -> Extract 128³ Mask -> Save
2) Process T2/FLAIR:
    a) Register unstripped moving scan to unstripped T1 (ANTs Rigid)
    b) Canonical -> Resize (Whole head)
    c) Apply 128³ T1 Mask (Removes skull & prevents interpolation halos)
    d) Normalize -> Save

Output:
data/preprocessed/<session_id>/<session_id>_<modality>.npy
"""

import os
from pathlib import Path
import logging
import config
import shutil
import time

import nibabel as nib
import numpy as np
import torch
import ants
import subprocess

from monai.transforms import Resize

# =========================
# Config
# =========================
RAW_DATA_DIR = config.DATA_ROOT
OUTPUT_DIR = config.PREPROCESSED_DIR
TARGET_SHAPE = (128, 128, 128)
MODALITIES = config.MODALITIES

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("preprocessing.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def run_hdbet(input_path, output_path):
    safe_input = str(Path(input_path).resolve())
    safe_output = str(Path(output_path).resolve())

    cmd = [
        "hd-bet",
        "-i", safe_input,
        "-o", safe_output,
        "-device", "cuda:0",
        "--disable_tta"  # Mask flag completely removed
    ]

    logger.info(f"Running HD-BET: {' '.join(cmd)}")
    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end="")
        process.wait()
    except Exception as e:
        logger.error(f"HD-BET CRASHED: {e}")
        raise


# =========================
# Utilities
# =========================
def load_nifti(path):
    return nib.load(str(path))


def save_array(path, array):
    np.save(str(path), array.astype(np.float32))


def to_canonical(img):
    """
    Reorient image to nibabel canonical orientation.
    """
    return nib.as_closest_canonical(img)


# =========================
# Preprocessing steps
# =========================
def normalize_volume(arr):
    """
    Robust Min-Max normalization (0 to 1).
    Clips extreme outliers for better visual contrast and model stability.
    """
    mask = arr > 1e-3
    if not mask.any():
        return arr

    # Find the 1st and 99th percentiles of the brain tissue
    # This prevents one ultra-bright pixel from ruining the scale
    vmin = np.percentile(arr[mask], 1)
    vmax = np.percentile(arr[mask], 99)

    # Clip the array to ignore those extreme outliers
    arr_clipped = np.clip(arr, vmin, vmax)

    # Scale strictly to [0, 1]
    out = (arr_clipped - vmin) / (vmax - vmin + 1e-8)

    # Force the deep background strictly to 0 (pitch black)
    out[~mask] = 0.0

    return out


def resize_volume(arr, target_shape=TARGET_SHAPE):
    """
    Resize 3D array to target_shape via trilinear interpolation.
    """
    tensor = torch.from_numpy(arr).unsqueeze(0)  # [1, D, H, W]
    out = Resize(spatial_size=target_shape, mode="trilinear")(tensor)
    return out.squeeze(0).numpy()


def preprocess_nifti_image(img, target_shape=TARGET_SHAPE, mask_128=None, return_steps=False):
    canonical_img = to_canonical(img)
    canonical_array = canonical_img.get_fdata(dtype=np.float32)

    # 1. Resize first (If unstripped, no halo is created)
    resized_array = resize_volume(canonical_array, target_shape=target_shape)

    # 2. Apply the 128x128 mask if provided (Creates a perfect, sharp 0.0 background)
    if mask_128 is not None:
        resized_array[~mask_128] = 0.0

    # 3. Normalize (Calculates percentiles ONLY on the >0 masked brain tissue)
    normalized_array = normalize_volume(resized_array)

    final_array = normalized_array[np.newaxis, ...]  # [1, D, H, W]

    if return_steps:
        return {
            "original_img": img,
            "canonical_img": canonical_img,
            "canonical_array": canonical_array,
            "resized_array": resized_array,
            "normalized_array": normalized_array,
            "final_array": final_array,
        }

    return final_array


# =========================
# Session-Level Processing
# =========================
def preprocess_session(session_id, paths, output_dir=OUTPUT_DIR, target_shape=TARGET_SHAPE):
    session_output_dir = output_dir / session_id
    session_output_dir.mkdir(parents=True, exist_ok=True)

    # --- Session-level skip logic ---
    expected_outputs = [session_output_dir /
                        f"{session_id}_{mod}.npy" for mod in paths.keys()]
    if all(p.exists() for p in expected_outputs):
        logger.info(f"[{session_id}] All modalities exist. Skipping session.")
        return

    if "T1" not in paths:
        raise FileNotFoundError(f"Missing T1 volume for {session_id}.")

    t1_original_path = Path(paths["T1"])
    t1_real_path = t1_original_path.parent / f"{session_id}_T1.nii.gz"
    t1_path = t1_real_path if t1_real_path.exists() else t1_original_path

    # --- STEP 1: RUN HD-BET ---
    logger.info(f"[{session_id}] Stripping T1 with HD-BET...")

    temp_t1_in = session_output_dir / "input_T1.nii.gz"
    shutil.copy(t1_path, temp_t1_in)

    temp_t1_out = session_output_dir / "output_T1.nii.gz"

    run_hdbet(temp_t1_in, temp_t1_out)

    logger.info("Waiting for stripped brain to save...")

    # Wait for the single output file to physically appear and have a file size
    for _ in range(60):
        if temp_t1_out.exists() and temp_t1_out.stat().st_size > 1000:
            break
        time.sleep(1)

    if not temp_t1_out.exists():
        raise RuntimeError(f"FATAL: HD-BET failed to write {temp_t1_out}")

    # Process and save T1
    t1_out_path = session_output_dir / f"{session_id}_T1.npy"
    t1_img = load_nifti(temp_t1_out)
    t1_arr = preprocess_nifti_image(t1_img, target_shape=target_shape)
    save_array(t1_out_path, t1_arr)

    # --- EXTRACT 128x128 MASK FROM T1 ---
    # t1_arr is shape [1, 128, 128, 128]. The background is exactly 0.0
    t1_mask = t1_arr.squeeze(0) > 0

    # --- STEP 2: REGISTER AND MASK T2 & FLAIR ---
    for mod in ["T2", "FLAIR"]:
        if mod not in paths:
            continue

        mod_out_path = session_output_dir / f"{session_id}_{mod}.npy"
        if mod_out_path.exists():  # Don't forget your skip logic!
            continue

        moving_path = paths[mod]
        logger.info(f"[{session_id}] Registering and Masking {mod}...")

        # Register unstripped T2/FLAIR to unstripped T1
        fixed_ants = ants.image_read(str(t1_path))
        moving_ants = ants.image_read(str(moving_path))
        registration = ants.registration(
            fixed=fixed_ants, moving=moving_ants, type_of_transform='Rigid')
        registered_image = registration['warpedmovout']

        tmp_reg_path = session_output_dir / f"temp_reg_{mod}.nii.gz"
        ants.image_write(registered_image, str(tmp_reg_path))

        reg_img = nib.load(str(tmp_reg_path))

        # Pass the unstripped registered image, but provide the 128x128 mask!
        reg_arr = preprocess_nifti_image(
            reg_img, target_shape=target_shape, mask_128=t1_mask)

        save_array(mod_out_path, reg_arr)
        logger.info(f"[{session_id}] Finished {mod}.")

    # --- STEP 3: CLEANUP ---
    for temp_file in session_output_dir.glob("*.nii*"):
        try:
            temp_file.unlink()
        except Exception:
            pass

# =========================
# Dataset discovery
# =========================


def find_session_files(raw_dir=RAW_DATA_DIR, modalities=MODALITIES):
    """
    Recursively find files matching *{modality}*.nii* under raw_dir.
    Returns a dictionary grouped by session_id:
    { "session_id": {"T1": Path, "T2": Path, ...} }
    """
    sessions = {}

    for modality in modalities:
        for path in sorted(raw_dir.rglob(f"*{modality}*.nii*")):
            if path.name.startswith("._"):
                continue

            session_id = path.parent.name
            if session_id not in sessions:
                sessions[session_id] = {}

            sessions[session_id][modality] = path

    return sessions


# =========================
# Helpers for notebook tests
# =========================
def inspect_nifti(path):
    img = load_nifti(path)
    data = img.get_fdata(dtype=np.float32)

    return {
        "path": str(path),
        "shape": data.shape,
        "dtype": str(data.dtype),
        "min": float(data.min()),
        "max": float(data.max()),
        "mean": float(data.mean()),
        "std": float(data.std()),
        "affine": img.affine,
    }


# =========================
# Batch Runner
# =========================
def preprocess_batch(sessions_dict, output_dir=OUTPUT_DIR, target_shape=TARGET_SHAPE, limit=None):
    """
    Preprocess a dictionary of sessions.
    Saves outputs grouped by session_id, mirroring the raw structure.
    """
    failed = []
    session_items = list(sessions_dict.items())

    if limit is not None:
        session_items = session_items[:limit]

    for session_id, paths in session_items:
        try:
            logger.info(f"========== [START] {session_id} ==========")
            preprocess_session(session_id, paths, output_dir, target_shape)
            logger.info(f"========== [DONE] {session_id} ==========")
        except Exception as e:
            logger.error(f"[FAIL] {session_id}: {e}")
            failed.append((session_id, str(e)))

    return failed


# =========================
# Checks on saved outputs
# =========================
def check_preprocessed_file(path, expected_shape=(1, 128, 128, 128)):
    """
    Check one saved .npy file. Accounts for the 0.0 background.
    """
    arr = np.load(path)
    # Background is now 0.0 so brain tissue is everything else
    brain_tissue = arr[arr > 1e-5]

    return {
        "path": str(path),
        "shape_ok": tuple(arr.shape) == tuple(expected_shape),
        "shape": tuple(arr.shape),
        "has_nan": bool(np.isnan(arr).any()),
        "has_inf": bool(np.isinf(arr).any()),
        "brain_mean": float(brain_tissue.mean()) if brain_tissue.size else 0.0,
        "brain_std": float(brain_tissue.std()) if brain_tissue.size else 0.0,
    }


def check_preprocessed_folder(folder=OUTPUT_DIR, expected_shape=(1, 128, 128, 128)):
    results = []
    for path in sorted(folder.rglob("*.npy")):
        try:
            results.append(check_preprocessed_file(
                path, expected_shape=expected_shape))
        except Exception as e:
            results.append({"path": str(path), "error": str(e)})
    return results


# =========================
# Full run
# =========================
def run_full_preprocessing(raw_dir=RAW_DATA_DIR, output_dir=OUTPUT_DIR, start_idx=0, end_idx=None):
    # Find all files
    sessions = find_session_files(raw_dir=raw_dir)
    logger.info(f"Found {len(sessions)} total sessions in raw_data.")

    # Slice the dictionary into your specific chunk
    # (Convert to list, slice it, convert back to dict for preprocess_batch)
    chunked_sessions = dict(list(sessions.items())[start_idx:end_idx])
    logger.info(
        f"Processing chunk from index {start_idx} to {end_idx}. Total in this chunk: {len(chunked_sessions)}")

    # Pass ONLY the chunked dictionary to the batch processor
    failed = preprocess_batch(
        chunked_sessions,
        output_dir=output_dir,
        target_shape=TARGET_SHAPE,
        limit=None  # Set to None because the slicing already acts as the limit
    )
    return failed


if __name__ == "__main__":
    failed = run_full_preprocessing()
    logger.info(f"Done. Failed sessions: {len(failed)}")
    if failed:
        logger.error(f"Failed list: {failed}")
