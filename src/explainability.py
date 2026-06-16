import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from captum.attr import LayerGradCam, LayerAttribution
import config
from model import EarlyFusion3DCNN
from medical_net import MedicalNetTransfer
from dataloader import get_dataloader
import nibabel as nib
from nilearn import datasets, image


# def calculate_localization_score(heatmap_128, raw_nifti_path):
#     """
#     Reverse-stretches the 128x128 heatmap back to the patient's native raw MRI size
#     to steal its GPS coordinates, then overlays the Harvard-Oxford Atlas.
#     """
#     try:
#         # Load the RAW, original T1 and make it canonical (just like the preprocess.py did)
#         raw_img = nib.load(str(raw_nifti_path))
#         canonical_img = nib.as_closest_canonical(raw_img)

#         # Steal the original GPS coordinates and original shape
#         affine = canonical_img.affine
#         original_shape = canonical_img.shape  # e.g., (256, 256, 170)

#         # Stretch the 128x128x128 heatmap back to the original size
#         # Convert numpy array to PyTorch tensor: [Batch, Channel, D, H, W]
#         hm_tensor = torch.tensor(heatmap_128).unsqueeze(0).unsqueeze(0)

#         # Stretch using Trilinear interpolation
#         hm_stretched = F.interpolate(
#             hm_tensor,
#             size=original_shape,
#             mode='trilinear',
#             align_corners=False
#         )
#         hm_stretched_np = hm_stretched.squeeze().numpy()

#         # Create a perfect, native-space NIfTI image for the heatmap
#         heatmap_img = nib.Nifti1Image(hm_stretched_np, affine)

#         atlas = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')        
#         # If it's already an image, just use it. If it's a string, load it.
#         if isinstance(atlas.maps, str):
#             atlas_img = nib.load(atlas.maps)
#         else:
#             atlas_img = atlas.maps

#         # Shrink the Atlas to perfectly match our native-space heatmap
#         resampled_atlas = image.resample_to_img(
#             atlas_img, heatmap_img, interpolation='nearest')
#         atlas_data = resampled_atlas.get_fdata()

#         # Find the exact label IDs for Alzheimer's regions
#         ad_regions = [
#             'Left Hippocampus', 'Right Hippocampus',
#             'Left Amygdala', 'Right Amygdala',
#             'Left Lateral Ventricle', 'Right Lateral Ventricle'
#         ]
#         ad_indices = [atlas.labels.index(
#             region) for region in ad_regions if region in atlas.labels]

#         # Create a cookie-cutter mask and calculate the score
#         ad_mask = np.isin(atlas_data, ad_indices)

#         total_glow = np.sum(hm_stretched_np)
#         ad_glow = np.sum(hm_stretched_np[ad_mask])

#         if total_glow == 0:
#             return 0.0

#         return ad_glow / total_glow

#     except Exception as e:
#         print(f"Localization failed for {raw_nifti_path}: {e}")
#         return 0.0


# ==========================================
# Visualization
# ==========================================
def plot_gradcam_overlay(mri_volume, heatmap, save_path, class_name, subject_id):
    """Finds the 2D slice with the highest activation inside the 3D cube and plots it."""

    # We use the T1 channel (channel 0) as the anatomical background
    t1_volume = mri_volume[0].cpu().numpy()

    # Find the Z-slice (Axial) with the highest average heatmap activation
    slice_intensities = np.mean(heatmap, axis=(0, 1))
    best_z = np.argmax(slice_intensities)

    # Extract the 2D slices
    bg_slice = t1_volume[:, :, best_z]
    hm_slice = heatmap[:, :, best_z]

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Grad-CAM | True Class: {class_name} | Subject: {subject_id} | Slice Z={best_z}", fontsize=16)

    axes[0].imshow(bg_slice.T, cmap='gray', origin='lower')
    axes[0].set_title("Original T1")
    axes[0].axis('off')

    axes[1].imshow(hm_slice.T, cmap='jet', origin='lower')
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis('off')

    axes[2].imshow(bg_slice.T, cmap='gray', origin='lower')
    axes[2].imshow(hm_slice.T, cmap='jet', alpha=0.5,
                   origin='lower')  # alpha creates the overlay
    axes[2].set_title("Overlay")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ==========================================
# Search & Generation Loop
# ==========================================
def run_explainability(experiment_name="baseline_3dcnn"):
    """
    Generates Grad-CAM heatmaps and calculates localization scores 
    for a specific experiment, saving outputs to a dedicated folder.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"--- GENERATING Grad-CAM EXPLANATIONS FOR '{experiment_name}' ON {device} ---")

    # Dynamically set the output directory based on the experiment name
    out_dir = config.OUTPUTS_DIR / "explainability" / "gradcam" / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dynamically locate the correct model weights
    model_filename = f"{experiment_name}.pt"
    model_path = config.MODELS_DIR / model_filename

    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not find model weights at {model_path}. Did you run training with EXPERIMENT_NAME='{experiment_name}'?")

    # Load Model (Dynamically choose architecture based on the name)
    if "medicalnet" in experiment_name:
        # pretrained_weights_path=None because then I load MY trained weights
        model = MedicalNetTransfer(pretrained_weights_path=None).to(device)
    else:
        model = EarlyFusion3DCNN(num_classes=3).to(device)
        
    # load saved weights into the correct architecture
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.eval()  # CRITICAL for explanation!

    # Initialize Captum Grad-CAM
    if "medicalnet" in experiment_name:
        # MedicalNet's deepest convolutional layer
        target_layer = model.backbone.layer4
    else:
        # my custom CNN's deepest layer
        target_layer = model.block4
        
    layer_gc = LayerGradCam(model, target_layer)

    # Load Test Data
    test_loader = get_dataloader(
        "test_sessions", config.SPLIT_LAB_CSV, config.PREPROCESSED_DIR, batch_size=1, augment=False)

    classes = {0: "CN", 1: "MCI", 2: "AD"}
    found_counts = {0: 0, 1: 0, 2: 0}
    target_count = 5

    print("Searching for correctly classified subjects...")

    for batch_x, batch_y, batch_sessions in test_loader:
        if all(count >= target_count for count in found_counts.values()):
            break

        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        true_class = batch_y.item()
        subject_id = batch_sessions[0]

        if found_counts[true_class] >= target_count:
            continue

        # Run inference
        logits = model(batch_x)
        pred_class = torch.argmax(logits, dim=1).item()

        # Only process CORRECTLY classified subjects
        if pred_class == true_class:
            class_name = classes[true_class]
            print(f"[{class_name}] Found match: {subject_id}. Generating heatmap...")

            # --- CAPTUM HEATMAP GENERATION ---
            attribution = layer_gc.attribute(batch_x, target=true_class)
            upsampled_attr = LayerAttribution.interpolate(
                attribution, batch_x.shape[2:])

            heatmap_3d = upsampled_attr.squeeze().cpu().detach().numpy()
            heatmap_3d = np.maximum(heatmap_3d, 0)
            if np.max(heatmap_3d) > 0:
                heatmap_3d /= np.max(heatmap_3d)

            # # --- CALCULATE LOCALIZATION SCORE ---
            # subject_raw_dir = config.DATA_ROOT / subject_id

            # # Find the T1 file and EXPLICITLY filter out Mac's hidden "._" files
            # raw_t1_files = [
            #     f for f in subject_raw_dir.rglob("*T1*.nii*")
            #     if not f.name.startswith("._")
            # ]

            # if len(raw_t1_files) > 0:
            #     raw_nifti_path = raw_t1_files[0]
            #     loc_score = calculate_localization_score(
            #         heatmap_3d, str(raw_nifti_path))
            #     print(
            #         f"   -> AD Region Localization Score: {loc_score * 100:.2f}%")
            # else:
            #     print(
            #         f"   -> Warning: Could not find valid raw T1 for {subject_id} to calculate score.")
            # # ---------------------------------------------

            # --- PLOT AND SAVE THE OVERLAY ---
            save_name = out_dir / f"{class_name}_{subject_id}_gradcam.png"
            plot_gradcam_overlay(batch_x[0], heatmap_3d, str(
                save_name), class_name, subject_id)

            found_counts[true_class] += 1

    print(f"\nFinished! Heatmaps saved to: {out_dir}")


if __name__ == "__main__":
    # Easily run visualizations for different experiments
    run_explainability(experiment_name="baseline_3dcnn")
