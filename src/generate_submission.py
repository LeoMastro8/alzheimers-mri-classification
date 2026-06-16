import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

import config
from model import EarlyFusion3DCNN
from medical_net import MedicalNetTransfer
from dataloader import get_dataloader
from train import compute_metrics


def create_binary_submission(df_ternary, class_A, class_B, name_A, name_B, out_path):
    """
    Creates the binary sub-task CSV
    Filters out the 3rd class, re-normalizes probabilities, and sets the excluded class to NaN.
    """
    # Keep only patients whose true label is Class A or Class B
    df_bin = df_ternary[df_ternary["y_true"].isin([class_A, class_B])].copy()

    # Re-normalize the probabilities so they sum to 1.0
    prob_a_raw = df_bin[f"prob_{name_A}"]
    prob_b_raw = df_bin[f"prob_{name_B}"]
    sum_probs = prob_a_raw + prob_b_raw

    df_bin[f"prob_{name_A}"] = prob_a_raw / sum_probs
    df_bin[f"prob_{name_B}"] = prob_b_raw / sum_probs

    # Set the excluded class to NaN
    all_names = ["CN", "MCI", "AD"]
    excluded_name = [n for n in all_names if n not in [name_A, name_B]][0]
    df_bin[f"prob_{excluded_name}"] = np.nan

    # Recalculate y_pred, confidence, and correct for the binary context
    df_bin["y_pred"] = np.where(
        df_bin[f"prob_{name_A}"] > df_bin[f"prob_{name_B}"], class_A, class_B)
    df_bin["confidence"] = df_bin[[
        f"prob_{name_A}", f"prob_{name_B}"]].max(axis=1)
    df_bin["correct"] = (df_bin["y_pred"] == df_bin["y_true"]).astype(int)

    # Save
    df_bin.to_csv(out_path, index=False)
    print(f"Saved Binary CSV: {out_path.name}")


def generate_submission(experiment_name="baseline_3dcnn"):
    """
    Evaluates a specific model experiment, calculates metrics, 
    and saves formatted CSVs into a dedicated experiment folder.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"--- GENERATING FINAL SUBMISSIONS FOR '{experiment_name}' ON {device} ---")

    # Dynamically set the output directory based on the experiment name
    out_dir = config.OUTPUTS_DIR / "predictions" / experiment_name
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
        
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.eval()

    # Load Test Data
    test_loader = get_dataloader(
        "test_sessions", config.SPLIT_LAB_CSV, config.PREPROCESSED_DIR, batch_size=4, augment=False)

    all_y_true, all_y_pred, all_probs, all_sessions = [], [], [], []

    # Run Inference
    with torch.no_grad():
        for batch_x, batch_y, batch_sessions in test_loader:
            batch_x = batch_x.to(device)

            logits = model(batch_x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_y_true.extend(batch_y.numpy())
            all_y_pred.extend(preds)
            all_probs.extend(probs)
            all_sessions.extend(batch_sessions)

    all_probs = np.array(all_probs)
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    # Enforce Constraints & Build DataFrame
    df = pd.DataFrame({
        "model": experiment_name,
        "session_id": all_sessions,
        "y_true": all_y_true,
        "y_pred": all_y_pred,
        "prob_CN": all_probs[:, 0],
        "prob_MCI": all_probs[:, 1],
        "prob_AD": all_probs[:, 2],
        "confidence": np.max(all_probs, axis=1),
        "correct": (all_y_pred == all_y_true).astype(int)
    })

    # CONSTRAINT CHECKS
    assert len(df) == len(df['session_id'].unique()
                          ), "FAILED: Duplicate session_IDs found!"
    assert df.isnull().sum().sum() == 0, "FAILED: Missing values in ternary predictions!"
    assert np.allclose(df[['prob_CN', 'prob_MCI', 'prob_AD']].sum(
        axis=1), 1.0), "FAILED: Probabilities do not sum to 1.0!"
    print("All constraints verified successfully.")

    # Save Ternary CSV (Bug fixed: removed the duplicate 'predictions' subfolder string)
    df.to_csv(out_dir / "submission_ternary.csv", index=False)
    print(f"Saved Ternary CSV to: {out_dir.name}/submission_ternary.csv")

    # Generate the 3 Binary CSVs
    create_binary_submission(df, 0, 1, "CN", "MCI",
                             out_dir / "submission_binary_CN_vs_MCI.csv")
    create_binary_submission(df, 1, 2, "MCI", "AD",
                             out_dir / "submission_binary_MCI_vs_AD.csv")
    create_binary_submission(df, 0, 2, "CN", "AD",
                             out_dir / "submission_binary_CN_vs_AD.csv")

    # Calculate & Print Final 7 Metrics for the Report
    print(f"\n================================================")
    print(f"FINAL TEST SET EVALUATION: {experiment_name}")
    print(f"================================================")
    acc, bal_acc, mac_prec, mac_rec, mac_f1, mac_spec, auc = compute_metrics(
        all_y_true, all_y_pred, all_probs)

    print(f"Accuracy:           {acc:.4f}")
    print(f"Balanced Accuracy:  {bal_acc:.4f}")
    print(f"Macro Precision:    {mac_prec:.4f}")
    print(f"Macro Recall:       {mac_rec:.4f}")
    print(f"Macro F1-Score:     {mac_f1:.4f}")
    print(f"Macro Specificity:  {mac_spec:.4f}")
    print(f"AUC (OVR):          {auc:.4f}")
    print("================================================\n")


if __name__ == "__main__":
    generate_submission(experiment_name="baseline_3dcnn")
