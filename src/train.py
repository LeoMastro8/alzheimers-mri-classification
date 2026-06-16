import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

import config
from model import EarlyFusion3DCNN
from dataloader import get_dataloader, compute_class_weights, set_global_seeds

# ==========================================
# Focal Loss (For Class Imbalance)
# ==========================================


class FocalLoss(nn.Module):
    """
    Focal Loss: FL = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Down-weights well-classified examples to force the model to focus on hard cases (like MCI).
    """

    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight  # Tensor of class weights (alpha)
        self.gamma = gamma  # larger gamma means more focus on hard examples
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', weight=self.weight)

        # tensor[batch_size] where each element is the probability assigned to the correct class
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

# ==========================================
# Metrics Calculation
# ==========================================


def compute_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    # Macro metrics (treats all classes equally regardless of support size)
    mac_prec = precision_score(
        y_true, y_pred, average='macro', zero_division=0)
    mac_rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    mac_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Macro Specificity (Custom calculation using Confusion Matrix)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    specs = []
    for i in range(3):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        # Add 1e-7 to prevent divide-by-zero
        specs.append(tn / (tn + fp + 1e-7))
    mac_spec = np.mean(specs)

    # AUC-OVR
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except ValueError:
        auc = 0.5  # Fallback if a batch randomly misses a class

    return acc, bal_acc, mac_prec, mac_rec, mac_f1, mac_spec, auc

# # ==========================================
# Main Training Loop
# ==========================================


def train_model():
    # EXPERIMENT CONTROL PANEL
    # ==========================================
    # Run Identification
    EXPERIMENT_NAME = "focal_run_01"  # e.g., "baseline"
    MODEL_SAVE_NAME  = f"{EXPERIMENT_NAME}.pt"
    LOG_SAVE_NAME = f"training_log_{EXPERIMENT_NAME}.csv"

    # Hyperparameters
    EPOCHS = 100
    BATCH_SIZE = 8
    PATIENCE = 15
    LR = 2e-4     # Learning Rate
    WEIGHT_DECAY = 1e-3     # L2 Regularization

    # Loss Function
    USE_FOCAL_LOSS = False     # Set to True to focus on hard MCI cases
    FOCAL_GAMMA = 2.0      # How aggressively to focus on hard cases
    # ==========================================

    # Setup Paths & Devices
    set_global_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    csv_log_path = config.OUTPUTS_DIR / LOG_SAVE_NAME
    model_save_path = config.MODELS_DIR / MODEL_SAVE_NAME

    # Dataloaders
    train_loader = get_dataloader("train_sessions", config.SPLIT_LAB_CSV,
                                  config.PREPROCESSED_DIR, batch_size=BATCH_SIZE, augment=True)
    val_loader = get_dataloader("eval_sessions", config.SPLIT_LAB_CSV,
                                config.PREPROCESSED_DIR, batch_size=BATCH_SIZE, augment=False)

    # Model Initialization (With Multi-GPU support if available)
    model = EarlyFusion3DCNN(num_classes=3).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    class_weights = compute_class_weights(
        config.SPLIT_LAB_CSV, "train_sessions").to(device)

    # Loss Setup
    if USE_FOCAL_LOSS:
        print(f"Using Focal Loss (gamma={FOCAL_GAMMA})")
        criterion = FocalLoss(weight=class_weights, gamma=FOCAL_GAMMA)
    else:
        print("Using Weighted Cross-Entropy Loss")
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=LR,
                           weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5)

    # Tracking
    best_val_bal_acc = 0.0
    epochs_no_improve = 0

    # Create CSV Header
    with open(csv_log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train_Loss", "Val_Loss", "Accuracy", "Balanced_Accuracy",
                        "Macro_Precision", "Macro_Recall", "Macro_F1", "Macro_Specificity", "AUC_OVR"])

    # Training Loop
    for epoch in range(1, EPOCHS + 1):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0

        for batch_x, batch_y, _ in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_loader.dataset)

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        all_y_true, all_y_pred, all_y_prob = [], [], []

        with torch.no_grad():
            for batch_x, batch_y, _ in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * batch_x.size(0)

                # Convert logits to probabilities using Softmax
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_y_true.extend(batch_y.cpu().numpy())
                all_y_pred.extend(preds.cpu().numpy())
                all_y_prob.extend(probs.cpu().numpy())

        val_loss /= len(val_loader.dataset)

        # Calculate Metrics
        acc, bal_acc, mac_prec, mac_rec, mac_f1, mac_spec, auc = compute_metrics(
            np.array(all_y_true), np.array(all_y_pred), np.array(all_y_prob)
        )

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Bal-Acc: {bal_acc:.4f} | AUC: {auc:.4f}")

        # Log to CSV
        with open(csv_log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, acc,
                            bal_acc, mac_prec, mac_rec, mac_f1, mac_spec, auc])

        # Scheduler Step
        scheduler.step(bal_acc)

        # Early Stopping & Checkpointing (Safe Save included!)
        if bal_acc > best_val_bal_acc:
            best_val_bal_acc = bal_acc
            epochs_no_improve = 0

            # Safe save that prevents the DataParallel Dictionary Bug
            saved_weights = model.module.state_dict() if hasattr(
                model, 'module') else model.state_dict()
            torch.save(saved_weights, model_save_path)

            print(
                f"   -> Saved new best model to {MODEL_SAVE_NAME}! (Bal-Acc: {bal_acc:.4f})")
        else:
            epochs_no_improve += 1
            print(
                f"   -> No improvement. Patience: {epochs_no_improve}/{PATIENCE}")

            if epochs_no_improve >= PATIENCE:
                print("\n Early Stopping Triggered! Training Halted.")
                break


if __name__ == "__main__":
    train_model()
