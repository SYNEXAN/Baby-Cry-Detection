"""
Script: scripts/build_compact_audio_notebook.py
Purpose:
Generate a clean, professional, emoji-free Kaggle notebook for training the
Compact Audio CNN architecture (~241k parameters) with aggressive multi-layer
regularization (Spatial Dropout, Label Smoothing, Weight Decay, SpecAugment, Early Stopping)
on NVIDIA Tesla T4 GPU under strict 100% Zero Data Leakage.
"""

import json
import os

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_md(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(text):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

# CELL 1: Markdown Header
add_md("""# Compact Audio CNN for Infant Cry Cause Classification (5 Classes)
### Anti-Overfitting Acoustic Architecture: Compact Audio CNN (~241k Parameters)
- Dataset: vuitang/donateacry-eureka-10s-balanced (10.0s Audio @ 16kHz, 5 Classes, 2,500 Samples)
- Protocol: 100% Zero Data Leakage (Patient/Recording-level independent split)
- Regularization: Spatial Dropout2d (0.2 -> 0.4) + Classifier Dropout (0.5) + SpecAugment
- Loss Function: Label Smoothing Cross-Entropy (epsilon=0.1) + Weight Decay (1e-2)
- Convergence Control: Early Stopping (Patience = 7 on Validation Loss)
- Target Hardware: NVIDIA Tesla T4 GPU
""")

# CELL 2: Environment and GPU Verification
add_code("""# Cell 1: Environment and GPU Verification
import os
import sys
import time
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=== ENVIRONMENT AND GPU VERIFICATION ===")
print(f"Python Version   : {sys.version.split()[0]}")
print(f"PyTorch Version  : {torch.__version__}")
print(f"Torchaudio       : {torchaudio.__version__}")
print(f"CUDA Available   : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"Active Device    : {gpu_name}")
    print(f"GPU Memory Total : {gpu_mem:.2f} GB")
else:
    device = torch.device("cpu")
    print("WARNING: CUDA is not available. Execution will proceed on CPU.")

print(f"Execution Target : {device}")
""")

# CELL 3: Dataset Connection & Configuration
add_code("""# Cell 2: Dataset Connection and Configuration
possible_roots = [
    "/kaggle/input/donateacry-eureka-10s-balanced",
    "/kaggle/input/datasets/vuitang/donateacry-eureka-10s-balanced",
    "/kaggle/input/datasets/thittn/donateacry-eureka-10s-balanced",
    "data/processed/audio_eureka_10s_balanced"
]

DATASET_ROOT = None
for candidate in possible_roots:
    if os.path.exists(candidate):
        DATASET_ROOT = candidate
        break

if DATASET_ROOT is None:
    for root, dirs, files in os.walk("/kaggle/input"):
        if "metadata.csv" in files:
            DATASET_ROOT = root
            break

assert DATASET_ROOT is not None, "Error: Could not locate dataset root directory with metadata.csv"

META_CSV = os.path.join(DATASET_ROOT, "metadata.csv")
assert os.path.exists(META_CSV), f"Error: Metadata file missing at {META_CSV}"

df_meta = pd.read_csv(META_CSV)
print("=== DATASET MOUNTING VERIFICATION ===")
print(f"Dataset Root Directory : {DATASET_ROOT}")
print(f"Total Audio Samples    : {len(df_meta)}")
print("\\nSamples distribution by split and category:")
print(df_meta.groupby(["split", "category"]).size())

CLASSES = sorted(df_meta["category"].unique())
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
print(f"Target Classes ({len(CLASSES)}): {CLASSES}")
""")

# CELL 4: Zero Data Leakage Verification
add_code("""# Cell 3: Dataset Structure and Annotation Validation
print("=== DATASET INTEGRITY VALIDATION ===")
df_train = df_meta[df_meta["split"] == "train"].reset_index(drop=True)
df_val = df_meta[df_meta["split"] == "val"].reset_index(drop=True)
df_test = df_meta[df_meta["split"] == "test"].reset_index(drop=True)

print(f"Train Set Count : {len(df_train)} samples ({len(df_train) // len(CLASSES)} per class)")
print(f"Val Set Count   : {len(df_val)} samples ({len(df_val) // len(CLASSES)} per class)")
print(f"Test Set Count  : {len(df_test)} samples ({len(df_test) // len(CLASSES)} per class)")

train_orig = set(df_train["orig_file"].unique())
val_orig = set(df_val["orig_file"].unique())
test_orig = set(df_test["orig_file"].unique())

overlap_tr_val = train_orig.intersection(val_orig)
overlap_tr_te = train_orig.intersection(test_orig)
overlap_val_te = val_orig.intersection(test_orig)

print(f"Unique original recordings in Train: {len(train_orig)}")
print(f"Unique original recordings in Val  : {len(val_orig)}")
print(f"Unique original recordings in Test : {len(test_orig)}")
print(f"Overlap Train & Val  : {len(overlap_tr_val)}")
print(f"Overlap Train & Test : {len(overlap_tr_te)}")
print(f"Overlap Val & Test   : {len(overlap_val_te)}")

assert len(overlap_tr_val) == 0 and len(overlap_tr_te) == 0 and len(overlap_val_te) == 0, "Validation Failure: Data leakage detected across splits!"
print("Status: 100% Zero Data Leakage verified successfully.")
""")

# CELL 5: Data Visualization & Waveform / Spectrogram Sample Inspection
add_code("""# Cell 4: Data Visualization and Sample Inspection
TARGET_SR = 16000
DURATION_SEC = 10.0
TARGET_SAMPLES = int(TARGET_SR * DURATION_SEC)
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512

mel_transform = T.MelSpectrogram(
    sample_rate=TARGET_SR,
    n_fft=N_FFT,
    win_length=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    power=2.0
)
amp_to_db = T.AmplitudeToDB()

plt.figure(figsize=(15, 6))
for idx, cat in enumerate(CLASSES):
    sample_row = df_train[df_train["category"] == cat].iloc[0]
    rel_path = str(sample_row["filepath"]).replace(chr(92), "/")
    fname = os.path.basename(rel_path)
    split = sample_row["split"]
    
    candidates = [
        os.path.join(DATASET_ROOT, split, cat, fname),
        os.path.join(DATASET_ROOT, split, split, cat, fname),
        os.path.join(DATASET_ROOT, cat, fname),
        os.path.join(DATASET_ROOT, rel_path)
    ]
    full_path = candidates[0]
    for c in candidates:
        if os.path.exists(c):
            full_path = c
            break
            
    if not os.path.exists(full_path):
        matches = list(Path(DATASET_ROOT).rglob(fname))
        if matches:
            full_path = str(matches[0])
            
    waveform, sr = torchaudio.load(full_path)
    if sr != TARGET_SR:
        waveform = T.Resample(sr, TARGET_SR)(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if waveform.shape[1] > TARGET_SAMPLES:
        waveform = waveform[:, :TARGET_SAMPLES]
        
    mel = amp_to_db(mel_transform(waveform))[0].numpy()
    
    plt.subplot(2, 5, idx + 1)
    plt.plot(waveform[0].numpy(), color="navy", lw=0.5)
    plt.title(f"Waveform: {cat}", fontsize=9)
    plt.axis("off")
    
    plt.subplot(2, 5, idx + 6)
    plt.imshow(mel, aspect="auto", origin="lower", cmap="magma")
    plt.title(f"Log-Mel: {cat}", fontsize=9)
    plt.axis("off")

plt.tight_layout()
plt.savefig("data_inspection_samples.png", dpi=300)
plt.show()
print("Saved sample inspection plot to data_inspection_samples.png")
""")

# CELL 6: Acoustic Feature Extraction & Dataset Class with SpecAugment
add_code("""# Cell 5: Acoustic Feature Extraction and Dataset Definition
class CompactCryDataset(Dataset):
    def __init__(self, df, data_root, is_train=False):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = str(row["filepath"]).replace(chr(92), "/")
        fname = os.path.basename(rel_path)
        split = row["split"]
        cat = row["category"]
        
        candidates = [
            os.path.join(self.data_root, split, cat, fname),
            os.path.join(self.data_root, split, split, cat, fname),
            os.path.join(self.data_root, cat, fname),
            os.path.join(self.data_root, rel_path)
        ]
        full_path = candidates[0]
        for c in candidates:
            if os.path.exists(c):
                full_path = c
                break
                
        if not os.path.exists(full_path):
            matches = list(Path(self.data_root).rglob(fname))
            if matches:
                full_path = str(matches[0])
                
        waveform, sr = torchaudio.load(full_path)
        if sr != TARGET_SR:
            waveform = T.Resample(sr, TARGET_SR)(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        if waveform.shape[1] < TARGET_SAMPLES:
            repeats = int(np.ceil(TARGET_SAMPLES / waveform.shape[1]))
            waveform = waveform.repeat(1, repeats)[:, :TARGET_SAMPLES]
        elif waveform.shape[1] > TARGET_SAMPLES:
            waveform = waveform[:, :TARGET_SAMPLES]

        # Log-Mel Spectrogram extraction (1 Channel: 1 x 128 x 313)
        mel = mel_transform(waveform)
        log_mel = amp_to_db(mel)
        mean = log_mel.mean()
        std = log_mel.std() + 1e-6
        norm_mel = (log_mel - mean) / std

        # SpecAugment on Train set to prevent overfitting
        if self.is_train:
            if random.random() > 0.3:
                f_mask = T.FrequencyMasking(freq_mask_param=16)
                norm_mel = f_mask(norm_mel)
            if random.random() > 0.3:
                t_mask = T.TimeMasking(time_mask_param=40)
                norm_mel = t_mask(norm_mel)

        label = CLASS_TO_IDX[row["category"]]
        return norm_mel, label

train_dataset = CompactCryDataset(df_train, DATASET_ROOT, is_train=True)
val_dataset = CompactCryDataset(df_val, DATASET_ROOT, is_train=False)
test_dataset = CompactCryDataset(df_test, DATASET_ROOT, is_train=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True if torch.cuda.is_available() else False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True if torch.cuda.is_available() else False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True if torch.cuda.is_available() else False)

print("DataLoaders initialized:")
print(f"Train: {len(train_loader)} batches | Val: {len(val_loader)} batches | Test: {len(test_loader)} batches")
""")

# CELL 7: Compact Audio CNN Architecture (~241k parameters)
add_code("""# Cell 6: Compact Audio CNN Architecture
class CompactAudioCNN(nn.Module):
    \"\"\"
    Ultra-lightweight Compact Audio CNN designed specifically to prevent
    acoustic identity memorization on small infant cry datasets.
    Total Parameters: ~241,605 (88x smaller than ResNet34)
    \"\"\"
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.4),

            # Block 4
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        logits = self.classifier(feat)
        return logits

model = CompactAudioCNN(num_classes=len(CLASSES)).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("=== COMPACT MODEL INITIALIZATION ===")
print(f"Architecture         : CompactAudioCNN (4-Block Conv2d + Global Pooling)")
print(f"Total Parameters     : {total_params:,} (88x smaller than ResNet34)")
print(f"Trainable Parameters : {trainable_params:,}")
print(f"Parameter-to-Sample  : ~{total_params / len(train_dataset):.1f} params per train sample (vs 10,700 in ResNet34)")
""")

# CELL 8: Training Configuration & Regularization
add_code("""# Cell 7: Training Configuration and Regularization
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

TOTAL_EPOCHS = 35
PATIENCE = 7
INITIAL_LR = 1e-3
WEIGHT_DECAY = 1e-2

optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-5)

print("=== REGULARIZED TRAINING HYPERPARAMETERS ===")
print(f"Total Epochs         : {TOTAL_EPOCHS}")
print(f"Early Stopping       : Patience = {PATIENCE} epochs on Validation Loss")
print(f"Initial LR           : {INITIAL_LR}")
print(f"Weight Decay         : {WEIGHT_DECAY}")
print(f"Loss Function        : CrossEntropyLoss (label_smoothing=0.1)")
print(f"Batch Size           : 32")
""")

# CELL 9: Model Training Loop with Early Stopping
add_code("""# Cell 8: Model Training Execution with Early Stopping
print("=== STARTING COMPACT MODEL TRAINING ===")

best_val_loss = float("inf")
best_val_f1 = 0.0
best_epoch = 0
best_model_path = "best_compact_audio_classifier.pt"
history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

patience_counter = 0
t_start = time.time()

for epoch in range(1, TOTAL_EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for x_b, y_b in train_loader:
        x_b, y_b = x_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        out = model(x_b)
        loss = criterion(out, y_b)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(y_b)

    scheduler.step()
    train_loss = running_loss / len(train_dataset)

    model.eval()
    val_loss = 0.0
    y_true, y_pred = [], []
    with torch.no_grad():
        for x_b, y_b in val_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            out = model(x_b)
            loss = criterion(out, y_b)
            val_loss += loss.item() * len(y_b)
            preds = torch.argmax(out, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y_b.cpu().numpy())

    val_loss /= len(val_dataset)
    val_acc = accuracy_score(y_true, y_pred)
    val_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["val_f1"].append(val_f1)

    checkpoint_note = ""
    # Checkpoint criterion: lower val loss with good macro F1
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_f1 = val_f1
        best_epoch = epoch
        torch.save(model.state_dict(), best_model_path)
        checkpoint_note = "(Optimal Checkpoint Saved)"
        patience_counter = 0
    else:
        patience_counter += 1

    print(f"Epoch [{epoch:2d}/{TOTAL_EPOCHS:2d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro F1: {val_f1:.4f} {checkpoint_note}")

    if patience_counter >= PATIENCE:
        print(f"\\nEarly stopping triggered at epoch {epoch}. Optimal checkpoint was at epoch {best_epoch} with Val Loss {best_val_loss:.4f}.")
        break

total_time = time.time() - t_start
print(f"\\nTraining completed in {total_time:.1f} seconds ({total_time / 60:.2f} minutes).")
""")

# CELL 10: Training History and Metric Visualization
add_code("""# Cell 9: Training History and Metric Visualization
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label="Train Loss", color="blue", lw=1.5)
plt.plot(history["val_loss"], label="Val Loss", color="darkorange", lw=1.5)
if best_epoch > 0:
    plt.axvline(best_epoch - 1, color="red", linestyle="--", alpha=0.7, label=f"Best Checkpoint (Ep {best_epoch})")
plt.title("Label Smoothing Loss Progression", fontsize=11, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot([a * 100 for a in history["val_acc"]], label="Val Accuracy (%)", color="green", lw=1.5)
plt.plot([f * 100 for f in history["val_f1"]], label="Val Macro F1 (%)", color="purple", lw=1.5)
plt.title("Validation Performance Metrics", fontsize=11, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Percentage (%)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves_compact.png", dpi=300)
plt.show()
print("Saved training curves plot to training_curves_compact.png")
""")

# CELL 11: Independent Test Dataset Evaluation
add_code("""# Cell 10: Independent Test Dataset Evaluation (Zero Data Leakage)
print("=== COMPACT MODEL TEST SET OBJECTIVE EVALUATION ===")
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

y_test_true, y_test_pred = [], []
with torch.no_grad():
    for x_b, y_b in test_loader:
        x_b, y_b = x_b.to(device), y_b.to(device)
        out = model(x_b)
        preds = torch.argmax(out, dim=1).cpu().numpy()
        y_test_pred.extend(preds)
        y_test_true.extend(y_b.cpu().numpy())

test_acc = accuracy_score(y_test_true, y_test_pred)
test_macro_f1 = f1_score(y_test_true, y_test_pred, average="macro", zero_division=0)
test_weighted_f1 = f1_score(y_test_true, y_test_pred, average="weighted", zero_division=0)

print(f"Test Accuracy    : {test_acc * 100:.2f}%")
print(f"Test Macro F1    : {test_macro_f1:.4f}")
print(f"Test Weighted F1 : {test_weighted_f1:.4f}")

print("\\nClassification Report Across 5 Classes:")
report = classification_report(y_test_true, y_test_pred, target_names=CLASSES, digits=4, zero_division=0)
print(report)

cm = confusion_matrix(y_test_true, y_test_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
plt.title(f"Confusion Matrix - CompactAudioCNN (Test Acc: {test_acc*100:.1f}%)", fontsize=11, fontweight="bold")
plt.colorbar()
tick_marks = np.arange(len(CLASSES))
plt.xticks(tick_marks, CLASSES, rotation=45)
plt.yticks(tick_marks, CLASSES)
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], "d"),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix_compact.png", dpi=300)
plt.show()
print("Saved confusion matrix to confusion_matrix_compact.png")
""")

# CELL 12: Test Set Inference Demonstration
add_code("""# Cell 11: Test Set Inference Demonstration
print("=== SAMPLE INFERENCE DEMONSTRATION ON UNSEEN TEST AUDIO ===")
model.eval()

inference_results = []
for idx, cat in enumerate(CLASSES):
    subset = df_test[df_test["category"] == cat]
    if len(subset) == 0:
        continue
    sample_row = subset.iloc[0]
    sample_dataset = CompactCryDataset(pd.DataFrame([sample_row]), DATASET_ROOT, is_train=False)
    tensor, true_label = sample_dataset[0]
    
    with torch.no_grad():
        out = model(tensor.unsqueeze(0).to(device))
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        pred_label = int(np.argmax(probs))
        confidence = float(probs[pred_label])
        
    inference_results.append({
        "Original Category": cat,
        "Predicted Category": IDX_TO_CLASS[pred_label],
        "Confidence (%)": f"{confidence * 100:.2f}%",
        "Result": "CORRECT" if pred_label == true_label else "INCORRECT"
    })

df_infer = pd.DataFrame(inference_results)
print(df_infer.to_string(index=False))
""")

# CELL 13: Model Checkpoint Export & Final Summary
add_code("""# Cell 12: Checkpoint Export and Final Summary
metadata = {
    "project": "Cry Baby Detection - Compact Audio CNN Regularized",
    "architecture": "CompactAudioCNN-4Block-LabelSmoothing-EarlyStopping",
    "target_hardware": "NVIDIA Tesla T4 GPU",
    "total_parameters": int(total_params),
    "sample_rate": TARGET_SR,
    "duration_sec": DURATION_SEC,
    "n_mels": N_MELS,
    "classes": CLASSES,
    "class_to_idx": CLASS_TO_IDX,
    "best_epoch": int(best_epoch),
    "best_val_loss": float(best_val_loss),
    "best_val_f1": float(best_val_f1),
    "test_accuracy": float(test_acc),
    "test_macro_f1": float(test_macro_f1),
    "test_weighted_f1": float(test_weighted_f1),
    "training_time_sec": float(total_time)
}

with open("compact_audio_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4, ensure_ascii=False)

print("=== EXPERIMENT ARTIFACTS VERIFICATION ===")
artifacts = [
    "best_compact_audio_classifier.pt",
    "compact_audio_metadata.json",
    "confusion_matrix_compact.png",
    "training_curves_compact.png",
    "data_inspection_samples.png"
]

for art in artifacts:
    if os.path.exists(art):
        size_kb = os.path.getsize(art) / 1024
        print(f"Artifact Verified: {art} ({size_kb:.1f} KB)")
    else:
        print(f"Warning: Expected artifact missing: {art}")

print("\\nCompact Audio CNN experiment completed successfully.")
""")

out_dir = "kaggle_compact_kernel"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "train_compact_audio_cry_classifier.ipynb")

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Successfully generated clean compact notebook at: {out_path}")
