"""
Script: scripts/build_eletriby_4class_notebook.py
Purpose:
Generate a clean, professional, emoji-free Kaggle notebook for training a Robust SOTA
Audio Classifier on the 4,871-sample Karim Eletriby dataset across 4 clinical classes:
burping, needs, physical_pain, scared.
Enhancements:
1. Procedural Harsh Acoustic Domain Randomization (Pink/White/Burst Noise, SNR -5dB to +18dB, Room Reverb FIR, Pre-Amp Clipping).
2. Balanced Benchmark Evaluation (50 Test / 50 Val per class) with 100% Zero Data Leakage.
3. Multi-Condition Stress-Test Evaluation across 5 acoustic environments.
Target Hardware: NVIDIA Tesla T4 GPU on account vuitang.
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
add_md("""# Robust Clinical Infant Cry Audio Classification (4 Classes)
### Acoustic Domain Randomization for Harsh Real-World Environments
- Dataset: karimeletriby/infant-cry-classification-4-classes (4,871 Clean Deduplicated Audio Files)
- Target Classes: burping (310), needs (3,103), physical_pain (774), scared (684)
- Splitting Protocol: Balanced Evaluation Benchmark (50 Test / 50 Val per class)
- Integrity Guarantee: 100% Zero Data Leakage (File-level disjoint partition)
- Harsh Domain Randomization: Multispectral Noise (Pink/White/Burst SNR -5dB to +18dB), Room Reverb FIR, Soft Clipping, Gain Perturbation
- Regularization: SpecAugment + Class-Balanced Loss + Label Smoothing (epsilon=0.1)
- Target Hardware: NVIDIA Tesla T4 GPU (Account: vuitang)
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
import torchvision.models as models
import torchaudio
import torchaudio.transforms as T
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score

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

# CELL 3: Dataset Direct Connection & File Inventory
add_code("""# Cell 2: Dataset Connection and Inventory
DATASET_ROOT = None
candidates = [
    "/kaggle/input/datasets/karimeletriby/infant-cry-classification-4-classes/audio",
    "/kaggle/input/infant-cry-classification-4-classes/audio",
    "/kaggle/input/datasets/karimeletriby/infant-cry-classification-4-classes",
    "/kaggle/input/infant-cry-classification-4-classes",
    "data/inspect_eletriby/audio"
]

for c in candidates:
    if os.path.exists(c) and os.path.exists(os.path.join(c, "burping")):
        DATASET_ROOT = c
        break

if DATASET_ROOT is None:
    for root, dirs, files in os.walk("/kaggle/input"):
        if "burping" in dirs and "needs" in dirs:
            DATASET_ROOT = root
            break

if DATASET_ROOT is None:
    print("DEBUG: Directory listing under /kaggle/input:")
    for root, dirs, files in os.walk("/kaggle/input"):
        print(f"  {root} -> dirs: {dirs[:5]}")
    raise FileNotFoundError("Could not find directory containing 'burping' and 'needs' in /kaggle/input.")

CLASSES = sorted([d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d)) and not d.startswith(".")])
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

all_records = []
for cat in CLASSES:
    cat_dir = os.path.join(DATASET_ROOT, cat)
    wavs = [os.path.join(cat_dir, f) for f in os.listdir(cat_dir) if f.lower().endswith(".wav")]
    for w in wavs:
        all_records.append({"filepath": w, "category": cat, "filename": os.path.basename(w)})

df_all = pd.DataFrame(all_records)
print("=== DATASET MOUNTING VERIFICATION ===")
print(f"Dataset Root Directory  : {DATASET_ROOT}")
print(f"Total Audio Files Found : {len(df_all)}")
print(f"Target Classes ({len(CLASSES)}): {CLASSES}")
print("\\nSamples breakdown by category:")
print(df_all["category"].value_counts())
""")

# CELL 4: Balanced Evaluation Benchmark Split & Zero Data Leakage Verification
add_code("""# Cell 3: Balanced Evaluation Benchmark Split and Zero Data Leakage
print("=== BALANCED BENCHMARK PARTITIONING (50 TEST / 50 VAL PER CLASS) ===")

TEST_PER_CLASS = 50
VAL_PER_CLASS = 50
MAX_TRAIN_NEEDS = 1000  # Cap the dominant needs class to keep training balanced

train_rows, val_rows, test_rows = [], [], []

for cat in CLASSES:
    cat_df = df_all[df_all["category"] == cat].sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    
    # 1. Test partition (strictly 50 unseen files)
    test_subset = cat_df.iloc[:TEST_PER_CLASS]
    test_rows.append(test_subset)
    
    # 2. Validation partition (strictly 50 unseen files)
    val_subset = cat_df.iloc[TEST_PER_CLASS : TEST_PER_CLASS + VAL_PER_CLASS]
    val_rows.append(val_subset)
    
    # 3. Training partition
    train_subset = cat_df.iloc[TEST_PER_CLASS + VAL_PER_CLASS :]
    if cat == "needs" and len(train_subset) > MAX_TRAIN_NEEDS:
        train_subset = train_subset.sample(n=MAX_TRAIN_NEEDS, random_state=SEED)
    train_rows.append(train_subset)

df_train = pd.concat(train_rows).reset_index(drop=True)
df_val = pd.concat(val_rows).reset_index(drop=True)
df_test = pd.concat(test_rows).reset_index(drop=True)

print(f"Train Set Count : {len(df_train)} files")
print(df_train["category"].value_counts())
print(f"\\nVal Set Count   : {len(df_val)} files (Balanced: {VAL_PER_CLASS} per class)")
print(f"Test Set Count  : {len(df_test)} files (Balanced: {TEST_PER_CLASS} per class)")

# Mathematical Zero Data Leakage Proof
train_files = set(df_train["filepath"].unique())
val_files = set(df_val["filepath"].unique())
test_files = set(df_test["filepath"].unique())

overlap_tr_val = train_files.intersection(val_files)
overlap_tr_te = train_files.intersection(test_files)
overlap_val_te = val_files.intersection(test_files)

print(f"\\nOverlap Train & Val  : {len(overlap_tr_val)}")
print(f"Overlap Train & Test : {len(overlap_tr_te)}")
print(f"Overlap Val & Test   : {len(overlap_val_te)}")

assert len(overlap_tr_val) == 0 and len(overlap_tr_te) == 0 and len(overlap_val_te) == 0, \\
    "Validation Failure: Data leakage detected across splits!"
print("Status: 100% Zero Data Leakage verified successfully.")
""")

# CELL 5: Waveform & Spectrogram Inspection
add_code("""# Cell 4: Data Visualization and Sample Inspection
TARGET_SR = 16000
DURATION_SEC = 5.0
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

plt.figure(figsize=(14, 6))
for idx, cat in enumerate(CLASSES):
    sample_path = df_train[df_train["category"] == cat].iloc[0]["filepath"]
    waveform, sr = torchaudio.load(sample_path)
    if sr != TARGET_SR:
        waveform = T.Resample(sr, TARGET_SR)(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if waveform.shape[1] > TARGET_SAMPLES:
        waveform = waveform[:, :TARGET_SAMPLES]
        
    mel = amp_to_db(mel_transform(waveform))[0].numpy()
    
    plt.subplot(2, 4, idx + 1)
    plt.plot(waveform[0].numpy(), color="navy", lw=0.5)
    plt.title(f"Waveform: {cat}", fontsize=10)
    plt.axis("off")
    
    plt.subplot(2, 4, idx + 5)
    plt.imshow(mel, aspect="auto", origin="lower", cmap="magma")
    plt.title(f"Log-Mel: {cat}", fontsize=10)
    plt.axis("off")

plt.tight_layout()
plt.savefig("data_inspection_samples.png", dpi=300)
plt.show()
print("Saved sample inspection plot to data_inspection_samples.png")
""")

# CELL 6: Harsh Acoustic Domain Randomization & Dataset Definition
add_code("""# Cell 5: Harsh Acoustic Domain Randomization and Dataset Definition
class HarshAcousticAugmentor:
    \"\"\"
    Procedural Acoustic Domain Randomization for Harsh Real-World Deployment.
    Generates:
    1. Pink Noise (1/f) - HVAC/Fan/Air Conditioner/Traffic rumble.
    2. White Noise (Gaussian) - Microphone electronic sensor noise.
    3. Transient Burst Noise - Random ambient impacts/shuffling.
    4. Room Reverberation FIR - Multi-tap decaying room reflections (RT60 ~ 0.2s - 0.5s).
    5. Microphone Pre-Amp Saturation / Soft Clipping (tanh).
    6. Random Gain Perturbation (-12dB to +6dB) - Distance variations.
    \"\"\"
    def __init__(self, sample_rate=16000):
        self.sr = sample_rate

    def generate_pink_noise(self, num_samples):
        white = torch.randn(1, num_samples)
        freqs = torch.fft.rfftfreq(num_samples)
        scale = 1.0 / torch.sqrt(torch.clamp(freqs, min=1.0 / num_samples))
        scale[0] = 0.0
        fft_white = torch.fft.rfft(white)
        fft_pink = fft_white * scale
        pink = torch.fft.irfft(fft_pink, n=num_samples)
        pink = pink / (pink.std() + 1e-7)
        return pink

    def apply_reverb(self, waveform, rt60=0.3):
        reverb_len = int(self.sr * rt60)
        t = torch.linspace(0, rt60, reverb_len)
        decay = torch.exp(-3.0 * t / max(rt60, 0.05))
        sparse_noise = torch.randn(1, reverb_len) * (torch.rand(1, reverb_len) > 0.85).float()
        rir = decay * sparse_noise
        rir[0, 0] = 1.0
        rir = rir / (rir.norm(p=2) + 1e-7)
        n_fft = waveform.shape[1] + reverb_len - 1
        wf_f = torch.fft.rfft(waveform, n=n_fft)
        rir_f = torch.fft.rfft(rir, n=n_fft)
        reverbed = torch.fft.irfft(wf_f * rir_f, n=n_fft)
        return reverbed[:, :waveform.shape[1]]

    def __call__(self, waveform, is_train=True, fixed_condition=None):
        if not is_train and fixed_condition is None:
            return waveform

        wf = waveform.clone()
        num_samples = wf.shape[1]

        # Evaluation mode with fixed stress-test conditions
        if fixed_condition is not None:
            if fixed_condition == "clean":
                return wf
            elif fixed_condition == "noise_snr15":
                noise = 0.7 * self.generate_pink_noise(num_samples) + 0.3 * torch.randn(1, num_samples)
                snr = 15.0
                scale = (wf.norm(p=2) / (noise.norm(p=2) + 1e-7)) * (10 ** (-snr / 20.0))
                return wf + scale * noise
            elif fixed_condition == "noise_snr10":
                noise = 0.7 * self.generate_pink_noise(num_samples) + 0.3 * torch.randn(1, num_samples)
                snr = 10.0
                scale = (wf.norm(p=2) / (noise.norm(p=2) + 1e-7)) * (10 ** (-snr / 20.0))
                return wf + scale * noise
            elif fixed_condition == "noise_snr5":
                noise = 0.7 * self.generate_pink_noise(num_samples) + 0.3 * torch.randn(1, num_samples)
                snr = 5.0
                scale = (wf.norm(p=2) / (noise.norm(p=2) + 1e-7)) * (10 ** (-snr / 20.0))
                return wf + scale * noise
            elif fixed_condition == "reverb":
                return self.apply_reverb(wf, rt60=0.4)

        # Training procedural domain randomization
        # 1. Gain / Distance variation (-12dB to +6dB)
        if random.random() > 0.3:
            gain_db = random.uniform(-12.0, 6.0)
            wf = wf * (10.0 ** (gain_db / 20.0))

        # 2. Room Reverberation (p=0.4)
        if random.random() > 0.6:
            rt60 = random.uniform(0.15, 0.45)
            wf = self.apply_reverb(wf, rt60=rt60)

        # 3. Multispectral Noise Injection (p=0.65) with SNR between -5dB and +18dB
        if random.random() > 0.35:
            noise_type = random.choice(["pink", "white", "hybrid"])
            if noise_type == "pink":
                noise = self.generate_pink_noise(num_samples)
            elif noise_type == "white":
                noise = torch.randn(1, num_samples)
            else:
                noise = 0.6 * self.generate_pink_noise(num_samples) + 0.4 * torch.randn(1, num_samples)
            
            # Transient burst noise (p=0.25)
            if random.random() > 0.75:
                burst_len = random.randint(int(0.05 * self.sr), int(0.3 * self.sr))
                burst_start = random.randint(0, max(0, num_samples - burst_len))
                noise[:, burst_start:burst_start+burst_len] += 3.0 * torch.randn(1, burst_len)

            snr_db = random.uniform(-5.0, 18.0)
            sig_p = wf.norm(p=2)
            noi_p = noise.norm(p=2) + 1e-7
            scale = (sig_p / noi_p) * (10.0 ** (-snr_db / 20.0))
            wf = wf + scale * noise

        # 4. Microphone Pre-Amp Saturation / Soft Clipping (p=0.35)
        if random.random() > 0.65:
            drive = random.uniform(1.2, 2.5)
            wf = torch.tanh(wf * drive)

        return wf

augmentor = HarshAcousticAugmentor(sample_rate=TARGET_SR)

class EletribyCryDataset(Dataset):
    def __init__(self, df, is_train=False, fixed_condition=None):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        self.fixed_condition = fixed_condition

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row["filepath"]
        
        waveform, sr = torchaudio.load(filepath)
        if sr != TARGET_SR:
            waveform = T.Resample(sr, TARGET_SR)(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        if waveform.shape[1] < TARGET_SAMPLES:
            repeats = int(np.ceil(TARGET_SAMPLES / waveform.shape[1]))
            waveform = waveform.repeat(1, repeats)[:, :TARGET_SAMPLES]
        elif waveform.shape[1] > TARGET_SAMPLES:
            if self.is_train and waveform.shape[1] > TARGET_SAMPLES + 1600:
                max_start = waveform.shape[1] - TARGET_SAMPLES
                start = random.randint(0, max_start)
                waveform = waveform[:, start : start + TARGET_SAMPLES]
            else:
                mid = waveform.shape[1] // 2
                half = TARGET_SAMPLES // 2
                start = max(0, mid - half)
                waveform = waveform[:, start : start + TARGET_SAMPLES]

        # Apply Harsh Acoustic Domain Randomization
        waveform = augmentor(waveform, is_train=self.is_train, fixed_condition=self.fixed_condition)

        # Extract 3-channel feature: Log-Mel + Delta + Delta2
        mel = mel_transform(waveform)
        log_mel = amp_to_db(mel)
        mean = log_mel.mean()
        std = log_mel.std() + 1e-6
        norm_mel = (log_mel - mean) / std

        delta_mel = torchaudio.functional.compute_deltas(norm_mel)
        delta2_mel = torchaudio.functional.compute_deltas(delta_mel)
        tensor_3ch = torch.cat([norm_mel, delta_mel, delta2_mel], dim=0)

        # SpecAugment on Train set
        if self.is_train:
            if random.random() > 0.4:
                f_mask = T.FrequencyMasking(freq_mask_param=12)
                tensor_3ch = f_mask(tensor_3ch)
            if random.random() > 0.4:
                t_mask = T.TimeMasking(time_mask_param=30)
                tensor_3ch = t_mask(tensor_3ch)

        label = CLASS_TO_IDX[row["category"]]
        return tensor_3ch, label

train_dataset = EletribyCryDataset(df_train, is_train=True)
val_dataset = EletribyCryDataset(df_val, is_train=False)
test_dataset = EletribyCryDataset(df_test, is_train=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True if torch.cuda.is_available() else False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True if torch.cuda.is_available() else False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True if torch.cuda.is_available() else False)

print("Harsh Acoustic Domain Randomization Pipeline initialized successfully:")
print(f"Train: {len(train_loader)} batches | Val: {len(val_loader)} batches | Test: {len(test_loader)} batches")
""")

# CELL 7: SOTA Transfer Learning Model Architecture (ResNet18)
add_code("""# Cell 6: SOTA Model Architecture (Pretrained ResNet18)
class AudioResNet18(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.4),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

model = AudioResNet18(num_classes=len(CLASSES), pretrained=True).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("=== SOTA MODEL INITIALIZATION ===")
print(f"Architecture         : ResNet18 (3-Channel Log-Mel Audio Backbone)")
print(f"Total Parameters     : {total_params:,}")
print(f"Trainable Parameters : {trainable_params:,}")
print(f"Target Classes ({len(CLASSES)}): {CLASSES}")
""")

# CELL 8: Class-Balanced Loss Configuration & Optimizer
add_code("""# Cell 7: Class-Balanced Loss Configuration and Optimizer
train_counts = df_train["category"].value_counts()
total_train = len(df_train)
class_weights = []
for c in CLASSES:
    count = train_counts.get(c, 1)
    w = total_train / (len(CLASSES) * count)
    class_weights.append(w)

weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
print("=== CLASS BALANCED WEIGHTS ===")
for c, w in zip(CLASSES, class_weights):
    print(f"  {c:<15}: weight = {w:.3f}")

criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.1)

TOTAL_EPOCHS = 25
PATIENCE = 6
INITIAL_LR = 3e-4
WEIGHT_DECAY = 1e-3

optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-6)

print(f"\\nTotal Epochs         : {TOTAL_EPOCHS}")
print(f"Early Stopping       : Patience = {PATIENCE} epochs on Validation Macro F1")
print(f"Initial LR           : {INITIAL_LR}")
print(f"Loss Function        : CrossEntropyLoss (Class-Weighted + Label Smoothing 0.1)")
""")

# CELL 9: Training Loop with Early Stopping on Val Macro F1
add_code("""# Cell 8: Model Training Execution with Early Stopping
print("=== STARTING TRAINING PIPELINE WITH HARSH DOMAIN RANDOMIZATION ===")

best_val_f1 = 0.0
best_val_acc = 0.0
best_epoch = 0
best_model_path = "best_eletriby_4class_audio_classifier.pt"
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
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_val_acc = val_acc
        best_epoch = epoch
        torch.save(model.state_dict(), best_model_path)
        checkpoint_note = "(Optimal Checkpoint Saved)"
        patience_counter = 0
    else:
        patience_counter += 1

    print(f"Epoch [{epoch:2d}/{TOTAL_EPOCHS:2d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro F1: {val_f1:.4f} {checkpoint_note}")

    if patience_counter >= PATIENCE:
        print(f"\\nEarly stopping triggered at epoch {epoch}. Optimal checkpoint was at epoch {best_epoch} with Val Macro F1 {best_val_f1:.4f}.")
        break

total_time = time.time() - t_start
print(f"\\nTraining completed in {total_time:.1f} seconds ({total_time / 60:.2f} minutes).")
""")

# CELL 10: Training History and Metric Visualization
add_code("""# Cell 9: Training History and Metric Visualization
epochs = list(range(1, len(history["train_loss"]) + 1))
xticks = [1] + [e for e in range(5, len(epochs) + 1, 5)]

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, history["train_loss"], label="Train Loss", color="blue", lw=1.5)
plt.plot(epochs, history["val_loss"], label="Val Loss", color="darkorange", lw=1.5)
if best_epoch > 0:
    plt.axvline(best_epoch, color="red", linestyle="--", alpha=0.7, label=f"Best Checkpoint (Ep {best_epoch})")
plt.title("Weighted Loss Progression", fontsize=11, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(xticks)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(epochs, [a * 100 for a in history["val_acc"]], label="Val Accuracy (%)", color="green", lw=1.5)
plt.plot(epochs, [f * 100 for f in history["val_f1"]], label="Val Macro F1 (%)", color="purple", lw=1.5)
plt.title("Validation Performance Across 4 Classes", fontsize=11, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Percentage (%)")
plt.xticks(xticks)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves_eletriby.png", dpi=300)
plt.show()
print("Saved training curves plot to training_curves_eletriby.png")
""")

# CELL 11: Multi-Condition Stress-Test Evaluation (Zero Data Leakage)
add_code("""# Cell 10: Multi-Condition Stress-Test Evaluation (Zero Data Leakage)
print("=== MULTI-CONDITION STRESS-TEST EVALUATION (200 TEST SAMPLES) ===")
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

conditions = [
    ("clean", "1. Clean Baseline (Ideal Audio)"),
    ("noise_snr15", "2. Moderate Room Noise (SNR = 15 dB)"),
    ("noise_snr10", "3. Medium Fan/AC Noise (SNR = 10 dB)"),
    ("noise_snr5", "4. Harsh Loud Noise / TV (SNR = 5 dB)"),
    ("reverb", "5. Room Reverberation (RT60 ~ 0.4s)")
]

stress_results = []
clean_preds, clean_trues = [], []

for cond_key, cond_name in conditions:
    cond_test_dataset = EletribyCryDataset(df_test, is_train=False, fixed_condition=cond_key)
    cond_test_loader = DataLoader(cond_test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    y_test_true, y_test_pred = [], []
    with torch.no_grad():
        for x_b, y_b in cond_test_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            out = model(x_b)
            preds = torch.argmax(out, dim=1).cpu().numpy()
            y_test_pred.extend(preds)
            y_test_true.extend(y_b.cpu().numpy())

    acc = accuracy_score(y_test_true, y_test_pred)
    macro_p = precision_score(y_test_true, y_test_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_test_true, y_test_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test_true, y_test_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test_true, y_test_pred, average="weighted", zero_division=0)
    
    if cond_key == "clean":
        clean_preds = y_test_pred
        clean_trues = y_test_true
        
    stress_results.append({
        "Condition": cond_name,
        "Accuracy (%)": acc * 100,
        "Macro Precision (%)": macro_p * 100,
        "Macro Recall (%)": macro_r * 100,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1
    })

df_stress = pd.DataFrame(stress_results)
print("\\n--- STRESS-TEST BENCHMARK ACROSS 5 ACOUSTIC ENVIRONMENTS ---")
print(df_stress.to_string(index=False))

print("\\n--- DETAILED CLASSIFICATION REPORT (CLEAN BENCHMARK - 4 CLINICAL CLASSES) ---")
report = classification_report(clean_trues, clean_preds, target_names=CLASSES, digits=4, zero_division=0)
print(report)

# Visualization of Stress-Test Robustness
plt.figure(figsize=(10, 5))
labels = [c[0].replace("_", " ").upper() for c in conditions]
accuracies = [r["Accuracy (%)"] for r in stress_results]
f1_scores = [r["Macro F1"] * 100 for r in stress_results]

x = np.arange(len(labels))
width = 0.35

plt.bar(x - width/2, accuracies, width, label="Test Accuracy (%)", color="#1f77b4")
plt.bar(x + width/2, f1_scores, width, label="Macro F1 (%)", color="#2ca02c")

plt.ylabel("Percentage (%)", fontsize=11)
plt.title("Model Robustness Under Real-World Acoustic Degradations", fontsize=12, fontweight="bold")
plt.xticks(x, labels, fontsize=10)
plt.ylim(0, 105)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)

for i in range(len(labels)):
    plt.text(i - width/2, accuracies[i] + 1.5, f"{accuracies[i]:.1f}%", ha="center", fontsize=9, fontweight="bold")
    plt.text(i + width/2, f1_scores[i] + 1.5, f"{f1_scores[i]:.1f}%", ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("stress_test_comparison.png", dpi=300)
plt.show()
print("Saved stress test comparison plot to stress_test_comparison.png")

# Confusion Matrix on Clean Benchmark
cm = confusion_matrix(clean_trues, clean_preds)
clean_acc = accuracy_score(clean_trues, clean_preds)
plt.figure(figsize=(7, 6))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix - Clean Benchmark (Acc: {clean_acc*100:.1f}%)", fontsize=11, fontweight="bold")
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
plt.savefig("confusion_matrix_eletriby.png", dpi=300)
plt.show()
print("Saved confusion matrix to confusion_matrix_eletriby.png")
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
    sample_dataset = EletribyCryDataset(pd.DataFrame([sample_row]), is_train=False)
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
clean_result = next(r for r in stress_results if "Clean" in r["Condition"])
metadata = {
    "project": "Cry Baby Detection - Robust Clinical 4-Class Audio Classifier",
    "dataset": "karimeletriby/infant-cry-classification-4-classes",
    "architecture": "ResNet18-3Ch-LogMel-HarshDomainRandomized",
    "target_hardware": "NVIDIA Tesla T4 GPU",
    "total_parameters": int(total_params),
    "sample_rate": TARGET_SR,
    "duration_sec": DURATION_SEC,
    "n_mels": N_MELS,
    "classes": CLASSES,
    "class_to_idx": CLASS_TO_IDX,
    "best_epoch": int(best_epoch),
    "best_val_f1": float(best_val_f1),
    "best_val_acc": float(best_val_acc),
    "clean_test_accuracy": float(clean_result["Accuracy (%)"]),
    "clean_test_macro_precision": float(clean_result.get("Macro Precision (%)", 0.0)),
    "clean_test_macro_recall": float(clean_result.get("Macro Recall (%)", 0.0)),
    "clean_test_macro_f1": float(clean_result["Macro F1"]),
    "stress_test_benchmark": stress_results,
    "training_time_sec": float(total_time)
}

with open("eletriby_4class_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4, ensure_ascii=False)

print("=== EXPERIMENT ARTIFACTS VERIFICATION ===")
artifacts = [
    "best_eletriby_4class_audio_classifier.pt",
    "eletriby_4class_metadata.json",
    "confusion_matrix_eletriby.png",
    "training_curves_eletriby.png",
    "stress_test_comparison.png",
    "data_inspection_samples.png"
]

for art in artifacts:
    if os.path.exists(art):
        size_kb = os.path.getsize(art) / 1024
        print(f"Artifact Verified: {art} ({size_kb:.1f} KB)")
    else:
        print(f"Warning: Expected artifact missing: {art}")

print("\\nRobust Clinical 4-Class Audio Classifier experiment completed successfully.")
""")

out_dir = "kaggle_eletriby_kernel"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "train_eletriby_4class_audio_classifier.ipynb")

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Successfully generated clean robust 4-class notebook at: {out_path}")
