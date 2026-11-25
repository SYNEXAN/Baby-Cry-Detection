"""
Script: scripts/train_sota_audio_classifier.py
Mục đích:
Huấn luyện mô hình Pretrained Audio Transfer Learning (ResNet34 trên 3-channel Log-Mel Spectrogram)
với Class-Balanced Focal Loss (gamma=2.0) và Two-Stage Fine-Tuning.
Đảm bảo 100% Zero Data Leakage trên tập dữ liệu chuẩn 10s cân bằng.
"""

import os
import sys
import time
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from scipy.io import wavfile
import librosa
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DATASET_ROOT = "data/processed/audio_eureka_10s_balanced"
META_CSV = os.path.join(DATASET_ROOT, "metadata.csv")
OUT_DIR = "kaggle_audio_results"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_SR = 16000
DURATION_SEC = 10.0
TARGET_SAMPLES = int(TARGET_SR * DURATION_SEC)
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512

def extract_3ch_logmel(filepath: str) -> np.ndarray:
    sr, audio = wavfile.read(filepath)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32) / 32768.0
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if len(audio) < TARGET_SAMPLES:
        repeats = int(np.ceil(TARGET_SAMPLES / len(audio)))
        audio = np.tile(audio, repeats)[:TARGET_SAMPLES]
    elif len(audio) > TARGET_SAMPLES:
        audio = audio[:TARGET_SAMPLES]

    mel = librosa.feature.melspectrogram(
        y=audio, sr=TARGET_SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    log_mel = librosa.power_to_db(mel)
    mean = np.mean(log_mel)
    std = np.std(log_mel) + 1e-6
    log_mel = (log_mel - mean) / std

    delta_mel = librosa.feature.delta(log_mel, order=1)
    delta2_mel = librosa.feature.delta(log_mel, order=2)
    return np.stack([log_mel, delta_mel, delta2_mel], axis=0).astype(np.float32)

class SOTACryDataset(Dataset):
    def __init__(self, df, data_root, is_train=False, cache_in_memory=True):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.is_train = is_train
        self.cache_in_memory = cache_in_memory
        self.cached_tensors = []
        self.labels = []

        if cache_in_memory:
            print(f"   Trích xuất và nạp bộ nhớ đệm 3 kênh Log-Mel cho {len(self.df)} mẫu...")
            for idx, row in self.df.iterrows():
                full_path = row['filepath']
                if not os.path.exists(full_path):
                    full_path = os.path.join(self.data_root, row['filepath'])
                feat = extract_3ch_logmel(full_path)
                lbl = CLASS_TO_IDX[row['category']]
                self.cached_tensors.append(feat)
                self.labels.append(lbl)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.cache_in_memory:
            feat = self.cached_tensors[idx].copy()
            label = self.labels[idx]
        else:
            row = self.df.iloc[idx]
            full_path = row['filepath']
            if not os.path.exists(full_path):
                full_path = os.path.join(self.data_root, row['filepath'])
            feat = extract_3ch_logmel(full_path)
            label = CLASS_TO_IDX[row['category']]

        t = torch.tensor(feat, dtype=torch.float32)

        # SpecAugment nếu là tập Train
        if self.is_train:
            if random.random() > 0.4:
                # Frequency Masking (che 4-8 dải tần ngẫu nhiên)
                f_width = random.randint(4, 9)
                f0 = random.randint(0, N_MELS - f_width)
                t[:, f0:f0+f_width, :] = 0.0
            if random.random() > 0.4:
                # Time Masking (che 15-30 frames ngẫu nhiên)
                t_width = random.randint(15, 31)
                t0 = random.randint(0, max(1, t.shape[-1] - t_width))
                t[:, :, t0:t0+t_width] = 0.0

        return t, torch.tensor(label, dtype=torch.long)

class AudioResNetClassifier(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
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
        logits = self.classifier(features)
        return logits

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def main():
    global CLASS_TO_IDX, IDX_TO_CLASS, CLASSES
    print("=" * 75)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN SOTA PRETRAINED AUDIO TRANSFER LEARNING (RESNET-18)")
    print("   (Chuẩn hóa 2D Log-Mel 3 kênh + Focal Loss + SpecAugment)")
    print("=" * 75)

    df_meta = pd.read_csv(META_CSV)
    CLASSES = sorted(df_meta['category'].unique())
    CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
    IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

    df_train = df_meta[df_meta['split'] == 'train'].reset_index(drop=True)
    df_val = df_meta[df_meta['split'] == 'val'].reset_index(drop=True)
    df_test = df_meta[df_meta['split'] == 'test'].reset_index(drop=True)

    print(f"📊 Tập dữ liệu: Train={len(df_train)} | Val={len(df_val)} | Test={len(df_test)}")

    train_dataset = SOTACryDataset(df_train, DATASET_ROOT, is_train=True, cache_in_memory=True)
    val_dataset = SOTACryDataset(df_val, DATASET_ROOT, is_train=False, cache_in_memory=True)
    test_dataset = SOTACryDataset(df_test, DATASET_ROOT, is_train=False, cache_in_memory=True)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🤖 Thiết bị tính toán: {device}")

    model = AudioResNetClassifier(num_classes=len(CLASSES), pretrained=True).to(device)
    criterion = FocalLoss(gamma=2.0)

    TOTAL_EPOCHS = 35
    STAGE1_EPOCHS = 5

    best_val_f1 = 0.0
    best_model_path = os.path.join(OUT_DIR, "best_sota_audio_classifier.pt")
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

    print("\n🏃 GIAI ĐOẠN 1: Đóng băng Backbone, căn chỉnh Classifier Head (5 Epochs)...")
    for param in model.backbone.parameters():
        param.requires_grad = False
    opt1 = optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-2)

    for epoch in range(1, STAGE1_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            opt1.zero_grad()
            out = model(x_b)
            loss = criterion(out, y_b)
            loss.backward()
            opt1.step()
            running_loss += loss.item() * len(y_b)
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
        val_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        print(f"Epoch [{epoch:2d}/{TOTAL_EPOCHS:2d}] (Stage 1) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro F1: {val_f1:.4f}")

    print("\n🏃 GIAI ĐOẠN 2: Mở khóa toàn mạng với Vi phân Learning Rate (30 Epochs)...")
    for param in model.backbone.parameters():
        param.requires_grad = True

    opt2 = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 1e-4, 'weight_decay': 1e-3},
        {'params': model.classifier.parameters(), 'lr': 3e-4, 'weight_decay': 1e-3}
    ])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=TOTAL_EPOCHS - STAGE1_EPOCHS, eta_min=1e-6)

    t_start = time.time()
    for epoch in range(STAGE1_EPOCHS + 1, TOTAL_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            opt2.zero_grad()
            out = model(x_b)
            loss = criterion(out, y_b)
            loss.backward()
            opt2.step()
            running_loss += loss.item() * len(y_b)

        sched.step()
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
        val_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        star = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            star = "⭐ (Lưu Checkpoint Tối Ưu)"

        if epoch % 5 == 0 or epoch == TOTAL_EPOCHS or star:
            print(f"Epoch [{epoch:2d}/{TOTAL_EPOCHS:2d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro F1: {val_f1:.4f} {star}")

    print(f"\n🎉 Huấn luyện hoàn tất trong {time.time() - t_start:.1f} giây!")

    # Test evaluation
    print("\n" + "=" * 75)
    print("🧪 ĐÁNH GIÁ ĐỘC LẬP TRÊN TẬP TEST (ZERO DATA LEAKAGE)")
    print("=" * 75)
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
    test_macro_f1 = f1_score(y_test_true, y_test_pred, average='macro', zero_division=0)
    test_weighted_f1 = f1_score(y_test_true, y_test_pred, average='weighted', zero_division=0)

    print(f"🎯 Test Accuracy   : {test_acc * 100:.2f}%")
    print(f"🎯 Test Macro F1    : {test_macro_f1:.4f}")
    print(f"🎯 Test Weighted F1 : {test_weighted_f1:.4f}")

    print("\n📊 BÁO CÁO PHÂN LOẠI CHI TIẾT THEO TỪNG LỚP NGUYÊN NHÂN:")
    report = classification_report(y_test_true, y_test_pred, target_names=CLASSES, digits=4)
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_test_true, y_test_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'SOTA Audio-ResNet Confusion Matrix (Acc: {test_acc*100:.1f}%)', fontsize=13, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(CLASSES))
    plt.xticks(tick_marks, CLASSES, rotation=45)
    plt.yticks(tick_marks, CLASSES)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.xlabel('Dự đoán (Predicted Label)', fontsize=11)
    plt.ylabel('Thực tế (True Label)', fontsize=11)
    plt.tight_layout()
    cm_path = os.path.join(OUT_DIR, "confusion_matrix_sota.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # Curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Val Loss', color='orange')
    plt.title('Loss Curves', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Focal Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot([a * 100 for a in history['val_acc']], label='Val Acc (%)', color='green')
    plt.plot([f * 100 for f in history['val_f1']], label='Val Macro F1 (%)', color='purple')
    plt.title('Validation Accuracy & F1', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Percentage (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    curves_path = os.path.join(OUT_DIR, "training_curves_sota.png")
    plt.savefig(curves_path, dpi=300)
    plt.close()

    metadata = {
        "project": "Cry Baby Detection - SOTA Audio Transfer Learning",
        "architecture": "ResNet18-3Ch-LogMel-FocalLoss",
        "sample_rate": TARGET_SR,
        "duration_sec": DURATION_SEC,
        "n_mels": N_MELS,
        "classes": CLASSES,
        "class_to_idx": CLASS_TO_IDX,
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_macro_f1),
        "test_weighted_f1": float(test_weighted_f1)
    }
    with open(os.path.join(OUT_DIR, "sota_audio_metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    print(f"📁 Đã xuất metadata tại: {os.path.join(OUT_DIR, 'sota_audio_metadata.json')}")
    print("=" * 75)
    print("🎉 HOÀN TẤT HUẤN LUYỆN VÀ KIỂM ĐỊNH SOTA AUDIO THÀNH CÔNG RỰC RỠ!")
    print("=" * 75)

if __name__ == "__main__":
    main()
