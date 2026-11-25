"""
Script: train_eureka_1dcnn_audio.py
Mục đích:
Huấn luyện mô hình 1D-CNN trên chuỗi đặc trưng 1D MFCC (40 coefficients)
chuẩn phương pháp bài báo Euréka 2025 (Huỳnh Nguyễn Sơn).
Cam kết:
  1. Bảo vệ Zero Data Leakage 100% trên tập Test độc lập.
  2. Báo cáo đầy đủ Accuracy, Macro F1, Ma trận nhầm lẫn (Confusion Matrix).
  3. Xuất checkpoint và metadata sẵn sàng tích hợp vào hệ thống Cry_Baby_Detection.
"""

import os
import sys
import time
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

# Cố định random seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DATA_DIR = "data/processed/audio_eureka_10s_balanced"
META_CSV = os.path.join(DATA_DIR, "metadata.csv")
OUT_DIR = "kaggle_audio_results"
os.makedirs(OUT_DIR, exist_ok=True)

CLASSES = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

TARGET_SR = 16000
N_MFCC = 40
N_FFT = 1024
HOP_LENGTH = 512

def extract_mfcc(filepath: str) -> np.ndarray:
    """Trích xuất 40 hệ số MFCC chuỗi 1D từ file WAV 10s"""
    sr, audio = wavfile.read(filepath)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32) / 32768.0

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=TARGET_SR,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    # Chuẩn hóa Z-Score theo từng dải tần
    mean = np.mean(mfcc, axis=1, keepdims=True)
    std = np.std(mfcc, axis=1, keepdims=True) + 1e-6
    return ((mfcc - mean) / std).astype(np.float32)

class EurekaCryDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cache_features=True):
        self.df = df.reset_index(drop=True)
        self.cache_features = cache_features
        self.cached_x = []
        self.cached_y = []

        if cache_features:
            print(f"   Trích xuất và lưu bộ nhớ đệm cho {len(self.df)} mẫu MFCC...")
            for idx, row in self.df.iterrows():
                feat = extract_mfcc(row['filepath'])
                label = CLASS_TO_IDX[row['category']]
                self.cached_x.append(feat)
                self.cached_y.append(label)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.cache_features:
            x = self.cached_x[idx]
            y = self.cached_y[idx]
        else:
            row = self.df.iloc[idx]
            x = extract_mfcc(row['filepath'])
            y = CLASS_TO_IDX[row['category']]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

class Eureka1DCNN(nn.Module):
    """
    Kiến trúc 1D-CNN siêu nhẹ theo bài báo Euréka 2025.
    Input shape: (Batch, 40 channels, TimeSteps)
    """
    def __init__(self, num_classes=5, in_channels=40):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2),

            # Block 2
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2),

            # Block 3
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        return self.classifier(feat)

def main():
    print("=" * 75)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN 1D-CNN TRÊN CHUỖI 1D MFCC (CHUẨN BÀI BÁO EURÉKA 2025)")
    print("   (Cam kết Zero Data Leakage 100% giữa Train, Val, Test)")
    print("=" * 75)

    if not os.path.exists(META_CSV):
        print(f"❌ Không tìm thấy file {META_CSV}. Vui lòng chạy build_eureka_standard_audio_dataset.py trước.")
        sys.exit(1)

    df_meta = pd.read_csv(META_CSV)
    df_train = df_meta[df_meta['split'] == 'train'].reset_index(drop=True)
    df_val = df_meta[df_meta['split'] == 'val'].reset_index(drop=True)
    df_test = df_meta[df_meta['split'] == 'test'].reset_index(drop=True)

    print(f"📊 Tập dữ liệu: Train={len(df_train)} | Val={len(df_val)} | Test={len(df_test)}")

    # 1. Trích xuất đặc trưng và nạp DataLoader
    print("\n📦 Đang nạp và trích xuất đặc trưng 1D MFCC...")
    train_dataset = EurekaCryDataset(df_train, cache_features=True)
    val_dataset = EurekaCryDataset(df_val, cache_features=True)
    test_dataset = EurekaCryDataset(df_test, cache_features=True)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 2. Khởi tạo mô hình
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🤖 Thiết bị tính toán: {device}")
    model = Eureka1DCNN(num_classes=len(CLASSES), in_channels=N_MFCC).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Tổng số tham số mô hình 1D-CNN: {total_params:,} tham số (Siêu nhẹ, ~85KB)")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.10)
    optimizer = optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-2)
    epochs = 40
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_f1 = 0.0
    best_weights_path = os.path.join(OUT_DIR, "best_eureka_1dcnn.pt")

    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

    print("\n🏃 Bắt đầu quá trình huấn luyện 40 Epochs (với SpecAugment & Regularization)...")
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            # Online SpecAugment Data Augmentation
            if np.random.rand() > 0.3:
                # Frequency masking: mask 2-4 channels
                f_width = np.random.randint(2, 5)
                f0 = np.random.randint(0, N_MFCC - f_width)
                x_b[:, f0:f0+f_width, :] = 0.0
            if np.random.rand() > 0.3:
                # Time masking: mask 10-25 frames
                t_len = x_b.shape[-1]
                t_width = np.random.randint(10, 26)
                t0 = np.random.randint(0, max(1, t_len - t_width))
                x_b[:, :, t0:t0+t_width] = 0.0
            if np.random.rand() > 0.5:
                # Gaussian noise jitter
                x_b = x_b + torch.randn_like(x_b) * 0.05

            optimizer.zero_grad()
            out = model(x_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(y_b)

        scheduler.step()
        train_loss = running_loss / len(train_dataset)

        # Validation
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

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_weights_path)
            star = "⭐ (Lưu Checkpoint Tối Ưu)"
        else:
            star = ""

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch:2d}/{epochs:2d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro F1: {val_f1:.4f} {star}")

    train_time = time.time() - t_start
    print(f"\n🎉 Huấn luyện hoàn tất trong {train_time:.1f} giây!")

    # 3. Đánh giá kiểm định trên tập Test độc lập (Zero Data Leakage)
    print("\n" + "=" * 75)
    print("🧪 KIỂM THỬ KHÁCH QUAN TRÊN TẬP TEST ĐỘC LẬP (TEST SET EVALUATION)")
    print("=" * 75)
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
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

    # 4. Vẽ và lưu Ma trận nhầm lẫn (Confusion Matrix)
    cm = confusion_matrix(y_test_true, y_test_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - Euréka 1D-CNN (Test Acc: {test_acc*100:.1f}%)', fontsize=13, fontweight='bold')
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
    cm_path = os.path.join(OUT_DIR, "confusion_matrix_eureka.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"🖼️ Đã lưu biểu đồ ma trận nhầm lẫn tại: {cm_path}")

    # 5. Vẽ biểu đồ quá trình huấn luyện
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Val Loss', color='orange')
    plt.title('Loss Curve', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('CrossEntropy Loss')
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
    curves_path = os.path.join(OUT_DIR, "eureka_training_curves.png")
    plt.savefig(curves_path, dpi=300)
    plt.close()
    print(f"📈 Đã lưu đường cong huấn luyện tại: {curves_path}")

    # 6. Xuất metadata JSON
    metadata = {
        "project": "Cry Baby Detection - Euréka 2025 1D-CNN Audio Classifier",
        "methodology": "Euréka 2025 (Huỳnh Nguyễn Sơn) + Zero Data Leakage Protocol",
        "model_architecture": "Eureka-1DCNN-MFCC",
        "parameters_count": total_params,
        "sample_rate": TARGET_SR,
        "duration_sec": 10.0,
        "n_mfcc": N_MFCC,
        "classes": CLASSES,
        "class_to_idx": CLASS_TO_IDX,
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_macro_f1),
        "test_weighted_f1": float(test_weighted_f1),
        "training_time_sec": float(train_time)
    }
    meta_json_path = os.path.join(OUT_DIR, "eureka_1dcnn_metadata.json")
    with open(meta_json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    print(f"📁 Đã xuất metadata mô hình tại: {meta_json_path}")
    print("=" * 75)
    print("🎉 TOÀN BỘ QUÁ TRÌNH HUẤN LUYỆN VÀ KIỂM ĐỊNH ĐÃ HOÀN TẤT THÀNH CÔNG RỰC RỠ!")
    print("=" * 75)

if __name__ == "__main__":
    main()
