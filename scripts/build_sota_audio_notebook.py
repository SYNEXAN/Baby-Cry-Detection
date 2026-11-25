"""
Script: scripts/build_sota_audio_notebook.py
Mục đích:
Tạo notebook huấn luyện chuẩn SOTA Audio Transfer Learning (Audio-ResNet34 trên 3-channel Log-Mel)
đáp ứng 100% tiêu chuẩn Kaggle Training Workflow Policy.
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

# 1. Header
add_md("""# 🎧 SOTA Infant Cry Audio Classification (5 Causes)
### Pretrained Audio Transfer Learning (3-Channel Log-Mel Spectrogram + ResNet34)
**Dataset:** `thittn/donateacry-eureka-10s-balanced` (10s Audio, 5 Classes, Zero Data Leakage)
**Authors / Research:** SYNEXAN AI LABS & Cry Baby Detection Project
""")

# 2. Environment & GPU Verification
add_code("""import os, sys, time, json, random
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

# Seed setting
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print('=' * 65)
print('🚀 GPU & ENVIRONMENT VALIDATION')
print('=' * 65)
print(f'PyTorch Version  : {torch.__version__}')
print(f'CUDA Available   : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Device Name  : {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory Total : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
    device = torch.device('cuda:0')
else:
    print('⚠️ WARNING: No GPU detected, fallback to CPU')
    device = torch.device('cpu')
""")

# 3. Dataset Connection & Validation
add_code("""print('=' * 65)
print('📦 DATASET MOUNTING & VALIDATION')
print('=' * 65)

possible_paths = [
    '/kaggle/input/donateacry-eureka-10s-balanced',
    '/kaggle/input/datasets/thittn/donateacry-eureka-10s-balanced',
    'data/processed/audio_eureka_10s_balanced'
]

DATASET_ROOT = None
for p in possible_paths:
    if os.path.exists(p):
        DATASET_ROOT = p
        break

if DATASET_ROOT is None:
    for root, dirs, files in os.walk('/kaggle/input'):
        if 'metadata.csv' in files:
            DATASET_ROOT = root
            break

print(f'✅ Dataset Root Path: {DATASET_ROOT}')
assert DATASET_ROOT is not None, '❌ Không tìm thấy dataset!'

META_CSV = os.path.join(DATASET_ROOT, 'metadata.csv')
assert os.path.exists(META_CSV), f'❌ Không tìm thấy file {META_CSV}'

df_meta = pd.read_csv(META_CSV)
print(f'📊 Tổng số mẫu: {len(df_meta)}')
print(df_meta.groupby(['split', 'category']).size())

CLASSES = sorted(df_meta['category'].unique())
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
print(f'🎯 5 Classes: {CLASSES}')
""")

# 4. Feature Extraction & Dataset Class
add_code("""# Cấu hình âm học
TARGET_SR = 16000
DURATION_SEC = 10.0
TARGET_SAMPLES = int(TARGET_SR * DURATION_SEC)
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512

# Khởi tạo MelSpectrogram transform từ torchaudio
mel_transform = T.MelSpectrogram(
    sample_rate=TARGET_SR,
    n_fft=N_FFT,
    win_length=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    power=2.0
)
amp_to_db = T.AmplitudeToDB()

class SOTACryDataset(Dataset):
    def __init__(self, df, data_root, is_train=False):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = str(row['filepath']).replace('\\\\', '/').replace('\\', '/')
        fname = os.path.basename(rel_path)
        split = row['split']
        cat = row['category']
        
        # Thử các đường dẫn tương thích Linux Kaggle và Windows
        candidates = [
            os.path.join(self.data_root, split, cat, fname),
            os.path.join(self.data_root, split, split, cat, fname),
            os.path.join(self.data_root, cat, fname),
            os.path.join(self.data_root, rel_path)
        ]
        full_path = None
        for c in candidates:
            if os.path.exists(c):
                full_path = c
                break
        if full_path is None:
            full_path = candidates[0]
        
        waveform, sr = torchaudio.load(full_path)
        if sr != TARGET_SR:
            resampler = T.Resample(sr, TARGET_SR)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Chuẩn hóa độ dài 10s
        if waveform.shape[1] < TARGET_SAMPLES:
            repeats = int(np.ceil(TARGET_SAMPLES / waveform.shape[1]))
            waveform = waveform.repeat(1, repeats)[:, :TARGET_SAMPLES]
        elif waveform.shape[1] > TARGET_SAMPLES:
            waveform = waveform[:, :TARGET_SAMPLES]

        # 1. Kênh 1: Log-Mel Spectrogram
        mel = mel_transform(waveform)
        log_mel = amp_to_db(mel) # shape: (1, 128, 313)
        
        # Chuẩn hóa Z-Score
        mean = log_mel.mean()
        std = log_mel.std() + 1e-6
        log_mel = (log_mel - mean) / std

        # 2. Kênh 2 & 3: Delta & Delta-Delta Mel
        delta_mel = torchaudio.functional.compute_deltas(log_mel)
        delta2_mel = torchaudio.functional.compute_deltas(delta_mel)

        # Gộp thành tensor 3 kênh: (3, 128, 313)
        tensor_3ch = torch.cat([log_mel, delta_mel, delta2_mel], dim=0)

        # SpecAugment nếu là tập Train
        if self.is_train:
            if random.random() > 0.4:
                f_mask = T.FrequencyMasking(freq_mask_param=8)
                tensor_3ch = f_mask(tensor_3ch)
            if random.random() > 0.4:
                t_mask = T.TimeMasking(time_mask_param=30)
                tensor_3ch = t_mask(tensor_3ch)

        label = CLASS_TO_IDX[row['category']]
        return tensor_3ch, label

# Tạo Datasets & DataLoaders
df_train = df_meta[df_meta['split'] == 'train'].reset_index(drop=True)
df_val = df_meta[df_meta['split'] == 'val'].reset_index(drop=True)
df_test = df_meta[df_meta['split'] == 'test'].reset_index(drop=True)

train_dataset = SOTACryDataset(df_train, DATASET_ROOT, is_train=True)
val_dataset = SOTACryDataset(df_val, DATASET_ROOT, is_train=False)
test_dataset = SOTACryDataset(df_test, DATASET_ROOT, is_train=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=False)

print(f'✅ DataLoaders sẵn sàng: Train={len(train_dataset)} | Val={len(val_dataset)} | Test={len(test_dataset)}')
""")

# 5. Model Architecture & Class-Balanced Focal Loss
add_code("""class AudioResNetClassifier(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet34(weights=weights)
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
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

model = AudioResNetClassifier(num_classes=len(CLASSES), pretrained=True).to(device)
criterion = FocalLoss(gamma=2.0)
print(f'🧠 Khởi tạo mô hình Audio-ResNet34 thành công trên {device}')
""")

# 6. Two-Stage Training
add_code("""print('=' * 65)
print('🏃 BẮT ĐẦU HUẤN LUYỆN HAI GIAI ĐOẠN (TWO-STAGE TRAINING)')
print('=' * 65)

TOTAL_EPOCHS = 35
STAGE1_EPOCHS = 5

print('>>> GIAI ĐOẠN 1: Đóng băng backbone, căn chỉnh Classifier Head (5 Epochs)...')
for param in model.backbone.parameters():
    param.requires_grad = False

optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-2)

best_val_f1 = 0.0
best_model_path = 'best_sota_audio_classifier.pt'
history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

for epoch in range(1, STAGE1_EPOCHS + 1):
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
    print(f'Epoch [{epoch:2d}/{TOTAL_EPOCHS:2d}] (Stage 1) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro F1: {val_f1:.4f}')

print('\\n>>> GIAI ĐOẠN 2: Mở khóa toàn mạng với Vi phân Learning Rate (30 Epochs)...')
for param in model.backbone.parameters():
    param.requires_grad = True

optimizer = optim.AdamW([
    {'params': model.backbone.parameters(), 'lr': 1e-4, 'weight_decay': 1e-3},
    {'params': model.classifier.parameters(), 'lr': 3e-4, 'weight_decay': 1e-3}
])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS - STAGE1_EPOCHS, eta_min=1e-6)

t_start = time.time()
for epoch in range(STAGE1_EPOCHS + 1, TOTAL_EPOCHS + 1):
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
    val_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_f1'].append(val_f1)
    
    star = ''
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), best_model_path)
        star = '⭐ (Lưu Checkpoint Tối Ưu)'
        
    if epoch % 5 == 0 or epoch == TOTAL_EPOCHS or star:
        print(f'Epoch [{epoch:2d}/{TOTAL_EPOCHS:2d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro F1: {val_f1:.4f} {star}')

print(f'🎉 Huấn luyện hoàn tất trong {time.time() - t_start:.1f} giây!')
""")

# 7. Independent Test Set Evaluation & Metrics
add_code("""print('=' * 65)
print('🧪 ĐÁNH GIÁ ĐỘC LẬP TRÊN TẬP TEST (ZERO DATA LEAKAGE)')
print('=' * 65)

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

print(f'🎯 Test Accuracy   : {test_acc * 100:.2f}%')
print(f'🎯 Test Macro F1    : {test_macro_f1:.4f}')
print(f'🎯 Test Weighted F1 : {test_weighted_f1:.4f}')

print('\\n📊 BÁO CÁO PHÂN LOẠI CHI TIẾT (CLASSIFICATION REPORT):')
report = classification_report(y_test_true, y_test_pred, target_names=CLASSES, digits=4)
print(report)

# Confusion Matrix
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
                 ha='center', va='center',
                 color='white' if cm[i, j] > thresh else 'black')
plt.xlabel('Dự đoán (Predicted)', fontsize=11)
plt.ylabel('Thực tế (True)', fontsize=11)
plt.tight_layout()
plt.savefig('confusion_matrix_sota.png', dpi=300)
plt.show()

# Training Curves
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
plt.savefig('training_curves_sota.png', dpi=300)
plt.show()
""")

# 8. Export Metadata
add_code("""metadata = {
    'project': 'Cry Baby Detection - SOTA Audio Transfer Learning',
    'architecture': 'ResNet34-3Ch-LogMel-FocalLoss',
    'sample_rate': TARGET_SR,
    'duration_sec': DURATION_SEC,
    'n_mels': N_MELS,
    'classes': CLASSES,
    'class_to_idx': CLASS_TO_IDX,
    'test_accuracy': float(test_acc),
    'test_macro_f1': float(test_macro_f1),
    'test_weighted_f1': float(test_weighted_f1)
}

with open('sota_audio_metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=4, ensure_ascii=False)

print('📁 Đã xuất metadata mô hình tại: sota_audio_metadata.json')
print('🎉 HOÀN TẤT THÍ NGHIỆM HUẤN LUYỆN SOTA AUDIO THÀNH CÔNG!')
""")

out_path = 'kaggle_audio_sota_kernel/train_sota_audio_cry_classifier.ipynb'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print(f'✅ Đã tạo thành công notebook tại: {out_path}')
