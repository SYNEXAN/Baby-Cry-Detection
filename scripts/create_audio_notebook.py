import os
import nbformat as nbf

def build_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title Markdown
    cells.append(nbf.v4.new_markdown_cell("""# 👶 Phân Loại 5 Nguyên Nhân Tiếng Khóc Trẻ Em (Donate-a-Cry Dataset)
**Mô hình:** Transfer Learning ResNet-18 trên Log-Mel Spectrogram  
**Tập dữ liệu:** Donate-a-Cry Corpus (5 nhóm: `belly_pain`, `burping`, `discomfort`, `hungry`, `tired`)  
**Mục tiêu:** 
1. Giải quyết triệt để mất cân bằng lớp (Hungry chiếm ~84%, Burping chỉ 8 mẫu) bằng **Class-Weighted CrossEntropy** và **SpecAugment**.
2. Huấn luyện siêu tốc trên GPU T4 / P100 (khoảng 10-15 phút, cách rất xa giới hạn 12h của Kaggle).
3. Đầy đủ biểu đồ Loss/F1, ma trận nhầm lẫn (Confusion Matrix) và suy luận dự đoán mẫu âm thanh thực tế.
"""))

    # Cell 1: Environment Setup
    cells.append(nbf.v4.new_markdown_cell("""### 🛠️ Cell 1: Cài đặt thư viện & Kiểm tra GPU"""))
    cells.append(nbf.v4.new_code_cell("""# Cell 1: Cài đặt thư viện & Kiểm tra cấu hình GPU
import os
import sys
import time
import random
import json
import math
import glob
import shutil
import urllib.request
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import torchvision.models as models

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

# Cố định random seed để kết quả tái lập được
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 PyTorch Version: {torch.__version__}")
print(f"🔥 TorchAudio Version: {torchaudio.__version__}")
print(f"🔥 Thiết bị huấn luyện (Device): {device}")
if torch.cuda.is_available():
    print(f"   GPU Model: {torch.cuda.get_device_name(0)}")
    print(f"   GPU VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
"""))

    # Cell 2: Hyperparameters
    cells.append(nbf.v4.new_markdown_cell("""### ⚙️ Cell 2: Cấu hình Hyperparameters tập trung (`AudioConfig`)"""))
    cells.append(nbf.v4.new_code_cell("""# Cell 2: Toàn bộ cấu hình hệ thống
class AudioConfig:
    # 1. Âm thanh gốc
    SAMPLE_RATE = 16000          # Chuẩn 16 kHz tối ưu cho nhận dạng giọng nói & tiếng khóc
    DURATION = 5.0               # Độ dài mỗi đoạn clip (giây)
    TARGET_SAMPLES = int(SAMPLE_RATE * DURATION) # 80,000 samples
    
    # 2. Đặc trưng Log-Mel Spectrogram
    N_MELS = 128                 # Số dải tần số Mel
    N_FFT = 1024                 # Kích thước cửa sổ FFT (~64ms)
    HOP_LENGTH = 512             # Bước nhảy giữa các khung (~32ms)
    F_MIN = 50.0                 # Tần số thấp nhất
    F_MAX = 8000.0                # Tần số cao nhất (Nyquist tại 16kHz)
    
    # 3. Quá trình huấn luyện
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    PATIENCE = 15                # Dừng sớm nếu Validation Macro-F1 không tăng sau 15 epochs
    
    # 4. Các lớp nhãn
    CLASSES = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
    CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASSES)}
    IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(CLASSES)}
    NUM_CLASSES = len(CLASSES)
    
    # 5. Đường dẫn lưu trữ
    OUTPUT_DIR = "/kaggle/working" if os.path.exists("/kaggle") else "./kaggle_results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    MODEL_PATH = os.path.join(OUTPUT_DIR, "best_audio_classifier.pt")
    REPORT_PATH = os.path.join(OUTPUT_DIR, "classification_report.json")

print("✅ Đã thiết lập xong cấu hình AudioConfig.")
for k, v in AudioConfig.__dict__.items():
    if not k.startswith("__") and not callable(v):
        print(f"   - {k}: {v}")
"""))

    # Cell 3: Dataset Locator & Download Fallback
    cells.append(nbf.v4.new_markdown_cell("""### 📂 Cell 3: Tự động phát hiện hoặc tải Dataset Donate-a-Cry"""))
    cells.append(nbf.v4.new_code_cell("""# Cell 3: Định vị đường dẫn dữ liệu (Tương thích cả Kaggle lẫn Local)
import requests

def locate_or_download_dataset():
    # 1. Tự động tìm kiếm trong /kaggle/input
    if os.path.exists("/kaggle/input"):
        # Ưu tiên tìm dataset có cấu trúc train/val/test đã cân bằng
        for root, dirs, files in os.walk("/kaggle/input"):
            if all(d in dirs for d in ['train', 'val', 'test']):
                print(f"✅ Đã tìm thấy Balanced Dataset (có sẵn train/val/test) tại: {root}")
                return root
        # Thứ hai: Tìm thư mục chứa 5 classes
        for root, dirs, files in os.walk("/kaggle/input"):
            if all(c in dirs for c in AudioConfig.CLASSES):
                print(f"✅ Đã tìm thấy dataset trong /kaggle/input tại: {root}")
                return root

    possible_roots = [
        "/kaggle/input/donate-a-cry-balanced-5classes",
        "/kaggle/input/donate-a-cry-balanced-5classes/audio_balanced_5classes",
        "data/processed/audio_balanced_5classes",
        "../data/processed/audio_balanced_5classes",
        "/kaggle/input/donate-a-cry-corpus",
        "data/raw/donate_a_cry",
        "/kaggle/working/donate_a_cry"
    ]
    
    for root in possible_roots:
        if os.path.exists(root):
            subdirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
            if all(d in subdirs for d in ['train', 'val', 'test']) or all(c in subdirs for c in AudioConfig.CLASSES):
                print(f"✅ Đã tìm thấy dataset tại: {root}")
                return root
                
    # Fallback: Tải từ Hugging Face
    print("🔄 Không tìm thấy dataset có sẵn. Đang tự động tải từ Hugging Face (Nooon/Donate_a_cry)...")
    dest_dir = "/kaggle/working/donate_a_cry"
    os.makedirs(dest_dir, exist_ok=True)
    
    api_url = "https://huggingface.co/api/datasets/Nooon/Donate_a_cry"
    base_url = "https://huggingface.co/datasets/Nooon/Donate_a_cry/resolve/main"
    
    resp = requests.get(api_url, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    
    wav_files = [s["rfilename"] for s in data.get("siblings", []) if s["rfilename"].endswith(".wav")]
    print(f"   Tìm thấy {len(wav_files)} file .wav. Đang tải về {dest_dir} ...")
    
    for rfile in wav_files:
        local_file = os.path.join(dest_dir, rfile)
        os.makedirs(os.path.dirname(local_file), exist_ok=True)
        if not os.path.exists(local_file) or os.path.getsize(local_file) == 0:
            urllib.request.urlretrieve(f"{base_url}/{rfile}", local_file)
            
    print(f"✅ Đã tải xong dataset tại: {dest_dir}")
    return dest_dir

DATASET_ROOT = locate_or_download_dataset()

# Quét danh sách file và tạo DataFrame
has_splits = (
    os.path.exists(os.path.join(DATASET_ROOT, "train")) and
    os.path.exists(os.path.join(DATASET_ROOT, "val")) and
    os.path.exists(os.path.join(DATASET_ROOT, "test"))
)

records = []
if has_splits:
    print("✨ Dataset đã có sẵn phân chia Train / Val / Test (Zero Data Leakage).")
    for split in ['train', 'val', 'test']:
        for cat in AudioConfig.CLASSES:
            cat_dir = os.path.join(DATASET_ROOT, split, cat)
            if os.path.exists(cat_dir):
                files = [os.path.join(cat_dir, f) for f in os.listdir(cat_dir) if f.lower().endswith(".wav")]
                for f in files:
                    records.append({
                        "filepath": f,
                        "filename": os.path.basename(f),
                        "category": cat,
                        "label": AudioConfig.CLASS_TO_IDX[cat],
                        "split": split
                    })
else:
    for cat in AudioConfig.CLASSES:
        cat_dir = os.path.join(DATASET_ROOT, cat)
        files = [os.path.join(cat_dir, f) for f in os.listdir(cat_dir) if f.lower().endswith(".wav")]
        for f in files:
            records.append({
                "filepath": f,
                "filename": os.path.basename(f),
                "category": cat,
                "label": AudioConfig.CLASS_TO_IDX[cat],
                "split": "unknown"
            })

df_samples = pd.DataFrame(records)
print(f"\\n📊 Tổng số mẫu âm thanh: {len(df_samples):,}")
class_counts = df_samples['category'].value_counts()
if has_splits:
    print(pd.crosstab(df_samples['category'], df_samples['split']))
else:
    print(class_counts.to_string())

# Vẽ biểu đồ phân bố lớp ban đầu
plt.figure(figsize=(8, 4))
sns.barplot(x=class_counts.index, y=class_counts.values, hue=class_counts.index, palette="mako", legend=False)
plt.title("Phân bố số lượng mẫu 5 lớp nguyên nhân (Donate-a-Cry)")
plt.xlabel("Nguyên nhân tiếng khóc")
plt.ylabel("Số lượng file .wav")
for i, v in enumerate(class_counts.values):
    plt.text(i, v + 5, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.show()
"""))

    # Cell 4: Audio Augmentation
    cells.append(nbf.v4.new_markdown_cell("""### 🎵 Cell 4: Định nghĩa Pipeline Tăng Cường Dữ Liệu Âm Thanh (SpecAugment)"""))
    cells.append(nbf.v4.new_code_cell("""# Cell 4: Các hàm tăng cường âm thanh và phổ âm (Data Augmentation)
class AudioAugmentation:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        # SpecAugment: Che dải tần số (Frequency Masking) và che dải thời gian (Time Masking)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=16)
        self.time_mask = T.TimeMasking(time_mask_param=35)

    def time_shift(self, waveform, max_ratio=0.15):
        # Dịch chuyển ngẫu nhiên sóng âm sang trái hoặc phải
        shift_samples = int(random.uniform(-max_ratio, max_ratio) * waveform.shape[-1])
        return torch.roll(waveform, shifts=shift_samples, dims=-1)

    def add_noise(self, waveform, noise_level=0.005):
        # Thêm nhiễu trắng nhỏ mô phỏng môi trường phòng ở
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise

    def apply_spectrogram_augment(self, spec):
        # Áp dụng SpecAugment trên ảnh phổ
        spec = self.freq_mask(spec)
        spec = self.time_mask(spec)
        return spec

augmenter = AudioAugmentation(sample_rate=AudioConfig.SAMPLE_RATE)
print("✅ Bộ tăng cường âm thanh AudioAugmentation (SpecAugment + TimeShift + Noise) đã sẵn sàng.")
"""))

    # Cell 5: PyTorch Dataset
    cells.append(nbf.v4.new_markdown_cell("""### 📦 Cell 5: Xây dựng PyTorch Dataset (`DonateACryDataset`)"""))
    cells.append(nbf.v4.new_code_cell("""# Cell 5: Lớp Dataset xử lý âm thanh sang Log-Mel Spectrogram
class DonateACryDataset(Dataset):
    def __init__(self, df, config, is_train=False, minority_oversample=True):
        self.df = df.reset_index(drop=True)
        self.config = config
        self.is_train = is_train
        self.minority_oversample = minority_oversample
        
        # Bộ biến đổi Mel-Spectrogram
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
            f_min=config.F_MIN,
            f_max=config.F_MAX
        )
        self.amp_to_db = T.AmplitudeToDB(top_db=80.0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row['filepath']
        label = row['label']
        
        # 1. Đọc file âm thanh
        waveform, sr = torchaudio.load(filepath)
        
        # 2. Chuyển thành mono nếu là stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # 3. Resample nếu tần số khác 16kHz
        if sr != self.config.SAMPLE_RATE:
            resampler = T.Resample(orig_freq=sr, new_freq=self.config.SAMPLE_RATE)
            waveform = resampler(waveform)
            
        # 4. Cắt hoặc đệm (Zero-padding) cho đủ 5.0 giây (80,000 samples)
        num_samples = waveform.shape[-1]
        target_len = self.config.TARGET_SAMPLES
        if num_samples < target_len:
            pad_amount = target_len - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        elif num_samples > target_len:
            # Nếu dài hơn thì cắt ngẫu nhiên khi train, cắt giữa khi val/test
            if self.is_train:
                max_start = num_samples - target_len
                start_idx = random.randint(0, max_start)
            else:
                start_idx = (num_samples - target_len) // 2
            waveform = waveform[:, start_idx : start_idx + target_len]
            
        # 5. Tăng cường trên sóng âm (Waveform Augmentation khi Train)
        if self.is_train:
            waveform = augmenter.time_shift(waveform)
            if random.random() < 0.5:
                waveform = augmenter.add_noise(waveform)
                
        # 6. Chuẩn hóa biên độ đỉnh (Peak Normalization)
        peak = torch.max(torch.abs(waveform))
        if peak > 0:
            waveform = waveform / peak
            
        # 7. Trích xuất Log-Mel Spectrogram
        mel_spec = self.mel_transform(waveform)     # [1, n_mels, time_steps]
        log_mel_spec = self.amp_to_db(mel_spec)     # Chuyển sang thang dB
        
        # 8. Áp dụng SpecAugment khi Train (đặc biệt ưu tiên cho các lớp thiểu số)
        if self.is_train:
            # Lớp thiểu số (không phải hungry) được áp dụng SpecAugment 100%
            if label != AudioConfig.CLASS_TO_IDX['hungry'] or random.random() < 0.7:
                log_mel_spec = augmenter.apply_spectrogram_augment(log_mel_spec)
                
        # 9. Chuẩn hóa phân phối chuẩn (Zero-mean, Unit-variance)
        spec_mean = log_mel_spec.mean()
        spec_std = log_mel_spec.std() + 1e-6
        normalized_spec = (log_mel_spec - spec_mean) / spec_std
        
        return normalized_spec, torch.tensor(label, dtype=torch.long), filepath

print("✅ Đã định nghĩa lớp DonateACryDataset.")
"""))

    # Cell 6: Stratified Splitting & Class Weights
    cells.append(nbf.v4.new_markdown_cell("""### 🔀 Cell 6: Phân chia tập dữ liệu & Tính toán Class Weights"""))
    cells.append(nbf.v4.new_code_cell("""# Cell 6: Nạp tập phân chia Train/Val/Test và tính trọng số hàm mất mát
from sklearn.model_selection import train_test_split

if has_splits:
    print("✅ Sử dụng tập phân chia Train / Val / Test cân bằng đã định sẵn (Zero Data Leakage):")
    df_train = df_samples[df_samples['split'] == 'train'].reset_index(drop=True)
    df_val = df_samples[df_samples['split'] == 'val'].reset_index(drop=True)
    df_test = df_samples[df_samples['split'] == 'test'].reset_index(drop=True)
else:
    # 1. Tách Test độc lập (15%)
    X = df_samples
    y = df_samples['label'].values

    df_train_val, df_test = train_test_split(
        X, test_size=0.15, stratify=y, random_state=42
    )

    # 2. Tách Train (70%) và Validation (15% trên tổng thể -> tương đương ~17.6% của train_val)
    val_relative_size = 0.15 / (0.70 + 0.15)
    df_train, df_val = train_test_split(
        df_train_val, test_size=val_relative_size, stratify=df_train_val['label'].values, random_state=42
    )

print(f"📊 Kích thước các tập dữ liệu:")
print(f"   - Train set:      {len(df_train)} mẫu ({len(df_train)/len(df_samples)*100:.1f}%)")
print(f"   - Validation set: {len(df_val)} mẫu ({len(df_val)/len(df_samples)*100:.1f}%)")
print(f"   - Test set:       {len(df_test)} mẫu ({len(df_test)/len(df_samples)*100:.1f}%)")

# 3. Tính toán Class Weights
train_counts = Counter(df_train['label'].values)
total_train_samples = len(df_train)
num_classes = AudioConfig.NUM_CLASSES

# w_c = N_total / (N_classes * N_c)
weights_list = []
for c_idx in range(num_classes):
    count_c = train_counts.get(c_idx, 1)
    w = total_train_samples / (num_classes * count_c)
    weights_list.append(w)

# Chuẩn hóa trọng số
weights_tensor = torch.tensor(weights_list, dtype=torch.float).to(device)
print(f"\\n⚖️ Trọng số Loss phạt từng lớp (Class Weights):")
for c_idx, cls_name in AudioConfig.IDX_TO_CLASS.items():
    print(f"   - {cls_name:12s} ({train_counts.get(c_idx, 0):3d} mẫu train) -> Trọng số Loss: {weights_list[c_idx]:.3f}")

# 4. Khởi tạo DataLoaders
train_dataset = DonateACryDataset(df_train, AudioConfig, is_train=True)
val_dataset = DonateACryDataset(df_val, AudioConfig, is_train=False)
test_dataset = DonateACryDataset(df_test, AudioConfig, is_train=False)

train_loader = DataLoader(train_dataset, batch_size=AudioConfig.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=AudioConfig.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=AudioConfig.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# Kiểm tra kích thước một batch mẫu
sample_spec, sample_label, _ = next(iter(train_loader))
print(f"\\n✅ Batch Tensor Shape: {sample_spec.shape} (Batch, Channel, Mels, Time_Steps)")
print(f"✅ Batch Labels Shape: {sample_label.shape}")
"""))

    # Cell 7: Model Architecture
    cells.append(nbf.v4.new_markdown_cell("""### 🧠 Cell 7: Xây dựng Kiến trúc Mô hình (Custom ResNet-18 Audio)"""))
    cells.append(nbf.v4.new_code_cell("""# Cell 7: Kiến trúc Transfer Learning ResNet-18 thích ứng cho 1-channel Log-Mel Spectrogram
class ResNetAudioClassifier(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(ResNetAudioClassifier, self).__init__()
        # Nạp ResNet-18 pretrained từ ImageNet
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        
        # 1. Điều chỉnh lớp Conv đầu tiên từ 3 channels sang 1 channel
        # Lấy trung bình trọng số của 3 kênh RGB cũ nạp vào kênh 1 mới để giữ đặc trưng cạnh phổ
        orig_conv1 = backbone.conv1
        new_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                new_conv1.weight.copy_(orig_conv1.weight.mean(dim=1, keepdim=True))
        self.conv1 = new_conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        
        # Các khối Residual Blocks
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        # Global Average Pooling & Adaptive Max Pooling kết hợp
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 2. Fully Connected Classifier Head với Dropout chống Overfitting
        in_features = backbone.fc.in_features # 512
        self.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)
        return out

model = ResNetAudioClassifier(num_classes=AudioConfig.NUM_CLASSES, pretrained=True).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ Mô hình ResNet-18 Audio đã khởi tạo thành công!")
print(f"   - Tổng số tham số:      {total_params:,}")
print(f"   - Tham số cần cập nhật: {trainable_params:,}")
"""))

    # Cell 8: Training Loop
    cells.append(nbf.v4.new_markdown_cell("""### 🚀 Cell 8: Vòng Lặp Huấn Luyện (Training Loop) với Class-Weighted Loss & Early Stopping"""))
    cells.append(nbf.v4.new_code_cell("""# Cell 8: Huấn luyện với Checkpoint theo dõi Validation Macro-F1
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = optim.AdamW(model.parameters(), lr=AudioConfig.LEARNING_RATE, weight_decay=AudioConfig.WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=AudioConfig.EPOCHS, eta_min=1e-6)

history = {
    'train_loss': [], 'val_loss': [],
    'train_acc': [], 'val_acc': [],
    'train_f1': [], 'val_f1': []
}

best_val_f1 = 0.0
best_epoch = 0
no_improve_epochs = 0

print(f"🚀 Bắt đầu huấn luyện {AudioConfig.EPOCHS} epochs trên {device}...")
start_time = time.time()

for epoch in range(1, AudioConfig.EPOCHS + 1):
    # --- A. GIAI ĐOẠN TRAIN ---
    model.train()
    running_loss = 0.0
    all_preds_train = []
    all_targets_train = []
    
    for specs, targets, _ in train_loader:
        specs, targets = specs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(specs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Gradient Clipping để ổn định huấn luyện
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        running_loss += loss.item() * specs.size(0)
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        all_preds_train.extend(preds)
        all_targets_train.extend(targets.cpu().numpy())
        
    scheduler.step()
    
    train_loss = running_loss / len(df_train)
    train_acc = accuracy_score(all_targets_train, all_preds_train)
    train_f1 = f1_score(all_targets_train, all_preds_train, average='macro', zero_division=0)
    
    # --- B. GIAI ĐOẠN VALIDATION ---
    model.eval()
    val_running_loss = 0.0
    all_preds_val = []
    all_targets_val = []
    
    with torch.no_grad():
        for specs, targets, _ in val_loader:
            specs, targets = specs.to(device), targets.to(device)
            outputs = model(specs)
            loss = criterion(outputs, targets)
            
            val_running_loss += loss.item() * specs.size(0)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds_val.extend(preds)
            all_targets_val.extend(targets.cpu().numpy())
            
    val_loss = val_running_loss / len(df_val)
    val_acc = accuracy_score(all_targets_val, all_preds_val)
    val_f1 = f1_score(all_targets_val, all_preds_val, average='macro', zero_division=0)
    
    # Ghi nhận lịch sử
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    history['train_f1'].append(train_f1)
    history['val_f1'].append(val_f1)
    
    # Checkpoint: Lưu model khi Val Macro-F1 cao nhất
    improved_str = ""
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch = epoch
        no_improve_epochs = 0
        config_dict = {k: v for k, v in AudioConfig.__dict__.items() if not k.startswith("__") and not callable(v)}
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_val_f1': best_val_f1,
            'val_loss': val_loss,
            'config': config_dict
        }, AudioConfig.MODEL_PATH)
        improved_str = " ⭐ [ĐÃ LƯU BEST MODEL]"
    else:
        no_improve_epochs += 1
        
    if epoch % 5 == 0 or epoch == 1 or improved_str:
        print(f"Epoch [{epoch:2d}/{AudioConfig.EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc*100:.1f}% | Val Macro-F1: {val_f1:.4f}{improved_str}")
              
    # Early Stopping
    if no_improve_epochs >= AudioConfig.PATIENCE:
        print(f"\\n⏹️ Dừng sớm tại Epoch {epoch} vì Val Macro-F1 không cải thiện trong {AudioConfig.PATIENCE} epochs.")
        break

total_duration = time.time() - start_time
print(f"\\n🏁 Hoàn tất huấn luyện sau {total_duration/60:.2f} phút!")
print(f"🏆 Best Validation Macro-F1: {best_val_f1:.4f} tại Epoch {best_epoch}")
"""))

    # Cell 9: Plot Training Curves
    cells.append(nbf.v4.new_markdown_cell("""### 📈 Cell 9: Trực Quan Hóa Đồ Thị Huấn Luyện (Loss & Macro-F1 Curves)"""))
    cells.append(nbf.v4.new_code_cell("""# Cell 9: Vẽ đồ thị Loss và Macro-F1
epochs_ran = range(1, len(history['train_loss']) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

# 1. Đồ thị Loss
ax1.plot(epochs_ran, history['train_loss'], label='Train Loss', color='#2563eb', linewidth=2)
ax1.plot(epochs_ran, history['val_loss'], label='Validation Loss', color='#dc2626', linewidth=2, linestyle='--')
ax1.axvline(x=best_epoch, color='green', linestyle=':', label=f'Best Model (Epoch {best_epoch})')
ax1.set_title("Quá trình hội tụ hàm mất mát (Loss)", fontsize=13, fontweight='bold')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# 2. Đồ thị Macro-F1 & Accuracy
ax2.plot(epochs_ran, history['val_f1'], label='Val Macro-F1', color='#16a34a', linewidth=2)
ax2.plot(epochs_ran, history['train_f1'], label='Train Macro-F1', color='#9333ea', linewidth=1.5, linestyle=':')
ax2.plot(epochs_ran, history['val_acc'], label='Val Accuracy', color='#ea580c', linewidth=1.5, linestyle='--')
ax2.axvline(x=best_epoch, color='green', linestyle=':', label=f'Best F1: {best_val_f1:.3f}')
ax2.set_title("Chỉ số Macro-F1 & Accuracy qua các Epochs", fontsize=13, fontweight='bold')
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Score")
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

plt.tight_layout()
chart_path = os.path.join(AudioConfig.OUTPUT_DIR, "training_curves.png")
plt.savefig(chart_path, dpi=300)
plt.show()
print(f"✅ Đã lưu biểu đồ huấn luyện tại: {chart_path}")
"""))

    # Cell 10: Evaluation on Test Set
    cells.append(nbf.v4.new_markdown_cell("""### 🎯 Cell 10: Đánh Giá Độc Lập Trên Tập Test (Confusion Matrix & Classification Report)"""))
    cells.append(nbf.v4.new_code_cell("""# Cell 10: Đánh giá mô hình tốt nhất trên tập Test (15% mẫu độc lập)
# 1. Nạp lại checkpoint tốt nhất
checkpoint = torch.load(AudioConfig.MODEL_PATH)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"✅ Đã nạp lại checkpoint từ Epoch {checkpoint['epoch']} với Val F1 = {checkpoint['best_val_f1']:.4f}")

# 2. Suy luận trên Test Set
y_true = []
y_pred = []
y_probs = []

with torch.no_grad():
    for specs, targets, _ in test_loader:
        specs = specs.to(device)
        outputs = model(specs)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        
        y_true.extend(targets.numpy())
        y_pred.extend(preds)
        y_probs.extend(probs)

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_probs = np.array(y_probs)

# 3. In Classification Report chi tiết
print("\\n" + "="*60)
print("              BÁO CÁO PHÂN LOẠI TRÊN TẬP TEST")
print("="*60)
report = classification_report(
    y_true, y_pred,
    target_names=AudioConfig.CLASSES,
    digits=4,
    zero_division=0
)
print(report)

test_macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
test_accuracy = accuracy_score(y_true, y_pred)
print(f"🎯 Test Accuracy: {test_accuracy*100:.2f}%")
print(f"🎯 Test Macro-F1: {test_macro_f1:.4f}")

# 4. Vẽ Heatmap Ma trận nhầm lẫn (Confusion Matrix)
cm = confusion_matrix(y_true, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Ma trận số lượng tuyệt đối
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=AudioConfig.CLASSES, yticklabels=AudioConfig.CLASSES, ax=ax1)
ax1.set_title("Confusion Matrix (Số lượng mẫu thực tế)", fontsize=12, fontweight='bold')
ax1.set_xlabel("Dự đoán của AI")
ax1.set_ylabel("Nhãn thực tế")

# Ma trận chuẩn hóa (%)
sns.heatmap(cm_normalized * 100, annot=True, fmt='.1f', cmap='Greens', xticklabels=AudioConfig.CLASSES, yticklabels=AudioConfig.CLASSES, ax=ax2)
ax2.set_title("Confusion Matrix Chuẩn Hóa (%)", fontsize=12, fontweight='bold')
ax2.set_xlabel("Dự đoán của AI")
ax2.set_ylabel("Nhãn thực tế")

plt.tight_layout()
cm_path = os.path.join(AudioConfig.OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=300)
plt.show()
print(f"✅ Đã lưu ma trận nhầm lẫn tại: {cm_path}")
"""))

    # Cell 11: Real-World Inference Demo
    cells.append(nbf.v4.new_markdown_cell("""### 🔍 Cell 11: Mô Phỏng Dự Đoán Mẫu Âm Thanh Thực Tế (Inference Demo)"""))
    cells.append(nbf.v4.new_code_cell("""# Cell 11: Thử nghiệm trực quan dự đoán từng mẫu âm thanh
import random

model.eval()
fig, axes = plt.subplots(3, 2, figsize=(16, 10))
axes = axes.flatten()

# Chọn ngẫu nhiên 6 mẫu từ tập Test
sample_indices = random.sample(range(len(test_dataset)), 6)

for plot_idx, idx in enumerate(sample_indices):
    spec, true_label_idx, filepath = test_dataset[idx]
    
    # Dự đoán
    with torch.no_grad():
        input_tensor = spec.unsqueeze(0).to(device) # [1, 1, n_mels, time_steps]
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_label_idx = np.argmax(probs)
        confidence = probs[pred_label_idx] * 100
        
    true_label_name = AudioConfig.IDX_TO_CLASS[true_label_idx.item()]
    pred_label_name = AudioConfig.IDX_TO_CLASS[pred_label_idx]
    
    # Hiển thị ảnh Log-Mel Spectrogram
    ax = axes[plot_idx]
    im = ax.imshow(spec.squeeze(0).numpy(), aspect='auto', origin='lower', cmap='inferno')
    
    is_correct = (true_label_idx.item() == pred_label_idx)
    status_icon = "✅ ĐÚNG" if is_correct else "❌ SAI"
    color = "green" if is_correct else "red"
    
    ax.set_title(
        f"{status_icon} | Nhãn thực: {true_label_name} ➔ AI đoán: {pred_label_name} ({confidence:.1f}%)\\nFile: {os.path.basename(filepath)[:35]}...",
        color=color, fontsize=10, fontweight='bold'
    )
    ax.set_ylabel("Mel Freq Bin")
    ax.set_xlabel("Time Frames")

plt.tight_layout()
demo_path = os.path.join(AudioConfig.OUTPUT_DIR, "inference_samples_demo.png")
plt.savefig(demo_path, dpi=300)
plt.show()
print(f"✅ Đã lưu kết quả suy luận demo tại: {demo_path}")
"""))

    # Cell 12: Export Checkpoint & Summary
    cells.append(nbf.v4.new_markdown_cell("""### 💾 Cell 12: Đóng Gói Trọng Số & Bảng Tóm Tắt Triển Khai"""))
    cells.append(nbf.v4.new_code_cell("""# Cell 12: Xuất metadata và hướng dẫn tích hợp vào hệ thống
summary_data = {
    "project": "Cry Baby Detection - 5 Classes Cause Classifier",
    "model_architecture": "ResNet18-Audio-TransferLearning",
    "best_epoch": best_epoch,
    "best_val_macro_f1": float(best_val_f1),
    "test_macro_f1": float(test_macro_f1),
    "test_accuracy": float(test_accuracy),
    "classes": AudioConfig.CLASSES,
    "class_to_idx": AudioConfig.CLASS_TO_IDX,
    "sample_rate": AudioConfig.SAMPLE_RATE,
    "duration_sec": AudioConfig.DURATION
}

summary_path = os.path.join(AudioConfig.OUTPUT_DIR, "audio_classifier_metadata.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary_data, f, indent=4, ensure_ascii=False)

print("="*60)
print("🎉 ĐÃ HOÀN TẤT HUẤN LUYỆN VÀ ĐÓNG GÓI THÀNH CÔNG!")
print("="*60)
print(f"📁 Trọng số mô hình: {AudioConfig.MODEL_PATH}")
print(f"📁 Metadata triển khai: {summary_path}")
print(f"📁 Đồ thị huấn luyện: {chart_path}")
print(f"📁 Ma trận nhầm lẫn: {cm_path}")
"""))

    nb['cells'] = cells
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    }
    output_dir = "notebooks"
    os.makedirs(output_dir, exist_ok=True)
    nb_path = os.path.join(output_dir, "train_audio_cry_classifier.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"✅ Notebook đã được tạo thành công tại: {nb_path}")
    
    # Also save a copy for kaggle_audio_kernel
    kaggle_audio_dir = "kaggle_audio_kernel"
    os.makedirs(kaggle_audio_dir, exist_ok=True)
    kaggle_audio_nb_path = os.path.join(kaggle_audio_dir, "train_audio_cry_classifier.ipynb")
    with open(kaggle_audio_nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"✅ Notebook đã được sao chép vào: {kaggle_audio_nb_path}")

if __name__ == "__main__":
    build_notebook()
