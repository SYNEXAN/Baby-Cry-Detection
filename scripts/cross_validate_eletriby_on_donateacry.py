import os
import glob
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn as nn
import torchvision.models as models
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

TARGET_SR = 16000
DURATION_SEC = 5.0
TARGET_SAMPLES = int(TARGET_SR * DURATION_SEC)
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512

CLASSES_4 = ['burping', 'needs', 'physical_pain', 'scared']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES_4)}

# ResNet18 Architecture matching checkpoint
class AudioResNet18(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Loading checkpoint on: {device}")

model_path = "models/best_eletriby_4class_audio_classifier.pt"
model = AudioResNet18(num_classes=4).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"Loaded {model_path} successfully.")

mel_transform = T.MelSpectrogram(
    sample_rate=TARGET_SR,
    n_fft=N_FFT,
    win_length=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    power=2.0
)
amp_to_db = T.AmplitudeToDB()

import soundfile as sf

def preprocess_audio(filepath):
    data, sr = sf.read(filepath, dtype='float32')
    waveform = torch.from_numpy(data)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.t() # (channels, samples)
    if sr != TARGET_SR:
        waveform = T.Resample(sr, TARGET_SR)(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    if waveform.shape[1] < TARGET_SAMPLES:
        repeats = int(np.ceil(TARGET_SAMPLES / waveform.shape[1]))
        waveform = waveform.repeat(1, repeats)[:, :TARGET_SAMPLES]
    elif waveform.shape[1] > TARGET_SAMPLES:
        mid = waveform.shape[1] // 2
        half = TARGET_SAMPLES // 2
        start = max(0, mid - half)
        waveform = waveform[:, start : start + TARGET_SAMPLES]

    mel = mel_transform(waveform)
    log_mel = amp_to_db(mel)
    norm_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    delta = torchaudio.functional.compute_deltas(norm_mel)
    delta2 = torchaudio.functional.compute_deltas(delta)
    tensor_3ch = torch.cat([norm_mel, delta, delta2], dim=0)
    return tensor_3ch.unsqueeze(0).to(device)

# Donate-a-Cry Mapping to 4 classes
# burping -> burping
# belly_pain -> physical_pain
# hungry -> needs
# tired -> needs
# discomfort -> needs

raw_dir = "data/raw/donate_a_cry"
records = []
categories = ['burping', 'belly_pain', 'discomfort', 'tired', 'hungry']

for cat in categories:
    cat_path = os.path.join(raw_dir, cat)
    if not os.path.exists(cat_path):
        continue
    files = [os.path.join(cat_path, f) for f in os.listdir(cat_path) if f.endswith('.wav')]
    # For hungry, sample 50 to avoid drowning out the other classes
    if cat == 'hungry' and len(files) > 50:
        import random
        random.seed(42)
        files = random.sample(files, 50)
        
    for f in files:
        # Expected target class in 4-class taxonomy
        if cat == 'burping':
            exp = 'burping'
        elif cat == 'belly_pain':
            exp = 'physical_pain'
        else:
            exp = 'needs'
        records.append({'filepath': f, 'raw_cat': cat, 'expected_4class': exp})

df_eval = pd.DataFrame(records)
print(f"\nTotal test files selected from raw Donate-a-Cry: {len(df_eval)}")
print(df_eval['raw_cat'].value_counts())

y_true = []
y_pred = []
pred_probs = []

with torch.no_grad():
    for _, row in df_eval.iterrows():
        tensor = preprocess_audio(row['filepath'])
        out = model(tensor)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        
        y_true.append(row['expected_4class'])
        y_pred.append(CLASSES_4[pred_idx])
        pred_probs.append(probs)

print("\n" + "="*65)
print("CROSS-DATASET VALIDATION RESULTS (Donate-a-Cry -> Karim Eletriby Model)")
print("="*65)

eval_classes = ['burping', 'needs', 'physical_pain']
report = classification_report(y_true, y_pred, labels=eval_classes, digits=4, zero_division=0)
print(report)

print("\nConfusion Matrix (Rows: True Expected, Cols: Model Prediction):")
cm = confusion_matrix(y_true, y_pred, labels=eval_classes)
df_cm = pd.DataFrame(cm, index=[f"True: {c}" for c in eval_classes], columns=[f"Pred: {c}" for c in eval_classes])
print(df_cm)

print("\nDetailed Breakdown by Raw Donate-a-Cry Sub-Categories:")
df_eval['predicted'] = y_pred
for raw_c in categories:
    sub = df_eval[df_eval['raw_cat'] == raw_c]
    if len(sub) == 0:
        continue
    print(f"\n[Raw: {raw_c.upper()}] (Total: {len(sub)} files)")
    print(sub['predicted'].value_counts())
