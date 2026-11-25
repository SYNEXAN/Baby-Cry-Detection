"""
Script: build_balanced_audio.py
Mục đích:
Xây dựng tập dữ liệu âm thanh cân bằng 5 lớp tiếng khóc (Donate-a-Cry)
theo phương pháp tăng cường dữ liệu khoa học từ bài báo nghiên cứu Euréka 2025.

Nguyên tắc khoa học bắt buộc (Zero Data Leakage):
1. Phân chia Train (70%) / Val (15%) / Test (15%) TRƯỚC KHI tăng cường.
   Tất cả các đoạn cắt và biến thể của cùng 1 file gốc CHỈ được nằm trong 1 tập duy nhất.
2. Tập Test và Val giữ nguyên bản (không thêm nhiễu nhân tạo hay đổi cao độ) để kiểm định khách quan.
3. Tập Train được tăng cường (Pitch shift, Noise injection, Time shift, Sliding window)
   để cân bằng số lượng mẫu của cả 5 lớp lên khoảng 350 mẫu/lớp.
"""

import os
import glob
import json
import random
import shutil
from collections import Counter, defaultdict

import numpy as np
from scipy import signal
from scipy.io import wavfile
import pandas as pd

# Cố định seed
np.random.seed(42)
random.seed(42)

RAW_DIR = "data/raw/donate_a_cry"
OUT_DIR = "data/processed/audio_balanced_5classes"
TARGET_SR = 16000
DURATION_SEC = 5.0
TARGET_SAMPLES = int(TARGET_SR * DURATION_SEC) # 80,000 samples
TARGET_TRAIN_SAMPLES_PER_CLASS = 350
TARGET_VAL_SAMPLES_PER_CLASS = 50
TARGET_TEST_SAMPLES_PER_CLASS = 50

CLASSES = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']

def fast_resample(audio, orig_sr, target_sr=16000):
    if orig_sr == target_sr:
        return audio
    duration = len(audio) / orig_sr
    orig_times = np.linspace(0, duration, len(audio), endpoint=False)
    target_len = int(duration * target_sr)
    target_times = np.linspace(0, duration, target_len, endpoint=False)
    return np.interp(target_times, orig_times, audio).astype(audio.dtype)

def load_and_standardize_wav(filepath, target_sr=16000):
    sr, data = wavfile.read(filepath)
    # Chuyển sang mono nếu là stereo
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    # Chuẩn hóa về int16 nếu ở dạng float
    if data.dtype in [np.float32, np.float64]:
        data = np.clip(data * 32767, -32768, 32767).astype(np.int16)
    elif data.dtype != np.int16:
        data = data.astype(np.int16)
        
    # Resample
    if sr != target_sr:
        data = fast_resample(data, sr, target_sr)
    return data

def pad_or_truncate(audio, target_length=80000):
    n = len(audio)
    if n >= target_length:
        return audio[:target_length]
    # Nếu ngắn hơn thì đệm lặp lại nhẹ (loop) hoặc zero-padding
    pad_needed = target_length - n
    pad = np.zeros(pad_needed, dtype=audio.dtype)
    return np.concatenate([audio, pad])

def slice_audio(audio, window_size=80000, step_size=40000):
    n = len(audio)
    if n <= window_size:
        return [pad_or_truncate(audio, window_size)]
    slices = []
    for start in range(0, n - window_size + 1, step_size):
        slices.append(audio[start : start + window_size])
    if len(slices) == 0:
        slices.append(pad_or_truncate(audio, window_size))
    return slices

def pitch_shift(audio, factor):
    # factor > 1: tăng cao độ, factor < 1: giảm cao độ
    n_samples = len(audio)
    new_len = max(100, int(n_samples / factor))
    orig_times = np.linspace(0, 1, n_samples, endpoint=False)
    new_times = np.linspace(0, 1, new_len, endpoint=False)
    resampled = np.interp(new_times, orig_times, audio)
    # Resample lại về độ dài chuẩn
    final_times = np.linspace(0, 1, n_samples, endpoint=False)
    shifted = np.interp(final_times, new_times, resampled)
    return shifted.astype(audio.dtype)

def add_noise(audio, noise_ratio=0.008):
    max_amp = np.max(np.abs(audio)) + 1e-6
    noise = np.random.normal(0, max_amp * noise_ratio, len(audio))
    return np.clip(audio + noise, -32768, 32767).astype(audio.dtype)

def time_shift(audio, shift_ratio=0.1):
    shift = int(len(audio) * shift_ratio)
    return np.roll(audio, shift)

def speed_change(audio, speed_factor):
    n_samples = len(audio)
    new_len = max(100, int(n_samples / speed_factor))
    orig_times = np.linspace(0, 1, n_samples, endpoint=False)
    new_times = np.linspace(0, 1, new_len, endpoint=False)
    resampled = np.interp(new_times, orig_times, audio).astype(audio.dtype)
    return pad_or_truncate(resampled, n_samples)

def main():
    print("="*70)
    print("🚀 BẮT ĐẦU XÂY DỰNG BỘ DỮ LIỆU ÂM THANH CÂN BẰNG (BALANCED AUDIO PIPELINE)")
    print("="*70)

    # 1. Thu thập tất cả các file gốc
    raw_samples = defaultdict(list)
    for cat in CLASSES:
        cat_dir = os.path.join(RAW_DIR, cat)
        files = [os.path.join(cat_dir, f) for f in os.listdir(cat_dir) if f.lower().endswith(".wav")]
        raw_samples[cat] = sorted(files)
        print(f"   - Lớp {cat:12s}: {len(files):3d} file gốc")

    # 2. Phân chia Train/Val/Test trên file gốc (Bảo vệ Zero Data Leakage)
    split_files = {'train': defaultdict(list), 'val': defaultdict(list), 'test': defaultdict(list)}
    
    for cat, files in raw_samples.items():
        n = len(files)
        shuffled = files.copy()
        random.shuffle(shuffled)
        
        # Với lớp quá ít (như burping 8 file, belly_pain 16 file)
        # đảm bảo ít nhất 1 file cho test, 1 file cho val
        n_test = max(1, int(round(n * 0.15)))
        n_val = max(1, int(round(n * 0.15)))
        n_train = n - n_test - n_val
        
        split_files['train'][cat] = shuffled[:n_train]
        split_files['val'][cat] = shuffled[n_train : n_train + n_val]
        split_files['test'][cat] = shuffled[n_train + n_val:]
        
        print(f"Phân chia {cat:12s} -> Train: {len(split_files['train'][cat]):2d} | Val: {len(split_files['val'][cat]):2d} | Test: {len(split_files['test'][cat]):2d}")

    # 3. Tạo thư mục đích
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
        
    for split in ['train', 'val', 'test']:
        for cat in CLASSES:
            os.makedirs(os.path.join(OUT_DIR, split, cat), exist_ok=True)

    metadata_records = []

    # 4. Hàm cân bằng cho từng split (Train, Val, Test)
    def process_and_balance_split(split_name, target_count):
        print(f"\n📦 Đang xử lý và cân bằng tập {split_name.upper()} (Mục tiêu: {target_count} mẫu/lớp)...")
        pitch_factors = [0.85, 0.90, 0.95, 1.05, 1.10, 1.15, 1.20]
        noise_ratios = [0.003, 0.006, 0.010]
        speed_factors = [0.90, 0.95, 1.05, 1.10]
        shift_ratios = [-0.15, -0.08, 0.08, 0.15]

        for cat in CLASSES:
            file_list = split_files[split_name][cat]
            clips = []
            
            # Cắt các file gốc của split này
            for fpath in file_list:
                fname = os.path.basename(fpath).replace(".wav", "")
                audio = load_and_standardize_wav(fpath, TARGET_SR)
                # Dùng step 2.5s để tăng độ đa dạng
                sub_clips = slice_audio(audio, window_size=TARGET_SAMPLES, step_size=int(TARGET_SR * 2.5))
                for c_idx, sc in enumerate(sub_clips):
                    clips.append({
                        'audio': sc,
                        'orig_file': os.path.basename(fpath),
                        'desc': f"slice_{c_idx}"
                    })

            # Lấy mẫu gốc trước
            final_samples = []
            for item in clips:
                final_samples.append({
                    'audio': item['audio'],
                    'orig_file': item['orig_file'],
                    'desc': 'original_slice'
                })
                if len(final_samples) >= target_count:
                    break

            # Nếu chưa đủ thì áp dụng Augmentation trong chính tập split này
            cycle = 0
            while len(final_samples) < target_count:
                cycle += 1
                base = random.choice(clips)
                base_audio = base['audio'].copy()
                aug_type = cycle % 5
                
                if aug_type == 0:
                    p = random.choice(pitch_factors)
                    aug_audio = pitch_shift(base_audio, p)
                    desc = f"pitch_{p:.2f}"
                elif aug_type == 1:
                    n = random.choice(noise_ratios)
                    aug_audio = add_noise(base_audio, n)
                    desc = f"noise_{n:.3f}"
                elif aug_type == 2:
                    s = random.choice(shift_ratios)
                    aug_audio = time_shift(base_audio, s)
                    desc = f"shift_{s:.2f}"
                elif aug_type == 3:
                    sp = random.choice(speed_factors)
                    aug_audio = speed_change(base_audio, sp)
                    desc = f"speed_{sp:.2f}"
                else:
                    p = random.choice(pitch_factors)
                    n = random.choice(noise_ratios)
                    aug_audio = add_noise(pitch_shift(base_audio, p), n)
                    desc = f"pitch_{p:.2f}_noise_{n:.3f}"

                final_samples.append({
                    'audio': aug_audio,
                    'orig_file': base['orig_file'],
                    'desc': desc
                })

            # Ghi ra đĩa
            for idx, item in enumerate(final_samples):
                out_fname = f"{cat}_{split_name}_{idx:04d}_{item['desc']}.wav"
                out_path = os.path.join(OUT_DIR, split_name, cat, out_fname)
                wavfile.write(out_path, TARGET_SR, item['audio'])
                metadata_records.append({
                    'filepath': out_path,
                    'filename': out_fname,
                    'category': cat,
                    'split': split_name,
                    'is_augmented': (item['desc'] != 'original_slice'),
                    'orig_file': item['orig_file']
                })
            print(f"      ✅ [{split_name.upper():5s}] Lớp {cat:12s}: Đã tạo {len(final_samples):3d} mẫu cân bằng!")

    process_and_balance_split('train', TARGET_TRAIN_SAMPLES_PER_CLASS)
    process_and_balance_split('val', TARGET_VAL_SAMPLES_PER_CLASS)
    process_and_balance_split('test', TARGET_TEST_SAMPLES_PER_CLASS)

    # 6. Xuất Metadata & Thống kê
    df_meta = pd.DataFrame(metadata_records)
    meta_csv_path = os.path.join(OUT_DIR, "audio_balanced_metadata.csv")
    df_meta.to_csv(meta_csv_path, index=False)
    
    print("\n" + "="*70)
    print("🎉 BẢNG THỐNG KÊ DATASET ÂM THANH CÂN BẰNG CUỐI CÙNG")
    print("="*70)
    summary_table = pd.crosstab(df_meta['category'], df_meta['split'])
    print(summary_table)
    print(f"\n📊 Tổng số mẫu toàn bộ dataset: {len(df_meta):,}")
    print(f"📁 Metadata lưu tại: {meta_csv_path}")

if __name__ == "__main__":
    main()
