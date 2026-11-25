import os
import wave
import csv
from collections import Counter

DATA_DIR = os.path.join("data", "raw", "donate_a_cry")
METADATA_OUT = os.path.join("data", "metadata.csv")

def parse_filename(filename, category):
    # Example format: 549a46d8-9c84-430e-ade8-97eae2bef787-1430130772174-1.7-m-48-bp.wav
    # Format components roughly: [uuid]-[timestamp]-[age_months]-[gender]-[user_age_or_weight?]-[short_code].wav
    base = filename[:-4]
    parts = base.split("-")
    
    # Try heuristic extraction
    # Gender is typically 'm' or 'f'
    gender = "unknown"
    age_months = None
    
    for i, p in enumerate(parts):
        if p in ["m", "f"]:
            gender = "male" if p == "m" else "female"
            if i > 0:
                try:
                    age_months = float(parts[i - 1])
                except ValueError:
                    pass
            break
            
    return {
        "filename": filename,
        "category": category,
        "gender": gender,
        "age_months": age_months
    }

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Directory {DATA_DIR} does not exist.")
        return
        
    records = []
    category_counts = Counter()
    total_duration = 0.0
    corrupted_files = []
    
    categories = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    for cat in sorted(categories):
        cat_dir = os.path.join(DATA_DIR, cat)
        files = [f for f in os.listdir(cat_dir) if f.endswith(".wav")]
        
        for f in files:
            filepath = os.path.join(cat_dir, f)
            rel_path = os.path.join(cat, f)
            file_size = os.path.getsize(filepath)
            
            try:
                with wave.open(filepath, "rb") as wf:
                    channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    framerate = wf.getframerate()
                    nframes = wf.getnframes()
                    duration = nframes / float(framerate) if framerate > 0 else 0
                    
                meta = parse_filename(f, cat)
                meta.update({
                    "relative_path": rel_path,
                    "size_bytes": file_size,
                    "channels": channels,
                    "sample_width_bytes": sampwidth,
                    "sample_rate_hz": framerate,
                    "duration_sec": round(duration, 3)
                })
                records.append(meta)
                category_counts[cat] += 1
                total_duration += duration
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                corrupted_files.append((filepath, str(e)))

    os.makedirs(os.path.dirname(METADATA_OUT), exist_ok=True)
    if records:
        fieldnames = [
            "filename", "category", "relative_path", "duration_sec",
            "sample_rate_hz", "channels", "sample_width_bytes", "size_bytes",
            "age_months", "gender"
        ]
        with open(METADATA_OUT, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
            
    print("=" * 50)
    print("DATASET VERIFICATION & ANALYSIS REPORT")
    print("=" * 50)
    print(f"Total valid audio files: {len(records)}")
    print(f"Corrupted files: {len(corrupted_files)}")
    print(f"Total audio duration: {round(total_duration, 2)} seconds ({round(total_duration/60, 2)} minutes)")
    print("\nClass distribution:")
    for cat, count in category_counts.most_common():
        pct = (count / len(records) * 100) if records else 0
        print(f"  - {cat:12s}: {count:4d} files ({pct:5.1f}%)")
        
    print(f"\nMetadata saved to: {METADATA_OUT}")
    print("=" * 50)

if __name__ == "__main__":
    main()
