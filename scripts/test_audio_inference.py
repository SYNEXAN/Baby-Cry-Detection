import py_compile
import glob
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

print("--- Step 1: Compiling source files ---")
py_compile.compile('engine/audio_detector.py', doraise=True)
py_compile.compile('app.py', doraise=True)
print("Compilation SUCCESS: Both engine/audio_detector.py and app.py compiled with 0 errors!")

print("\n--- Step 2: Instantiating AudioCryClassifier ---")
from engine.audio_detector import AudioCryClassifier
clf = AudioCryClassifier()
print(f"Loaded model name: {clf.model_name}")
print(f"Active classes: {clf.classes}")
print(f"Sampling rate: {clf.duration_sec}s target window")
assert clf.is_loaded, "Model must be loaded!"

print("\n--- Step 3: Running Inference on Real Baby Cry Recordings ---")
samples = {
    'burping': glob.glob('data/raw/donate_a_cry/burping/*.wav')[0],
    'belly_pain': glob.glob('data/raw/donate_a_cry/belly_pain/*.wav')[0],
    'hungry': glob.glob('data/raw/donate_a_cry/hungry/*.wav')[0],
    'tired': glob.glob('data/raw/donate_a_cry/tired/*.wav')[0],
    'discomfort': glob.glob('data/raw/donate_a_cry/discomfort/*.wav')[0],
}

for label, fpath in samples.items():
    res = clf.predict(fpath)
    print(f"\n[Ground Truth: {label.upper()}] (File: {os.path.basename(fpath)})")
    print(f" -> Predicted Cause: {res['vietnamese_cause']} (Confidence: {res['confidence']*100:.2f}%)")
    print(f" -> Feature shape: {res['feature_shape']}")
    print(" -> Probabilities breakdown:")
    for vi_k, prob in res['probabilities'].items():
        print(f"    * {vi_k}: {prob*100:.2f}%")
    print(f" -> Advice sample: {res['advice'].splitlines()[0]}")

print("\nIntegration test completed successfully!")
