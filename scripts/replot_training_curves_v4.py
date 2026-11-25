"""
Script: scripts/replot_training_curves_v4.py
Purpose: Re-generates kaggle_eletriby_results/training_curves_eletriby.png directly
from the real Kaggle Version 4 training log with exact identical formatting,
fixing 1-based Epoch X-axis (1..25) and aligning the Best Checkpoint red line at Epoch 21.
"""

import json
import re
import shutil
from pathlib import Path
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT_DIR / "kaggle_eletriby_results" / "train-eletriby-4class-audio-classifier.log"
OUT_PNG = ROOT_DIR / "kaggle_eletriby_results" / "training_curves_eletriby.png"

pattern = re.compile(
    r"Epoch \[\s*(\d+)/\s*\d+\] \| Train Loss: ([0-9.]+) \| Val Loss: ([0-9.]+) \| "
    r"Val Acc: ([0-9.]+)% \| Val Macro F1: ([0-9.]+)(.*)"
)

with open(LOG_PATH, "r", encoding="utf-8") as f:
    entries = json.load(f)

epochs = []
train_loss = []
val_loss = []
val_acc = []
val_f1 = []
best_epoch = 0

for entry in entries:
    data = entry.get("data", "")
    m = pattern.search(data)
    if m:
        ep = int(m.group(1))
        tl = float(m.group(2))
        vl = float(m.group(3))
        va = float(m.group(4))
        vf = float(m.group(5)) * 100.0
        note = m.group(6)
        epochs.append(ep)
        train_loss.append(tl)
        val_loss.append(vl)
        val_acc.append(va)
        val_f1.append(vf)
        if "Optimal Checkpoint Saved" in note:
            best_epoch = ep

assert len(epochs) == 25, f"Expected 25 epochs, parsed {len(epochs)}"
assert epochs[0] == 1 and epochs[-1] == 25, f"Unexpected epoch range: {epochs[0]}..{epochs[-1]}"
assert best_epoch == 21, f"Expected best_epoch=21, got {best_epoch}"

print(f"Parsed {len(epochs)} epochs (Ep {epochs[0]}..{epochs[-1]}), Best Checkpoint: Ep {best_epoch}")
print(f"Ep 1  : Train Loss={train_loss[0]:.4f}, Val Loss={val_loss[0]:.4f}, Val Acc={val_acc[0]:.2f}%, Val F1={val_f1[0]:.2f}%")
print(f"Ep 21 : Train Loss={train_loss[20]:.4f}, Val Loss={val_loss[20]:.4f}, Val Acc={val_acc[20]:.2f}%, Val F1={val_f1[20]:.2f}%")
print(f"Ep 25 : Train Loss={train_loss[24]:.4f}, Val Loss={val_loss[24]:.4f}, Val Acc={val_acc[24]:.2f}%, Val F1={val_f1[24]:.2f}%")

xticks = [1] + [e for e in range(5, len(epochs) + 1, 5)]

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label="Train Loss", color="blue", lw=1.5)
plt.plot(epochs, val_loss, label="Val Loss", color="darkorange", lw=1.5)
if best_epoch > 0:
    plt.axvline(best_epoch, color="red", linestyle="--", alpha=0.7, label=f"Best Checkpoint (Ep {best_epoch})")
plt.title("Weighted Loss Progression", fontsize=11, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(xticks)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(epochs, val_acc, label="Val Accuracy (%)", color="green", lw=1.5)
plt.plot(epochs, val_f1, label="Val Macro F1 (%)", color="purple", lw=1.5)
plt.title("Validation Performance Across 4 Classes", fontsize=11, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Percentage (%)")
plt.xticks(xticks)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(str(OUT_PNG), dpi=300)
plt.close()

print(f"Successfully saved fixed training curves plot to: {OUT_PNG}")
