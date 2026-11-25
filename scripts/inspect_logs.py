import glob
import os

logs = glob.glob("kaggle_results/**/*.log", recursive=True)
for log in logs:
    print(f"=== LOG FILE: {log} ===")
    with open(log, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    print(f"Total lines: {len(lines)}")
    for l in lines[-35:]:
        l_str = l.strip()
        if any(term in l_str for term in ["adult", "baby", "mAP", "Precision", "Recall", "Accuracy", "all", "Safe", "Danger"]):
            print("  ", l_str)
