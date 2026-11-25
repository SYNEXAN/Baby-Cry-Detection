import os
import json
import subprocess
import shutil

accounts = ['thitnguynthanh', 'jtachitv', 'thitenguyen', 'nguynthanhthit', 'thietnguyen']
base = r'C:\Users\thiet\.kaggle_accounts'

nb_content = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["import torch\n", "print(torch.cuda.is_available())\n"]
        }
    ],
    "metadata": {
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

results = {}

for acc in accounts:
    acc_dir = os.path.join(base, acc)
    env = os.environ.copy()
    env['KAGGLE_CONFIG_DIR'] = acc_dir
    
    test_dir = f'test_probe_{acc}'
    os.makedirs(test_dir, exist_ok=True)
    with open(os.path.join(test_dir, "test.ipynb"), "w") as f:
        json.dump(nb_content, f)
        
    meta = {
        "id": f"{acc}/gpu-test-{acc}",
        "title": f"GPU Test {acc}",
        "code_file": "test.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": []
    }
    with open(os.path.join(test_dir, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
        
    res = subprocess.run(["kaggle", "kernels", "push", "-p", test_dir], env=env, capture_output=True, text=True)
    out = (res.stdout + res.stderr).strip()
    results[acc] = {
        "code": res.returncode,
        "output": out
    }
    print(f"[{acc}] -> {out}")
    shutil.rmtree(test_dir, ignore_errors=True)

print("\n=== SUMMARY ===")
for acc, r in results.items():
    print(f"{acc}: {r['output']}")
