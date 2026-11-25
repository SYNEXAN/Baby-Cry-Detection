import os
import subprocess
import json
import shutil

probe_dir = "test_quota_probe"
os.makedirs(probe_dir, exist_ok=True)
with open(os.path.join(probe_dir, "probe.py"), "w") as f:
    f.write('print("GPU probe")\n')

accounts = ['thitnguynthanh', 'jtachitv', 'thitenguyen', 'nguynthanhthit', 'thietnguyen']
base = r"C:\Users\thiet\.kaggle_accounts"

selected_account = None

for acc in accounts:
    acc_dir = os.path.join(base, acc)
    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = acc_dir
    
    meta = {
        "id": f"{acc}/gpu-quota-probe",
        "title": "GPU Quota Probe",
        "code_file": "probe.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": []
    }
    with open(os.path.join(probe_dir, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
        
    res = subprocess.run(["kaggle", "kernels", "push", "-p", probe_dir], env=env, capture_output=True, text=True)
    out = (res.stdout + res.stderr).strip()
    print(f"Account: {acc} -> code: {res.returncode} | out: {out[:120]}")
    if "successfully pushed" in out.lower():
        print(f"  >>> FOUND AVAILABLE GPU ACCOUNT: {acc} <<<")
        selected_account = acc
        break
    elif "maximum weekly gpu quota" in out.lower():
        print(f"  Quota reached on {acc}")

shutil.rmtree(probe_dir, ignore_errors=True)
print(f"\nFinal Selected Account with Available GPU: {selected_account}")
