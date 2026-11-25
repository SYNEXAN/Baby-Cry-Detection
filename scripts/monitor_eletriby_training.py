import os
import subprocess
import time
import sys

KERNEL_ID = "vuitang/train-eletriby-4class-audio-classifier"
CONFIG_DIR = r"C:\Users\thiet\.kaggle_accounts\vuitang"
OUTPUT_DIR = "kaggle_eletriby_results"

os.environ["KAGGLE_CONFIG_DIR"] = CONFIG_DIR

print(f"Monitoring Kaggle Kernel: {KERNEL_ID}")

start_time = time.time()
while True:
    cmd = ["kaggle", "kernels", "status", KERNEL_ID]
    res = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)
    output = res.stdout.strip()
    elapsed_min = (time.time() - start_time) / 60
    print(f"[{elapsed_min:.1f}m] Status: {output}")
    
    if "KernelWorkerStatus.COMPLETE" in output:
        print("\nTraining completed successfully! Downloading artifacts...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        dl_cmd = ["kaggle", "kernels", "output", KERNEL_ID, "-p", OUTPUT_DIR]
        dl_res = subprocess.run(dl_cmd, capture_output=True, text=True, env=os.environ)
        print("Download Output:")
        print(dl_res.stdout)
        break
    elif "KernelWorkerStatus.ERROR" in output or "KernelWorkerStatus.CANCELLED" in output:
        print(f"\nKernel finished with error/cancelled status: {output}")
        # Try downloading log anyway
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        subprocess.run(["kaggle", "kernels", "output", KERNEL_ID, "-p", OUTPUT_DIR], capture_output=True, env=os.environ)
        break
        
    time.sleep(30)
