import os
import sys
import time
import subprocess

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
else:
    sys.stdout.reconfigure(line_buffering=True)

acc_dir = r"C:\Users\thiet\.kaggle_accounts\thittn"
env = os.environ.copy()
env["KAGGLE_CONFIG_DIR"] = acc_dir

kernel_slug = "thittn/train-yolo11-m-full-body-v7"
output_dir = r"kaggle_results\version7"
os.makedirs(output_dir, exist_ok=True)

print(f"Monitoring Kaggle Kernel: {kernel_slug}")
print(f"Artifacts will be saved to: {output_dir}")

start_time = time.time()
last_status = None

while True:
    res = subprocess.run(["kaggle", "kernels", "status", kernel_slug], env=env, capture_output=True, text=True)
    status_str = res.stdout.strip()
    elapsed_min = (time.time() - start_time) / 60
    
    if status_str != last_status or int(elapsed_min) % 2 == 0:
        print(f"[{elapsed_min:.1f}m] Status: {status_str}")
        last_status = status_str

    if "COMPLETE" in status_str:
        print("\n🎉 KERNEL EXECUTION COMPLETE! Downloading trained weights & visual artifacts...")
        dl_res = subprocess.run(["kaggle", "kernels", "output", kernel_slug, "-p", output_dir], env=env, capture_output=True, text=True)
        print("Download Output:")
        print(dl_res.stdout)
        if dl_res.stderr:
            print(dl_res.stderr)
        print(f"Artifacts successfully downloaded to {output_dir}!")
        break
    elif "ERROR" in status_str or "CANCELLED" in status_str:
        print(f"\n❌ Kernel ended with status: {status_str}")
        log_res = subprocess.run(["kaggle", "kernels", "output", kernel_slug, "-p", output_dir], env=env, capture_output=True, text=True)
        print("Error Logs / Output:")
        print(log_res.stdout)
        print(log_res.stderr)
        break

    time.sleep(45)
