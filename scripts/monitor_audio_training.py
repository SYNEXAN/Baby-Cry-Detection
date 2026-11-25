import time
import os
import sys

os.environ["KAGGLE_CONFIG_DIR"] = r"C:\Users\thiet\.kaggle_accounts\jtachitv"
from kaggle.api.kaggle_api_extended import KaggleApi
KERNEL_NAME = "jtachitv/train-audio-cry-classifier-5-classes"
OUTPUT_DIR = os.path.join("kaggle_results", "audio")
os.makedirs(OUTPUT_DIR, exist_ok=True)

api = KaggleApi()
api.authenticate()

print(f"Monitoring Kaggle Kernel: {KERNEL_NAME}")
print(f"Account: {api.get_config_value('username')}")
print(f"Destination folder for outputs: {OUTPUT_DIR}")

start_time = time.time()
while True:
    try:
        status_obj = api.kernels_status(KERNEL_NAME)
        if isinstance(status_obj, dict):
            status = str(status_obj.get("status", "UNKNOWN")).upper()
            failure = status_obj.get("failureMessage", None)
        else:
            status = str(getattr(status_obj, "status", "UNKNOWN")).upper()
            failure = getattr(status_obj, "failure_message", getattr(status_obj, "failureMessage", None))
            
        elapsed = int(time.time() - start_time)
        print(f"[{elapsed:04d}s] Status: {status} | Failure: {failure}", flush=True)
        
        if "COMPLETE" in status:
            print("\n🎉 Kernel completed successfully! Downloading outputs...", flush=True)
            api.kernels_output(KERNEL_NAME, path=OUTPUT_DIR)
            print(f"✅ Outputs downloaded to: {OUTPUT_DIR}", flush=True)
            break
        elif any(err in status for err in ["ERROR", "CANCELLED", "FAILED"]):
            print(f"\n❌ Kernel terminated with status: {status}, failure message: {failure}", flush=True)
            try:
                logs = api.kernels_logs(KERNEL_NAME)
                print(f"Logs: {logs}", flush=True)
            except Exception as e:
                print(f"Failed to fetch logs: {e}", flush=True)
            break
            
    except Exception as e:
        print(f"Error checking status: {e}", flush=True)
        
    time.sleep(25)
