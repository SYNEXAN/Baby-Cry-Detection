import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

REPO_ID = "Nooon/Donate_a_cry"
BASE_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main"
API_URL = f"https://huggingface.co/api/datasets/{REPO_ID}"
DEST_DIR = os.path.join("data", "raw", "donate_a_cry")

def download_one_file(rfilename, session):
    local_path = os.path.join(DEST_DIR, rfilename)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return True, rfilename, None
        
    file_url = f"{BASE_URL}/{rfilename}"
    try:
        r = session.get(file_url, timeout=30)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)
        return True, rfilename, None
    except Exception as e:
        return False, rfilename, str(e)

def main():
    print(f"Fetching dataset file list for {REPO_ID}...")
    resp = requests.get(API_URL)
    resp.raise_for_status()
    data = resp.json()
    
    siblings = data.get("siblings", [])
    files_to_download = [s["rfilename"] for s in siblings if s["rfilename"].endswith(".wav") or s["rfilename"] in ["README.md"]]
    print(f"Found {len(files_to_download)} files to download.")
    
    os.makedirs(DEST_DIR, exist_ok=True)
    
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=3)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    success_count = 0
    fail_count = 0
    
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(download_one_file, f, session): f for f in files_to_download}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading (Multi-threaded)"):
            success, rfilename, err = future.result()
            if success:
                success_count += 1
            else:
                fail_count += 1
                print(f"\nFailed to download {rfilename}: {err}")
                
    print(f"\nFinished: {success_count} succeeded, {fail_count} failed.")

if __name__ == "__main__":
    main()
