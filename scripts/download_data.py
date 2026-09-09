"""
Dataset Downloader for Twitter Customer Support Dataset
Fetches 'thoughtvector/customer-support-on-twitter' using kagglehub or Kaggle API.
Places twcs.csv directly into data/raw/twcs.csv.
"""

import sys
import shutil
from pathlib import Path

DATA_RAW_DIR = Path("data/raw")
TARGET_FILE = DATA_RAW_DIR / "twcs.csv"

def check_existing() -> bool:
    if TARGET_FILE.exists() and TARGET_FILE.stat().st_size > 10_000_000:
        print(f"[OK] twcs.csv already exists at {TARGET_FILE.resolve()} ({TARGET_FILE.stat().st_size / (1024*1024):.2f} MB)")
        return True
    return False

def download_via_kagglehub() -> bool:
    print("[*] Attempting download via kagglehub...")
    try:
        import kagglehub
        path = kagglehub.dataset_download("thoughtvector/customer-support-on-twitter")
        downloaded_dir = Path(path)
        print(f"[+] Download completed to cache: {downloaded_dir}")
        source_file = downloaded_dir / "twcs.csv"
        if not source_file.exists():
            # Check for any csv file in downloaded directory
            csvs = list(downloaded_dir.glob("*.csv"))
            if csvs:
                source_file = csvs[0]
        if source_file.exists():
            DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
            print(f"[*] Copying {source_file} to {TARGET_FILE}...")
            shutil.copy2(source_file, TARGET_FILE)
            print(f"[SUCCESS] twcs.csv placed at {TARGET_FILE.resolve()} ({TARGET_FILE.stat().st_size / (1024*1024):.2f} MB)")
            return True
    except Exception as e:
        print(f"[-] kagglehub download failed: {e}")
    return False

def print_manual_instructions():
    print("=" * 70)
    print("MANUAL DOWNLOAD INSTRUCTIONS:")
    print("=" * 70)
    print("1. Visit: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter")
    print("2. Click 'Download' (archive.zip ~169MB compressed / ~750MB uncompressed)")
    print(f"3. Unzip and place 'twcs.csv' directly into: {TARGET_FILE.resolve()}")
    print("=" * 70)

def main():
    if check_existing():
        sys.exit(0)
    
    success = download_via_kagglehub()
    if not success:
        print_manual_instructions()
        sys.exit(1)

if __name__ == "__main__":
    main()
