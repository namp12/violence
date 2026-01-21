"""
Download best_model.h5 from external source if not present.
Run this before starting the Flask app.
"""
import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm

# Model file path
MODEL_DIR = Path("/app/models/checkpoints")
MODEL_PATH = MODEL_DIR / "best_model.h5"

# Google Drive direct download link (you need to set this)
# Get this by: Share file → Anyone with link → Copy link → Convert to direct download
MODEL_URL = os.environ.get('MODEL_DOWNLOAD_URL', '')

def download_file(url, destination):
    """Download file with progress bar."""
    print(f"Downloading model from {url[:50]}...")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    with open(destination, 'wb') as f, tqdm(
        desc=destination.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    
    print(f"✓ Model downloaded to {destination}")

def ensure_model():
    """Ensure model file exists, download if necessary."""
    if MODEL_PATH.exists():
        print(f"✓ Model already exists at {MODEL_PATH}")
        return True
    
    if not MODEL_URL:
        print("⚠ MODEL_DOWNLOAD_URL not set. Cannot download model.")
        print("  Set environment variable MODEL_DOWNLOAD_URL in Railway")
        return False
    
    try:
        download_file(MODEL_URL, MODEL_PATH)
        return True
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        return False

if __name__ == "__main__":
    if ensure_model():
        sys.exit(0)
    else:
        sys.exit(1)
