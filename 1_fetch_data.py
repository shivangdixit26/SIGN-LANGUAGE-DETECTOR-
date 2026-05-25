"""
1_fetch_data.py
===============
Downloads and extracts the Kaggle ASL Alphabet Dataset.

Prerequisites:
- Kaggle API credentials configured (~/.kaggle/kaggle.json)
- kaggle package installed

Usage:
    python 1_fetch_data.py
"""

import os
import zipfile
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = Path("data")
DATASET_NAME = "grassnick/asl-alphabet"  # Kaggle dataset identifier
EXTRACT_DIR = DATA_DIR / "asl_alphabet"


def setup_directories():
    """Create necessary directories."""
    DATA_DIR.mkdir(exist_ok=True)
    EXTRACT_DIR.mkdir(exist_ok=True)
    print(f"✓ Data directory ready: {DATA_DIR}")


def authenticate_kaggle():
    """Check Kaggle API credentials."""
    creds_path = Path.home() / ".kaggle" / "kaggle.json"
    
    if not creds_path.exists():
        raise FileNotFoundError(
            f"Kaggle credentials not found at {creds_path}\n"
            f"Please download from https://www.kaggle.com/settings/account"
        )
    
    # Verify JSON is valid
    try:
        with open(creds_path) as f:
            creds = json.load(f)
        if "username" in creds and "key" in creds:
            print(f"✓ Kaggle credentials found for user: {creds['username']}")
            return True
        else:
            raise ValueError("Missing 'username' or 'key' in kaggle.json")
    except Exception as e:
        raise ValueError(f"Invalid kaggle.json format: {e}")


def download_dataset():
    """Download the ASL Alphabet dataset using kaggle Python package."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        print(f"\n⏳ Downloading {DATASET_NAME}...")
        
        # Create API instance
        api = KaggleApi()
        api.authenticate()
        
        # Download the dataset
        api.dataset_download_files(
            DATASET_NAME,
            path=str(DATA_DIR),
            unzip=False
        )
        
        print("✓ Dataset downloaded successfully")
        return True
        
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False


def extract_dataset():
    """Extract the downloaded zip file."""
    zip_files = list(DATA_DIR.glob("*.zip"))
    
    if not zip_files:
        print("✗ No .zip file found in data directory")
        return False
    
    for zip_file in zip_files:
        try:
            print(f"\n⏳ Extracting {zip_file.name}...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(EXTRACT_DIR)
            
            # Remove the zip file after extraction
            zip_file.unlink()
            print(f"✓ Extracted and cleaned up {zip_file.name}")
        except Exception as e:
            print(f"✗ Extraction failed: {e}")
            return False
    
    return True


def verify_dataset():
    """Verify the extracted dataset structure."""
    # Look for the dataset directories
    if not EXTRACT_DIR.exists():
        print("✗ Extract directory does not exist")
        return False
    
    # ASL Alphabet dataset typically has train/test splits
    subdirs = list(EXTRACT_DIR.glob("*"))
    
    if not subdirs:
        print("✗ Dataset is empty or not extracted properly")
        return False
    
    print(f"\n✓ Dataset extracted successfully")
    print(f"  Location: {EXTRACT_DIR}")
    print(f"  Contents:")
    
    for subdir in sorted(subdirs)[:10]:  # Show first 10 items
        item_type = "📁" if subdir.is_dir() else "📄"
        print(f"    {item_type} {subdir.name}")
    
    if len(subdirs) > 10:
        print(f"    ... and {len(subdirs) - 10} more items")
    
    # Count image files
    image_files = list(EXTRACT_DIR.rglob("*.jpg")) + list(EXTRACT_DIR.rglob("*.png"))
    print(f"  Total images found: {len(image_files)}")
    
    return len(image_files) > 0


def main():
    """Main execution flow."""
    print("=" * 60)
    print("ASL Alphabet Dataset Downloader")
    print("=" * 60)
    
    try:
        # Step 1: Setup directories
        setup_directories()
        
        # Step 2: Authenticate with Kaggle
        authenticate_kaggle()
        
        # Step 3: Download dataset
        if not download_dataset():
            raise RuntimeError("Dataset download failed")
        
        # Step 4: Extract dataset
        if not extract_dataset():
            raise RuntimeError("Dataset extraction failed")
        
        # Step 5: Verify dataset
        if not verify_dataset():
            raise RuntimeError("Dataset verification failed")
        
        print("\n" + "=" * 60)
        print("✓ Dataset fetch completed successfully!")
        print(f"  Data location: {EXTRACT_DIR}")
        print(f"  Next step: Run 2_extract_features.py")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during dataset fetch: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
