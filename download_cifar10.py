#!/usr/bin/env python3
"""
Download and extract CIFAR-10 dataset
"""

import os
import urllib.request
import tarfile
from pathlib import Path

def download_cifar10(data_dir="data"):
    """Download and extract CIFAR-10 dataset"""
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    cifar_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    cifar_file = data_path / "cifar-10-python.tar.gz"
    cifar_extract_dir = data_path / "cifar-10-batches-py"
    
    # Check if already downloaded and extracted
    if cifar_extract_dir.exists() and (cifar_extract_dir / "batches.meta").exists():
        print(f"✅ CIFAR-10 already exists at: {cifar_extract_dir}")
        return str(cifar_extract_dir)
    
    # Download if not exists
    if not cifar_file.exists():
        print(f"📥 Downloading CIFAR-10 from {cifar_url}...")
        print("This may take a few minutes...")
        urllib.request.urlretrieve(cifar_url, cifar_file)
        print(f"✅ Downloaded to: {cifar_file}")
    
    # Extract
    print(f"📂 Extracting CIFAR-10...")
    with tarfile.open(cifar_file, 'r:gz') as tar:
        tar.extractall(data_path)
    
    print(f"✅ CIFAR-10 extracted to: {cifar_extract_dir}")
    
    # Verify extraction
    required_files = [
        "batches.meta",
        "data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5",
        "test_batch"
    ]
    
    missing_files = []
    for file in required_files:
        if not (cifar_extract_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files after extraction: {missing_files}")
        return None
    
    print("✅ All CIFAR-10 files verified!")
    return str(cifar_extract_dir)

if __name__ == "__main__":
    download_cifar10()
