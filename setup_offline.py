#!/usr/bin/env python3
"""
Offline Setup Script for Universal Data Quality Analyzer

This script downloads and caches all necessary models and dependencies 
for offline use of the data quality analysis tools.

Run this once with internet connection to enable offline analysis.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

import torch
import timm
import requests
from tqdm import tqdm

def check_internet_connection():
    """Check if internet connection is available"""
    try:
        response = requests.get("https://www.google.com", timeout=5)
        return response.status_code == 200
    except:
        return False

def install_dependencies():
    """Install required packages"""
    
    print("📦 Installing required packages...")
    
    # Core packages
    packages = [
        "torch",
        "torchvision", 
        "torchaudio",
        "timm>=0.9.0",
        "cleanlab[datalab]>=2.4.0",
        "cleanvision>=0.3.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "Pillow>=8.0.0",
        "tqdm>=4.62.0",
        "requests>=2.25.0"
    ]
    
    for package in packages:
        try:
            print(f"  Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package, "--upgrade"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"  ✅ {package} installed")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Failed to install {package}: {e}")
            return False
    
    return True

def download_models():
    """Download and cache all supported models"""
    
    print("\n🧠 Downloading and caching models...")
    
    # Model configurations with descriptions
    models = {
        "vit_tiny_patch16_224": {
            "description": "Lightweight ViT - Fast inference, good for low-end hardware",
            "size": "~22MB",
            "recommended_for": "Quick analysis, GTX 1050-level GPUs"
        },
        "vit_small_patch16_224": {
            "description": "Balanced ViT - Good speed/accuracy tradeoff", 
            "size": "~88MB",
            "recommended_for": "Most use cases, GTX 1060+ GPUs"
        },
        "vit_base_patch16_224": {
            "description": "Large ViT - High accuracy, slower inference",
            "size": "~342MB", 
            "recommended_for": "High-end hardware, maximum accuracy"
        },
        "resnet50": {
            "description": "CNN baseline - Traditional architecture",
            "size": "~98MB",
            "recommended_for": "CNN comparison, stable features"
        },
        "efficientnet_b0": {
            "description": "Efficient CNN - Good efficiency/accuracy balance",
            "size": "~20MB",
            "recommended_for": "Mobile deployment, efficiency focus"
        },
        "convnext_tiny": {
            "description": "Modern CNN - Latest CNN architecture",
            "size": "~112MB",
            "recommended_for": "Modern CNN alternative to ViT"
        }
    }
    
    successful_downloads = []
    failed_downloads = []
    
    for model_name, info in models.items():
        try:
            print(f"\n📥 Downloading {model_name}...")
            print(f"   Description: {info['description']}")
            print(f"   Size: {info['size']}")
            print(f"   Recommended for: {info['recommended_for']}")
            
            start_time = time.time()
            
            # Download and cache model
            model = timm.create_model(model_name, pretrained=True)
            
            # Get model info
            num_features = model.num_features if hasattr(model, 'num_features') else "Unknown"
            num_params = sum(p.numel() for p in model.parameters())
            
            download_time = time.time() - start_time
            
            print(f"   ✅ Downloaded successfully in {download_time:.1f}s")
            print(f"   📊 Parameters: {num_params:,}")
            print(f"   📐 Features: {num_features}")
            
            successful_downloads.append(model_name)
            
            # Clean up memory
            del model
            
        except Exception as e:
            print(f"   ❌ Failed to download {model_name}: {e}")
            failed_downloads.append(model_name)
    
    # Summary
    print(f"\n📊 Download Summary:")
    print(f"   ✅ Successful: {len(successful_downloads)}")
    print(f"   ❌ Failed: {len(failed_downloads)}")
    
    if successful_downloads:
        print(f"\n✅ Successfully cached models:")
        for model in successful_downloads:
            print(f"   - {model}")
    
    if failed_downloads:
        print(f"\n❌ Failed to cache models:")
        for model in failed_downloads:
            print(f"   - {model}")
    
    return len(failed_downloads) == 0

def verify_installation():
    """Verify that all components are working correctly"""
    
    print("\n🔍 Verifying installation...")
    
    verification_tests = [
        ("PyTorch", lambda: torch.__version__),
        ("CUDA availability", lambda: torch.cuda.is_available()),
        ("timm", lambda: timm.__version__),
        ("Model loading", lambda: timm.create_model("vit_tiny_patch16_224", pretrained=True) is not None)
    ]
    
    results = []
    
    for test_name, test_func in verification_tests:
        try:
            result = test_func()
            print(f"   ✅ {test_name}: {result}")
            results.append(True)
        except Exception as e:
            print(f"   ❌ {test_name}: Failed - {e}")
            results.append(False)
    
    # Test optional components
    optional_tests = [
        ("CleanLab", lambda: __import__("cleanlab").__version__),
        ("CleanVision", lambda: __import__("cleanvision").__version__)
    ]
    
    print("\n🔧 Optional components:")
    for test_name, test_func in optional_tests:
        try:
            result = test_func()
            print(f"   ✅ {test_name}: {result}")
        except ImportError:
            print(f"   ⚠️  {test_name}: Not installed (optional)")
        except Exception as e:
            print(f"   ❌ {test_name}: Error - {e}")
    
    return all(results)

def create_test_dataset():
    """Create a small test dataset for verification"""
    
    print("\n🧪 Creating test dataset...")
    
    test_dir = Path("test_dataset")
    
    # Create directory structure
    classes = ["class1", "class2", "class3"]
    
    for class_name in classes:
        class_dir = test_dir / "train" / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy images
    try:
        from PIL import Image
        import numpy as np
        
        for class_idx, class_name in enumerate(classes):
            class_dir = test_dir / "train" / class_name
            
            for i in range(5):  # 5 images per class
                # Create random colored image
                img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                
                # Add some class-specific pattern
                if class_idx == 0:  # Red-ish
                    img_array[:, :, 0] = np.minimum(img_array[:, :, 0] + 50, 255)
                elif class_idx == 1:  # Green-ish
                    img_array[:, :, 1] = np.minimum(img_array[:, :, 1] + 50, 255)
                else:  # Blue-ish
                    img_array[:, :, 2] = np.minimum(img_array[:, :, 2] + 50, 255)
                
                img = Image.fromarray(img_array)
                img.save(class_dir / f"image_{i:02d}.png")
        
        print(f"   ✅ Test dataset created at: {test_dir}")
        print(f"   📊 Structure: 3 classes, 5 images each")
        
        return str(test_dir / "train")
        
    except Exception as e:
        print(f"   ❌ Failed to create test dataset: {e}")
        return None

def run_test_analysis(test_dataset_path):
    """Run a quick test analysis"""
    
    if not test_dataset_path:
        print("⏭️  Skipping test analysis (no test dataset)")
        return False
    
    print("\n🧪 Running test analysis...")
    
    try:
        # Import the main analyzer
        sys.path.append(str(Path(__file__).parent))
        
        # Run minimal analysis
        import subprocess
        
        cmd = [
            sys.executable,
            "universal_data_quality_analyzer.py",
            "--data", test_dataset_path,
            "--output", "test_output",
            "--preset", "lightweight",
            "--skip-cleanvision",  # Skip to save time
            "--cv-folds", "2",     # Minimal CV
            "--max-samples", "10"  # Limit samples
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("   ✅ Test analysis completed successfully")
            return True
        else:
            print(f"   ❌ Test analysis failed:")
            print(f"   Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Test analysis failed: {e}")
        return False

def cleanup_test_files():
    """Clean up test files"""
    
    print("\n🧹 Cleaning up test files...")
    
    import shutil
    
    test_paths = ["test_dataset", "test_output"]
    
    for path in test_paths:
        if Path(path).exists():
            try:
                shutil.rmtree(path)
                print(f"   ✅ Removed {path}")
            except Exception as e:
                print(f"   ⚠️  Could not remove {path}: {e}")

def print_usage_guide():
    """Print usage guide after successful setup"""
    
    print("\n" + "=" * 80)
    print("🎉 SETUP COMPLETE - READY FOR OFFLINE USE!")
    print("=" * 80)
    
    print("\n📖 USAGE GUIDE:")
    print("\n1. Basic Analysis:")
    print("   python universal_data_quality_analyzer.py --data your_dataset/train")
    
    print("\n2. With Custom Settings:")
    print("   python universal_data_quality_analyzer.py \\")
    print("     --data your_dataset/train \\")
    print("     --output my_reports \\")
    print("     --preset balanced \\")
    print("     --normalization imagenet")
    
    print("\n3. Quick Image Quality Check:")
    print("   python universal_data_quality_analyzer.py \\")
    print("     --data your_dataset/train \\")
    print("     --skip-cleanlab \\")
    print("     --preset lightweight")
    
    print("\n📁 EXPECTED DATASET STRUCTURE:")
    print("   your_dataset/")
    print("   ├── train/")
    print("   │   ├── class1/")
    print("   │   │   ├── image1.jpg")
    print("   │   │   ├── image2.png")
    print("   │   │   └── ...")
    print("   │   ├── class2/")
    print("   │   │   └── ...")
    print("   │   └── classN/")
    print("   │       └── ...")
    
    print("\n🎯 PRESETS AVAILABLE:")
    presets = {
        "lightweight": "Fast analysis for low-end hardware",
        "balanced": "Good speed/accuracy balance (recommended)",
        "accurate": "Maximum accuracy for high-end hardware",
        "cnn_fast": "CNN-based feature extraction",
        "efficient": "Efficient architecture"
    }
    
    for preset, desc in presets.items():
        print(f"   {preset}: {desc}")
    
    print("\n📊 OUTPUT FILES:")
    print("   data_quality_reports/")
    print("   ├── cleanvision/          # Image quality issues")
    print("   ├── cleanlab/             # Label quality issues")
    print("   └── analysis_metadata.json # Analysis summary")
    
    print("\n💡 TROUBLESHOOTING:")
    print("   - GPU memory error: Use --preset lightweight or --batch-size 8")
    print("   - Model download fails: Re-run this setup script")
    print("   - CleanLab fails: Check basic_quality_metrics.csv instead")
    print("   - Windows multiprocessing: Keep --num-workers 0 (default)")

def main():
    """Main setup process"""
    
    print("🚀 UNIVERSAL DATA QUALITY ANALYZER - OFFLINE SETUP")
    print("=" * 80)
    print("This script will prepare your system for offline data quality analysis.")
    print("Internet connection is required for initial setup only.")
    print("=" * 80)
    
    # Check internet connection
    if not check_internet_connection():
        print("❌ No internet connection detected.")
        print("Please connect to the internet and run this script again.")
        return 1
    
    print("✅ Internet connection verified")
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Failed to install dependencies.")
        print("Please check your internet connection and try again.")
        return 1
    
    # Download models
    if not download_models():
        print("\n⚠️  Some models failed to download.")
        print("You can still use the analyzer with the models that downloaded successfully.")
    
    # Verify installation
    if not verify_installation():
        print("\n❌ Installation verification failed.")
        print("Please check the error messages above and try again.")
        return 1
    
    # Create and test with dummy dataset
    test_dataset_path = create_test_dataset()
    test_success = run_test_analysis(test_dataset_path)
    
    if test_success:
        print("✅ All tests passed - system ready for offline use!")
    else:
        print("⚠️  Test analysis failed, but core components are installed.")
        print("You can still try using the analyzer manually.")
    
    # Cleanup
    cleanup_test_files()
    
    # Print usage guide
    print_usage_guide()
    
    print("\n🔗 For more information and examples, see:")
    print("   - README.md")
    print("   - DATASET_SETUP_GUIDE.md")
    
    return 0

if __name__ == "__main__":
    exit(main())