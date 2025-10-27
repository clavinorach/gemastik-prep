#!/usr/bin/env python3
"""
CleanVision + Cleanlab untuk CIFAR-10 offline mode.
Ekstrak CIFAR-10 pickle -> folder struktur -> analisis data quality.
"""

import os, argparse, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Set multiprocessing start method for macOS compatibility
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# Try different cleanlab imports based on version
try:
    from cleanlab import Datalab  # cleanlab >= 2.0
    CLEANLAB_VERSION = "2.x"
except ImportError:
    try:
        from cleanlab.datalab import Datalab  # cleanlab 1.x
        CLEANLAB_VERSION = "1.x"
    except ImportError:
        print("Warning: Cleanlab not properly installed. Install with: pip install cleanlab>=2.0")
        Datalab = None
        CLEANLAB_VERSION = None

try:
    from cleanvision import Imagelab
    CLEANVISION_AVAILABLE = True
except ImportError:
    print("Warning: CleanVision not installed. Install with: pip install cleanvision")
    CLEANVISION_AVAILABLE = False

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import timm
from tqdm import tqdm

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)

# -------- CIFAR-10 Unpacker --------
def unpickle(file):
    """Load CIFAR-10 pickle batch file"""
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def extract_cifar10_to_folders(cifar_dir, output_dir):
    """Extract CIFAR-10 pickle files to train/test folders"""
    cifar_path = Path(cifar_dir)
    output_path = Path(output_dir)
    
    # Load meta data for class names
    meta = unpickle(cifar_path / "batches.meta")
    class_names = [name.decode('utf-8') for name in meta[b'label_names']]
    
    # Create directory structure
    train_dir = output_path / "train"
    test_dir = output_path / "test"
    
    for split_dir in [train_dir, test_dir]:
        split_dir.mkdir(parents=True, exist_ok=True)
        for class_name in class_names:
            (split_dir / class_name).mkdir(exist_ok=True)
    
    # Extract training data (5 batches)
    print("Extracting training data...")
    train_images, train_labels = [], []
    for i in range(1, 6):
        batch = unpickle(cifar_path / f"data_batch_{i}")
        train_images.append(batch[b'data'])
        train_labels.extend(batch[b'labels'])
    
    train_images = np.concatenate(train_images, axis=0)
    train_images = train_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    # Save training images
    for idx, (img, label) in enumerate(tqdm(zip(train_images, train_labels), desc="Saving train")):
        class_name = class_names[label]
        img_pil = Image.fromarray(img)
        img_pil.save(train_dir / class_name / f"train_{idx:05d}.png")
    
    # Extract test data
    print("Extracting test data...")
    test_batch = unpickle(cifar_path / "test_batch")
    test_images = test_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_labels = test_batch[b'labels']
    
    # Save test images
    for idx, (img, label) in enumerate(tqdm(zip(test_images, test_labels), desc="Saving test")):
        class_name = class_names[label]
        img_pil = Image.fromarray(img)
        img_pil.save(test_dir / class_name / f"test_{idx:05d}.png")
    
    return class_names

# -------- Dataset --------
class LocalImageFolder(Dataset):
    def __init__(self, root, img_size=224):
        self.root = Path(root)
        assert self.root.exists(), f"Folder tidak ditemukan: {root}"
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.samples = []
        for ci, c in enumerate(self.classes):
            for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
                for p in (self.root / c).glob(ext):
                    self.samples.append((p, ci))
        assert len(self.samples)>0, "Tidak ada gambar ditemukan."
        
        # CIFAR-10 optimized transforms
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # CIFAR-10 stats
        ])
    
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, i):
        p, y = self.samples[i]
        try:
            img = Image.open(p).convert("RGB")
            return self.tf(img), y, str(p)
        except Exception as e:
            # Fallback for corrupted images
            print(f"Warning: Failed to load {p}: {e}")
            # Return black image as fallback
            img = Image.new('RGB', (32, 32), (0, 0, 0))
            return self.tf(img), y, str(p)

# -------- Features + OOF --------
def extract_features(backbone, folder, batch=64, device="cpu"):
    """Extract features using pretrained backbone"""
    ds = LocalImageFolder(folder, img_size=224)
    # Set num_workers=0 for better macOS compatibility
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0, pin_memory=(device=="cuda"))
    
    print(f"Loading backbone: {backbone}")
    try:
        model = timm.create_model(backbone, pretrained=True, num_classes=0).to(device).eval()
    except Exception as e:
        print(f"Failed to load {backbone}, falling back to resnet18: {e}")
        model = timm.create_model("resnet18", pretrained=True, num_classes=0).to(device).eval()
    
    feats, labels, names = [], [], []
    with torch.no_grad():
        for x, y, paths in tqdm(dl, desc="Extract features"):
            x = x.to(device)
            f = model(x).cpu().numpy()
            feats.append(f)
            labels.extend(y.numpy())
            names.extend(paths)
    
    feats = np.concatenate(feats, 0)
    labels = np.array(labels, int)
    return feats, labels, names, ds.classes

def oof_probs(feats, labels, K, splits=3):
    """Generate out-of-fold predictions using cross-validation"""
    pred = np.zeros((len(labels), K), float)
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=SEED)
    
    for fold, (tr, va) in enumerate(skf.split(feats, labels)):
        print(f"Training fold {fold+1}/{splits}...")
        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=1000, solver="liblinear", random_state=SEED))
        ])
        pipe.fit(feats[tr], labels[tr])
        pred[va] = pipe.predict_proba(feats[va])
    
    return pred

# -------- Download CIFAR-10 --------
def download_cifar10_if_needed(cifar_dir):
    """Download CIFAR-10 if not present"""
    cifar_path = Path(cifar_dir)
    
    # Check if CIFAR-10 exists
    if cifar_path.exists() and (cifar_path / "batches.meta").exists():
        return str(cifar_path)
    
    print(f"CIFAR-10 not found at {cifar_dir}")
    print("Attempting to download CIFAR-10...")
    
    try:
        import urllib.request
        import tarfile
        
        # Create parent directory
        cifar_path.parent.mkdir(parents=True, exist_ok=True)
        
        cifar_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        cifar_file = cifar_path.parent / "cifar-10-python.tar.gz"
        
        # Download
        print(f"📥 Downloading from {cifar_url}...")
        print("This may take a few minutes...")
        urllib.request.urlretrieve(cifar_url, cifar_file)
        
        # Extract
        print(f"📂 Extracting...")
        with tarfile.open(cifar_file, 'r:gz') as tar:
            tar.extractall(cifar_path.parent)
        
        # Clean up tar file
        cifar_file.unlink()
        
        # Verify
        if (cifar_path / "batches.meta").exists():
            print(f"✅ CIFAR-10 downloaded and extracted to: {cifar_path}")
            return str(cifar_path)
        else:
            print("❌ Download failed - batches.meta not found")
            return None
            
    except Exception as e:
        print(f"❌ Failed to download CIFAR-10: {e}")
        print("Please download manually from: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")
        print(f"Extract to: {cifar_path}")
        return None

# -------- Main --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cifar_dir", type=str, default="data/cifar-10-batches-py", 
                    help="Path to CIFAR-10 pickle files")
    ap.add_argument("--extract_dir", type=str, default="data/cifar10_extracted", 
                    help="Output directory for extracted images")
    ap.add_argument("--backbone", type=str, default="resnet18",
                    help="Feature extraction backbone")
    ap.add_argument("--batch", type=int, default=32, help="Batch size for feature extraction")
    ap.add_argument("--out", type=str, default="data_quality_reports", help="Output directory for reports")
    ap.add_argument("--skip-extract", action="store_true", help="Skip extraction if already done")
    ap.add_argument("--auto-download", action="store_true", help="Automatically download CIFAR-10 if missing")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Cleanlab version: {CLEANLAB_VERSION}")
    print(f"CleanVision available: {CLEANVISION_AVAILABLE}")

    # 0) Download CIFAR-10 if needed
    if args.auto_download or not Path(args.cifar_dir).exists():
        cifar_dir = download_cifar10_if_needed(args.cifar_dir)
        if cifar_dir is None:
            print("❌ Cannot proceed without CIFAR-10 dataset")
            print("Run with --auto-download flag or download manually")
            return
        args.cifar_dir = cifar_dir

    # 1) Extract CIFAR-10 if needed
    train_dir = Path(args.extract_dir) / "train"
    if not args.skip_extract or not train_dir.exists():
        print("== Extracting CIFAR-10 to image folders ==")
        try:
            class_names = extract_cifar10_to_folders(args.cifar_dir, args.extract_dir)
            print(f"Extracted {len(class_names)} classes: {class_names}")
        except FileNotFoundError as e:
            print(f"❌ CIFAR-10 files not found: {e}")
            print("Try running with --auto-download flag")
            return
    else:
        print("== Skipping extraction (already done) ==")

    # 2) CleanVision (image quality issues)
    if CLEANVISION_AVAILABLE:
        print("\n== Running CleanVision ==")
        cv_out = Path(args.out) / "cleanvision"
        cv_out.mkdir(parents=True, exist_ok=True)
        
        try:
            imagelab = Imagelab(data_path=str(train_dir))
            imagelab.find_issues(verbose=False)
            
            # Try different save methods based on CleanVision version
            try:
                imagelab.save(str(cv_out), force=True)
            except TypeError:
                try:
                    imagelab.report(str(cv_out), force=True)
                except TypeError:
                    try:
                        imagelab.save(str(cv_out))
                    except Exception as save_error:
                        print(f"CleanVision save failed: {save_error}")
                        # Create a basic summary instead
                        issues_summary = imagelab.get_issues()
                        if issues_summary is not None:
                            issues_df = pd.DataFrame(issues_summary)
                            issues_df.to_csv(cv_out / "cleanvision_issues.csv", index=False)
                            print(f"CleanVision issues saved to CSV: {cv_out}")
            
            print(f"CleanVision analysis completed")
        except Exception as e:
            print(f"CleanVision failed: {e}")
            print("Continuing with feature extraction...")
    else:
        print("\n== Skipping CleanVision (not installed) ==")

    # 3) Feature extraction + OOF predictions
    print(f"\n== Extracting features with {args.backbone} ==")
    feats, labels, names, classes = extract_features(args.backbone, train_dir, args.batch, device)
    print(f"Extracted {feats.shape[0]} samples, {feats.shape[1]} features")
    
    print("== Generating out-of-fold predictions ==")
    probs = oof_probs(feats, labels, K=len(classes), splits=3)

    # 4) Cleanlab (label/outlier/near-duplicate detection)
    if Datalab is not None:
        print("\n== Running Cleanlab analysis ==")
        
        try:
            # Create dataset - try different API versions
            try:
                # Method 1: Modern Cleanlab 2.x API
                lab = Datalab(data={"features": feats}, label="labels")
                lab.find_issues(features=feats, labels=labels, pred_probs=probs)
            except Exception:
                try:
                    # Method 2: Alternative API
                    data_dict = {"features": feats, "labels": labels}
                    lab = Datalab(data=data_dict, label_name="labels")
                    lab.find_issues(pred_probs=probs)
                except Exception:
                    # Method 3: Direct approach
                    lab = Datalab(data=feats, label=labels)
                    lab.find_issues(pred_probs=probs)
            
            # Save results
            cl_out = Path(args.out) / "cleanlab"
            cl_out.mkdir(parents=True, exist_ok=True)
            
            # Get issues summary
            issues = lab.get_issues()
            
            # Add filenames to issues
            issues_df = pd.DataFrame(issues)
            issues_df["filename"] = names
            issues_df["class_name"] = [classes[label] for label in labels]
            
            # Save with error handling
            try:
                issues_df.to_csv(cl_out / "cleanlab_issues_full.csv", index=False)
                
                # Save compact report
                important_cols = ["filename", "class_name"] + [col for col in issues_df.columns 
                                 if any(keyword in col for keyword in ["label", "outlier", "near_duplicate", "_score"])]
                
                if important_cols:
                    compact_df = issues_df[important_cols]
                    compact_df.to_csv(cl_out / "cleanlab_issues_compact.csv", index=False)
                
                print(f"Cleanlab report saved to: {cl_out}")
            except PermissionError:
                print("Warning: Could not save CSV files - they may be open in another program")
                print("Close Excel/CSV viewers and run again")
            
            # Print summary
            print(f"\nCleanlab report saved to: {cl_out}")
            print(f"Total samples analyzed: {len(issues_df)}")
            
            if "is_label_issue" in issues_df.columns:
                label_issues = issues_df["is_label_issue"].sum()
                print(f"Potential label issues: {label_issues} ({label_issues/len(issues_df)*100:.1f}%)")
            
            if "is_outlier_issue" in issues_df.columns:
                outlier_issues = issues_df["is_outlier_issue"].sum()
                print(f"Potential outliers: {outlier_issues} ({outlier_issues/len(issues_df)*100:.1f}%)")
            
            if "is_near_duplicate_issue" in issues_df.columns:
                dup_issues = issues_df["is_near_duplicate_issue"].sum()
                print(f"Potential near-duplicates: {dup_issues} ({dup_issues/len(issues_df)*100:.1f}%)")
                
        except Exception as e:
            print(f"Cleanlab analysis failed: {e}")
            print("Saving basic statistics instead...")
            
            # Fallback: basic statistics
            cl_out = Path(args.out) / "cleanlab"
            cl_out.mkdir(parents=True, exist_ok=True)
            
            try:
                basic_stats = pd.DataFrame({
                    "filename": names,
                    "class_name": [classes[label] for label in labels],
                    "predicted_class": [classes[np.argmax(prob)] for prob in probs],
                    "confidence": [np.max(prob) for prob in probs],
                    "correct_prediction": [labels[i] == np.argmax(probs[i]) for i in range(len(labels))]
                })
                
                # Close any open file handles and save with unique name if needed
                output_file = cl_out / "basic_predictions.csv"
                counter = 1
                while output_file.exists():
                    try:
                        basic_stats.to_csv(output_file, index=False)
                        break
                    except PermissionError:
                        output_file = cl_out / f"basic_predictions_{counter}.csv"
                        counter += 1
                        if counter > 10:  # Prevent infinite loop
                            print(f"Cannot save CSV - file may be open in another program")
                            break
                else:
                    basic_stats.to_csv(output_file, index=False)
                
                print(f"Basic prediction stats saved to: {output_file}")
            except Exception as csv_error:
                print(f"Failed to save basic stats: {csv_error}")
                print("You may need to close the CSV file if it's open in Excel/another program")
    else:
        print("\n== Skipping Cleanlab analysis (not properly installed) ==")

    print("\n✅ Data quality analysis complete!")
    print("📊 Check the CSV files to identify problematic samples for your training pipeline.")

if __name__ == "__main__":
    main()