#!/usr/bin/env python3
"""
Universal Data Quality Analyzer
A flexible template for analyzing image dataset quality using CleanVision + Cleanlab

Supports any ImageFolder-structured dataset for computer vision tasks.
Works offline after initial model download setup.

Author: Data Quality Analysis Template
Version: 1.0
Compatible with: PyTorch, CleanVision, Cleanlab
"""

import os
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Import with graceful fallbacks
try:
    from cleanlab import Datalab
    CLEANLAB_AVAILABLE = True
    CLEANLAB_VERSION = "2.x"
except ImportError:
    try:
        from cleanlab.datalab import Datalab
        CLEANLAB_AVAILABLE = True
        CLEANLAB_VERSION = "1.x"
    except ImportError:
        print("⚠️  Cleanlab not available. Install with: pip install 'cleanlab[datalab]'")
        CLEANLAB_AVAILABLE = False
        CLEANLAB_VERSION = None

try:
    from cleanvision import Imagelab
    CLEANVISION_AVAILABLE = True
except ImportError:
    print("⚠️  CleanVision not available. Install with: pip install cleanvision")
    CLEANVISION_AVAILABLE = False

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import timm
from tqdm import tqdm

# Configuration
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Supported image extensions
IMAGE_EXTENSIONS = {
    "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.tiff", "*.tif",
    "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.WEBP", "*.TIFF", "*.TIF"
}

# Normalization presets for different domains
NORMALIZATION_PRESETS = {
    "imagenet": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "description": "Standard ImageNet normalization for pretrained models"
    },
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "description": "CIFAR-10 dataset statistics"
    },
    "cifar100": {
        "mean": (0.5071, 0.4865, 0.4409),
        "std": (0.2673, 0.2564, 0.2762),
        "description": "CIFAR-100 dataset statistics"
    },
    "mnist": {
        "mean": (0.1307,),
        "std": (0.3081,),
        "description": "MNIST grayscale normalization"
    },
    "custom": {
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        "description": "Neutral normalization for custom datasets"
    }
}

# Model presets optimized for different hardware
MODEL_PRESETS = {
    "lightweight": {
        "model": "vit_tiny_patch16_224",
        "batch_size": 16,
        "description": "Fast analysis for low-end hardware"
    },
    "balanced": {
        "model": "vit_small_patch16_224", 
        "batch_size": 32,
        "description": "Good balance of speed and accuracy"
    },
    "accurate": {
        "model": "vit_base_patch16_224",
        "batch_size": 64,
        "description": "High accuracy for powerful hardware"
    },
    "cnn_fast": {
        "model": "resnet50",
        "batch_size": 32,
        "description": "CNN-based feature extraction"
    },
    "efficient": {
        "model": "efficientnet_b0",
        "batch_size": 48,
        "description": "Efficient architecture"
    }
}

class UniversalImageDataset(Dataset):
    """
    Universal dataset class that works with any ImageFolder structure.
    
    Expected structure:
    dataset_root/
    ├── class1/
    │   ├── image1.jpg
    │   ├── image2.png
    │   └── ...
    ├── class2/
    │   ├── image1.jpg
    │   └── ...
    └── classN/
        └── ...
    """
    
    def __init__(self, root, img_size=224, normalization="imagenet", max_samples=None):
        self.root = Path(root)
        self.img_size = img_size
        self.max_samples = max_samples
        
        # Validate root directory
        if not self.root.exists():
            raise FileNotFoundError(f"❌ Dataset directory not found: {root}")
        
        # Auto-detect classes from subdirectories
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        
        if len(self.classes) == 0:
            raise ValueError(f"❌ No class directories found in {root}")
        
        # Collect all image files
        self.samples = []
        self._collect_samples()
        
        if len(self.samples) == 0:
            raise ValueError(f"❌ No images found in {root}")
        
        # Apply sample limit if specified
        if max_samples and max_samples < len(self.samples):
            # Stratified sampling to maintain class balance
            samples_per_class = max_samples // len(self.classes)
            limited_samples = []
            
            for class_idx in range(len(self.classes)):
                class_samples = [s for s in self.samples if s[1] == class_idx]
                selected = class_samples[:samples_per_class]
                limited_samples.extend(selected)
            
            self.samples = limited_samples
        
        # Setup transforms
        self.transform = self._create_transforms(normalization)
        
        # Print dataset info
        self._print_dataset_info()
    
    def _collect_samples(self):
        """Collect all image files from class directories"""
        for class_idx, class_name in enumerate(self.classes):
            class_dir = self.root / class_name
            
            if not class_dir.is_dir():
                continue
                
            for ext in IMAGE_EXTENSIONS:
                for img_path in class_dir.glob(ext):
                    if img_path.is_file():
                        self.samples.append((img_path, class_idx))
    
    def _create_transforms(self, normalization):
        """Create image transforms based on normalization preset"""
        if normalization in NORMALIZATION_PRESETS:
            norm_config = NORMALIZATION_PRESETS[normalization]
            mean, std = norm_config["mean"], norm_config["std"]
        else:
            # Default to ImageNet
            mean, std = NORMALIZATION_PRESETS["imagenet"]["mean"], NORMALIZATION_PRESETS["imagenet"]["std"]
        
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    def _print_dataset_info(self):
        """Print dataset statistics"""
        print(f"📂 Dataset loaded: {self.root}")
        print(f"📊 Total samples: {len(self.samples)}")
        print(f"🏷️  Number of classes: {len(self.classes)}")
        print(f"📏 Image size: {self.img_size}x{self.img_size}")
        
        # Class distribution
        class_counts = {}
        for _, class_idx in self.samples:
            class_name = self.classes[class_idx]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print("📈 Class distribution:")
        for class_name, count in sorted(class_counts.items()):
            percentage = (count / len(self.samples)) * 100
            print(f"   {class_name}: {count} ({percentage:.1f}%)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            # Load and convert image
            image = Image.open(img_path).convert("RGB")
            tensor = self.transform(image)
            return tensor, label, str(img_path)
            
        except Exception as e:
            print(f"⚠️  Failed to load {img_path}: {e}")
            
            # Return black image as fallback
            black_image = Image.new('RGB', (self.img_size, self.img_size), (0, 0, 0))
            tensor = self.transform(black_image)
            return tensor, label, str(img_path)

def detect_hardware_capabilities():
    """Auto-detect hardware and suggest optimal settings"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        try:
            gpu_props = torch.cuda.get_device_properties(0)
            gpu_memory_gb = gpu_props.total_memory / 1e9
            gpu_name = gpu_props.name
            
            # Suggest settings based on GPU memory
            if gpu_memory_gb < 3:
                suggested_preset = "lightweight"
                max_batch_size = 8
            elif gpu_memory_gb < 6:
                suggested_preset = "lightweight" 
                max_batch_size = 16
            elif gpu_memory_gb < 10:
                suggested_preset = "balanced"
                max_batch_size = 32
            else:
                suggested_preset = "accurate"
                max_batch_size = 64
                
            print(f"🔧 GPU detected: {gpu_name} ({gpu_memory_gb:.1f}GB)")
            print(f"💡 Suggested preset: {suggested_preset} (max batch size: {max_batch_size})")
            
            return device, suggested_preset, max_batch_size
            
        except Exception as e:
            print(f"⚠️  GPU detection failed: {e}")
            return device, "lightweight", 16
    else:
        print("💻 Using CPU - recommended: lightweight preset with small batch size")
        return device, "lightweight", 4

def extract_features(backbone, dataset, batch_size, device, num_workers=0):
    """Extract features using pretrained backbone model"""
    
    # Create data loader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False
    )
    
    print(f"🔧 Loading backbone model: {backbone}")
    
    try:
        # Load pretrained model
        model = timm.create_model(backbone, pretrained=True, num_classes=0)
        model = model.to(device).eval()
        
        feature_dim = model.num_features
        print(f"✅ Model loaded successfully")
        print(f"📐 Feature dimension: {feature_dim}")
        
    except Exception as e:
        print(f"❌ Failed to load model {backbone}: {e}")
        print("💡 Make sure you've run the setup script to download models offline")
        raise
    
    # Extract features
    features_list = []
    labels_list = []
    paths_list = []
    
    print(f"🔍 Extracting features from {len(dataset)} samples...")
    
    with torch.no_grad():
        for batch_imgs, batch_labels, batch_paths in tqdm(dataloader, desc="Feature extraction"):
            # Move to device
            batch_imgs = batch_imgs.to(device)
            
            # Extract features
            batch_features = model(batch_imgs).cpu().numpy()
            
            # Collect results
            features_list.append(batch_features)
            labels_list.extend(batch_labels.numpy())
            paths_list.extend(batch_paths)
    
    # Concatenate all features
    features = np.concatenate(features_list, axis=0)
    labels = np.array(labels_list)
    
    print(f"✅ Feature extraction complete: {features.shape}")
    
    return features, labels, paths_list

def generate_cross_validation_predictions(features, labels, n_classes, cv_folds=3):
    """Generate out-of-fold predictions using cross-validation"""
    
    print(f"📈 Generating cross-validation predictions ({cv_folds} folds)...")
    
    # Initialize prediction array
    predictions = np.zeros((len(labels), n_classes))
    
    # Stratified K-Fold cross-validation
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
        print(f"   Training fold {fold + 1}/{cv_folds}...")
        
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                max_iter=2000,
                solver='saga',
                random_state=SEED,
                n_jobs=-1
            ))
        ])
        
        # Train and predict
        pipeline.fit(features[train_idx], labels[train_idx])
        fold_predictions = pipeline.predict_proba(features[val_idx])
        
        # Store predictions
        predictions[val_idx] = fold_predictions
    
    # Calculate accuracy
    predicted_labels = np.argmax(predictions, axis=1)
    accuracy = np.mean(predicted_labels == labels)
    
    print(f"✅ Cross-validation complete")
    print(f"📊 CV Accuracy: {accuracy:.3f}")
    
    return predictions

def run_cleanvision_analysis(data_dir, output_dir, force_overwrite=True):
    """Run CleanVision image quality analysis"""
    
    if not CLEANVISION_AVAILABLE:
        print("⏭️  Skipping CleanVision analysis (not installed)")
        return None
    
    print("🔍 Running CleanVision image quality analysis...")
    
    cv_output = Path(output_dir) / "cleanvision"
    cv_output.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize Imagelab
        imagelab = Imagelab(data_path=str(data_dir))
        
        # Find issues
        imagelab.find_issues(verbose=False)
        
        # Save results
        try:
            imagelab.save(str(cv_output), force=force_overwrite)
        except TypeError:
            # Fallback for older CleanVision versions
            imagelab.report(str(cv_output))
        
        print(f"✅ CleanVision analysis complete")
        print(f"📁 Report saved to: {cv_output}")
        
        return cv_output
        
    except Exception as e:
        print(f"❌ CleanVision analysis failed: {e}")
        return None

def run_cleanlab_analysis(features, labels, predictions, file_paths, class_names, output_dir):
    """Run Cleanlab data quality analysis"""
    
    if not CLEANLAB_AVAILABLE:
        print("⏭️  Skipping Cleanlab analysis (not installed)")
        return None, None
    
    print("🔍 Running Cleanlab data quality analysis...")
    
    cl_output = Path(output_dir) / "cleanlab"
    cl_output.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create dataset for Cleanlab
        dataset_dict = {
            "features": features,
            "labels": labels
        }
        
        # Initialize Datalab
        if CLEANLAB_VERSION == "2.x":
            lab = Datalab(data=dataset_dict, label_name="labels")
        else:
            lab = Datalab(data=dataset_dict, label_name="labels")
        
        # Find all types of issues
        lab.find_issues(pred_probs=predictions, verbose=False)
        
        # Get issue report
        issues_df = lab.get_issues()
        
        # Add metadata columns
        issues_df["file_path"] = file_paths
        issues_df["class_name"] = [class_names[label] for label in labels]
        issues_df["predicted_class"] = [class_names[np.argmax(pred)] for pred in predictions]
        issues_df["prediction_confidence"] = [np.max(pred) for pred in predictions]
        
        # Save comprehensive report
        full_report_path = cl_output / "cleanlab_full_report.csv"
        issues_df.to_csv(full_report_path, index=False)
        
        # Create and save summary
        summary = create_cleanlab_summary(issues_df)
        summary_path = cl_output / "cleanlab_summary.json"
        
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        # Save problematic samples for easy review
        problematic_samples = identify_problematic_samples(issues_df)
        if len(problematic_samples) > 0:
            problematic_path = cl_output / "problematic_samples.csv"
            problematic_samples.to_csv(problematic_path, index=False)
            print(f"⚠️  Found {len(problematic_samples)} problematic samples")
        
        print(f"✅ Cleanlab analysis complete")
        print(f"📁 Reports saved to: {cl_output}")
        
        return cl_output, summary
        
    except Exception as e:
        print(f"❌ Cleanlab analysis failed: {e}")
        print("💾 Saving basic quality metrics instead...")
        
        # Fallback: save basic predictions and confidence scores
        basic_metrics = create_basic_quality_metrics(
            labels, predictions, file_paths, class_names
        )
        
        basic_path = cl_output / "basic_quality_metrics.csv"
        basic_metrics.to_csv(basic_path, index=False)
        
        print(f"💾 Basic metrics saved to: {basic_path}")
        
        return cl_output, None

def create_cleanlab_summary(issues_df):
    """Create summary statistics from Cleanlab results"""
    
    summary = {
        "total_samples": len(issues_df),
        "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "issues_found": {}
    }
    
    # Count different types of issues
    issue_columns = [col for col in issues_df.columns if col.startswith("is_") and col != "is_issue"]
    
    for col in issue_columns:
        if col in issues_df.columns:
            count = int(issues_df[col].sum())
            percentage = round((count / len(issues_df)) * 100, 2)
            
            summary["issues_found"][col] = {
                "count": count,
                "percentage": percentage,
                "description": get_issue_description(col)
            }
    
    # Overall statistics
    if "prediction_confidence" in issues_df.columns:
        summary["confidence_stats"] = {
            "mean": float(issues_df["prediction_confidence"].mean()),
            "median": float(issues_df["prediction_confidence"].median()),
            "std": float(issues_df["prediction_confidence"].std()),
            "min": float(issues_df["prediction_confidence"].min()),
            "max": float(issues_df["prediction_confidence"].max())
        }
    
    return summary

def identify_problematic_samples(issues_df):
    """Identify the most problematic samples for manual review"""
    
    # Define problematic criteria
    problematic = issues_df[
        (issues_df.get("is_label_issue", False)) |
        (issues_df.get("is_outlier_issue", False)) |
        (issues_df.get("prediction_confidence", 1.0) < 0.3)
    ].copy()
    
    if len(problematic) > 0:
        # Sort by confidence (lowest first) and other issue indicators
        sort_columns = []
        if "prediction_confidence" in problematic.columns:
            sort_columns.append("prediction_confidence")
        if "is_label_issue" in problematic.columns:
            sort_columns.append("is_label_issue")
        
        if sort_columns:
            problematic = problematic.sort_values(sort_columns)
    
    return problematic

def create_basic_quality_metrics(labels, predictions, file_paths, class_names):
    """Create basic quality metrics when Cleanlab fails"""
    
    predicted_labels = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)
    
    return pd.DataFrame({
        "file_path": file_paths,
        "true_class": [class_names[label] for label in labels],
        "predicted_class": [class_names[pred] for pred in predicted_labels],
        "prediction_confidence": confidences,
        "correct_prediction": (labels == predicted_labels),
        "low_confidence": (confidences < 0.5),
        "very_low_confidence": (confidences < 0.3)
    })

def get_issue_description(issue_type):
    """Get human-readable description for issue types"""
    
    descriptions = {
        "is_label_issue": "Samples with potentially incorrect labels",
        "is_outlier_issue": "Samples that are significantly different from others",
        "is_near_duplicate_issue": "Samples that are very similar to others",
        "is_data_issue": "General data quality issues"
    }
    
    return descriptions.get(issue_type, "Quality issue detected")

def save_analysis_metadata(output_dir, config, dataset_info, results):
    """Save comprehensive metadata about the analysis"""
    
    metadata = {
        "analysis_info": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "template_version": "1.0",
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
            "device_used": config["device"]
        },
        "dataset_info": dataset_info,
        "analysis_config": {
            "backbone_model": config["backbone"],
            "batch_size": config["batch_size"],
            "cv_folds": config["cv_folds"],
            "normalization": config["normalization"],
            "max_samples": config.get("max_samples")
        },
        "results_summary": results,
        "output_files": {
            "cleanvision_report": "cleanvision/",
            "cleanlab_report": "cleanlab/",
            "metadata": "analysis_metadata.json"
        }
    }
    
    metadata_path = Path(output_dir) / "analysis_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📄 Analysis metadata saved to: {metadata_path}")

def main():
    """Main analysis pipeline"""
    
    parser = argparse.ArgumentParser(
        description="Universal Data Quality Analyzer for Image Datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis of a CIFAR-10 dataset
  python universal_data_quality_analyzer.py --data data/cifar10/train --output reports/cifar10
  
  # Medical images with ResNet backbone
  python universal_data_quality_analyzer.py --data data/medical/train --backbone resnet50 --preset medical
  
  # Large dataset with sample limiting
  python universal_data_quality_analyzer.py --data data/large_dataset/train --max-samples 10000
  
  # Quick image quality check only
  python universal_data_quality_analyzer.py --data data/test --skip-cleanlab --preset lightweight
        """
    )
    
    # Required arguments
    parser.add_argument("--data", type=str, required=True,
                       help="Path to dataset directory (ImageFolder structure)")
    
    # Output configuration
    parser.add_argument("--output", type=str, default="data_quality_reports",
                       help="Output directory for analysis reports")
    
    # Model and hardware configuration
    parser.add_argument("--backbone", type=str, default="auto",
                       choices=["auto", "vit_tiny_patch16_224", "vit_small_patch16_224", 
                               "vit_base_patch16_224", "resnet50", "efficientnet_b0"],
                       help="Feature extraction backbone model")
    
    parser.add_argument("--preset", type=str, default="auto",
                       choices=["auto", "lightweight", "balanced", "accurate", "cnn_fast", "efficient"],
                       help="Hardware-optimized preset configuration")
    
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size for feature extraction (auto-detected if not specified)")
    
    # Data configuration
    parser.add_argument("--normalization", type=str, default="imagenet",
                       choices=list(NORMALIZATION_PRESETS.keys()),
                       help="Image normalization preset")
    
    parser.add_argument("--img-size", type=int, default=224,
                       help="Input image size for model")
    
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Limit analysis to N samples (stratified sampling)")
    
    # Analysis configuration
    parser.add_argument("--cv-folds", type=int, default=3,
                       help="Number of cross-validation folds")
    
    parser.add_argument("--num-workers", type=int, default=0,
                       help="Number of data loading workers (0 for Windows compatibility)")
    
    # Analysis options
    parser.add_argument("--skip-cleanvision", action="store_true",
                       help="Skip CleanVision image quality analysis")
    
    parser.add_argument("--skip-cleanlab", action="store_true",
                       help="Skip Cleanlab label quality analysis")
    
    parser.add_argument("--force-overwrite", action="store_true",
                       help="Overwrite existing reports")
    
    # Verbose output
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 80)
    print("🔍 UNIVERSAL DATA QUALITY ANALYZER")
    print("=" * 80)
    print(f"📂 Dataset: {args.data}")
    print(f"💾 Output: {args.output}")
    
    # Auto-detect hardware capabilities
    device, suggested_preset, suggested_batch_size = detect_hardware_capabilities()
    
    # Configure analysis settings
    if args.preset == "auto":
        preset_config = MODEL_PRESETS[suggested_preset]
        backbone = preset_config["model"] if args.backbone == "auto" else args.backbone
        batch_size = args.batch_size or suggested_batch_size
    else:
        preset_config = MODEL_PRESETS[args.preset]
        backbone = preset_config["model"] if args.backbone == "auto" else args.backbone
        batch_size = args.batch_size or preset_config["batch_size"]
    
    print(f"🔧 Device: {device}")
    print(f"🧠 Backbone: {backbone}")
    print(f"📦 Batch size: {batch_size}")
    print(f"🎨 Normalization: {args.normalization}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("\n" + "=" * 50)
    print("📊 LOADING DATASET")
    print("=" * 50)
    
    try:
        dataset = UniversalImageDataset(
            root=args.data,
            img_size=args.img_size,
            normalization=args.normalization,
            max_samples=args.max_samples
        )
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return 1
    
    # Store dataset info for metadata
    dataset_info = {
        "path": str(args.data),
        "total_samples": len(dataset),
        "num_classes": len(dataset.classes),
        "class_names": dataset.classes,
        "image_size": args.img_size
    }
    
    # 1. CleanVision Analysis (Image Quality)
    if not args.skip_cleanvision:
        print("\n" + "=" * 50)
        print("🖼️  CLEANVISION ANALYSIS")
        print("=" * 50)
        
        cv_results = run_cleanvision_analysis(
            data_dir=args.data,
            output_dir=args.output,
            force_overwrite=args.force_overwrite
        )
    else:
        print("\n⏭️  Skipping CleanVision analysis")
        cv_results = None
    
    # 2. Feature Extraction
    print("\n" + "=" * 50)
    print("🔧 FEATURE EXTRACTION")
    print("=" * 50)
    
    try:
        features, labels, file_paths = extract_features(
            backbone=backbone,
            dataset=dataset,
            batch_size=batch_size,
            device=device,
            num_workers=args.num_workers
        )
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        return 1
    
    # 3. Cross-Validation Predictions
    print("\n" + "=" * 50)
    print("📈 CROSS-VALIDATION")
    print("=" * 50)
    
    try:
        predictions = generate_cross_validation_predictions(
            features=features,
            labels=labels,
            n_classes=len(dataset.classes),
            cv_folds=args.cv_folds
        )
    except Exception as e:
        print(f"❌ Cross-validation failed: {e}")
        return 1
    
    # 4. Cleanlab Analysis (Label Quality)
    if not args.skip_cleanlab:
        print("\n" + "=" * 50)
        print("🏷️  CLEANLAB ANALYSIS")
        print("=" * 50)
        
        cl_results, cl_summary = run_cleanlab_analysis(
            features=features,
            labels=labels,
            predictions=predictions,
            file_paths=file_paths,
            class_names=dataset.classes,
            output_dir=args.output
        )
    else:
        print("\n⏭️  Skipping Cleanlab analysis")
        cl_results, cl_summary = None, None
    
    # 5. Save Analysis Metadata
    print("\n" + "=" * 50)
    print("💾 SAVING METADATA")
    print("=" * 50)
    
    analysis_config = {
        "backbone": backbone,
        "batch_size": batch_size,
        "cv_folds": args.cv_folds,
        "normalization": args.normalization,
        "device": device,
        "max_samples": args.max_samples
    }
    
    results_summary = {
        "cv_accuracy": float(np.mean(np.argmax(predictions, axis=1) == labels)),
        "cleanvision_completed": cv_results is not None,
        "cleanlab_completed": cl_results is not None,
        "cleanlab_summary": cl_summary
    }
    
    save_analysis_metadata(
        output_dir=args.output,
        config=analysis_config,
        dataset_info=dataset_info,
        results=results_summary
    )
    
    # Print final summary
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"📁 All reports saved to: {output_dir}")
    print(f"📊 Total samples analyzed: {len(dataset)}")
    print(f"🎯 Cross-validation accuracy: {results_summary['cv_accuracy']:.3f}")
    
    if cl_summary and "issues_found" in cl_summary:
        print("\n🔍 Issues detected:")
        for issue_type, stats in cl_summary["issues_found"].items():
            if stats["count"] > 0:
                print(f"   {issue_type}: {stats['count']} ({stats['percentage']:.1f}%)")
    
    print(f"\n💡 Next steps:")
    print(f"   1. Review reports in {output_dir}")
    print(f"   2. Check problematic_samples.csv for manual inspection")
    print(f"   3. Clean dataset based on findings")
    print(f"   4. Re-run analysis to verify improvements")
    
    return 0

if __name__ == "__main__":
    exit(main())