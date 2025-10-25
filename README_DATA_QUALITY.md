# 🔍 Universal Data Quality Analyzer

A comprehensive, flexible template for analyzing image dataset quality using **CleanVision** and **Cleanlab**. Works with any ImageFolder-structured dataset and supports both online and offline analysis.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 **What This Template Does**

### **Image Quality Analysis (CleanVision)**
- ✅ **Blurry images** detection
- ✅ **Dark/bright images** identification  
- ✅ **Odd aspect ratios** finding
- ✅ **Exact duplicates** detection
- ✅ **Low information content** flagging

### **Label Quality Analysis (Cleanlab)**
- ✅ **Mislabeled samples** detection
- ✅ **Outliers/anomalies** identification
- ✅ **Near-duplicate images** finding
- ✅ **Prediction confidence** scoring
- ✅ **Cross-validation** accuracy assessment

---

## 🚀 **Quick Start Guide**

### **Step 1: One-Time Setup (Requires Internet)**

```bash
# Clone or download the template
git clone <your-repo> data-quality-analyzer
cd data-quality-analyzer

# Create virtual environment
python -m venv analyzer_env
source analyzer_env/bin/activate  # Linux/Mac
# analyzer_env\Scripts\activate   # Windows

# Run offline setup (downloads models and dependencies)
python setup_offline.py
```

### **Step 2: Prepare Your Dataset**

Organize your dataset in **ImageFolder structure**:

```
your_dataset/
├── train/                    # or any name (validation/, test/, etc.)
│   ├── class1/              # Class directories
│   │   ├── image1.jpg       # Image files
│   │   ├── image2.png
│   │   └── ...
│   ├── class2/
│   │   ├── image1.jpg
│   │   └── ...
│   └── classN/
│       └── ...
```

**Supported formats**: JPG, JPEG, PNG, BMP, WebP, TIFF
**Any resolution** (automatically resized to 224×224)

### **Step 3: Run Analysis (Works Offline)**

```bash
# Basic analysis
python universal_data_quality_analyzer.py --data your_dataset/train

# With custom settings
python universal_data_quality_analyzer.py \
  --data your_dataset/train \
  --output my_reports \
  --preset balanced \
  --normalization imagenet
```

### **Step 4: Review Results**

Check the generated reports in `data_quality_reports/`:

```
data_quality_reports/
├── cleanvision/
│   ├── issue_summary.csv      # Image quality overview
│   └── issues.csv             # Per-image quality scores
├── cleanlab/
│   ├── cleanlab_full_report.csv    # Comprehensive analysis
│   ├── problematic_samples.csv     # Most problematic files
│   └── cleanlab_summary.json       # Quick statistics
└── analysis_metadata.json          # Analysis configuration
```

---

## 📋 **Usage Examples**

### **Example 1: CIFAR-10 Dataset**

```bash
# Dataset structure:
# data/cifar10/train/airplane/*.png
# data/cifar10/train/automobile/*.png
# ...

python universal_data_quality_analyzer.py \
  --data data/cifar10/train \
  --output cifar10_reports \
  --preset lightweight \
  --normalization cifar10 \
  --batch-size 32
```

### **Example 2: Medical Images**

```bash
# Dataset structure:
# data/medical/train/normal/*.jpg
# data/medical/train/abnormal/*.jpg

python universal_data_quality_analyzer.py \
  --data data/medical/train \
  --output medical_reports \
  --backbone resnet50 \
  --preset balanced \
  --normalization imagenet
```

### **Example 3: Large Custom Dataset**

```bash
# Limit analysis to 10K samples for speed
python universal_data_quality_analyzer.py \
  --data data/large_dataset/train \
  --output large_dataset_reports \
  --preset accurate \
  --max-samples 10000 \
  --cv-folds 5
```

### **Example 4: Quick Image Quality Check**

```bash
# Skip heavy label analysis, just check image quality
python universal_data_quality_analyzer.py \
  --data data/unknown_quality \
  --output quick_check \
  --skip-cleanlab \
  --preset lightweight
```

---

## ⚙️ **Configuration Options**

### **Hardware Presets**

| Preset | Model | Batch Size | Best For |
|--------|-------|------------|----------|
| `lightweight` | ViT-Tiny | 16 | Low-end GPUs (2-4GB), fast analysis |
| `balanced` | ViT-Small | 32 | Mid-range GPUs (6-8GB), recommended |
| `accurate` | ViT-Base | 64 | High-end GPUs (12GB+), maximum accuracy |
| `cnn_fast` | ResNet-50 | 32 | CNN comparison, stable features |
| `efficient` | EfficientNet-B0 | 48 | Mobile deployment focus |

### **Normalization Presets**

| Preset | Use For |
|--------|---------|
| `imagenet` | General datasets, pretrained models (default) |
| `cifar10` | CIFAR-10 dataset |
| `cifar100` | CIFAR-100 dataset |
| `mnist` | Grayscale/MNIST-like datasets |
| `custom` | Custom datasets, neutral normalization |

### **Command Line Arguments**

```bash
# Required
--data PATH                    # Path to dataset directory

# Model & Hardware
--preset {auto,lightweight,balanced,accurate,cnn_fast,efficient}
--backbone {auto,vit_tiny_patch16_224,vit_small_patch16_224,vit_base_patch16_224,resnet50,efficientnet_b0}
--batch-size INT               # Batch size (auto-detected if not specified)

# Data Processing
--normalization {imagenet,cifar10,cifar100,mnist,custom}
--img-size INT                 # Input image size (default: 224)
--max-samples INT              # Limit analysis to N samples

# Analysis Options
--cv-folds INT                 # Cross-validation folds (default: 3)
--skip-cleanvision            # Skip image quality analysis
--skip-cleanlab               # Skip label quality analysis

# Output
--output PATH                  # Output directory (default: data_quality_reports)
--force-overwrite             # Overwrite existing reports
--verbose                     # Enable detailed output
```

---

## 📊 **Understanding the Results**

### **CleanVision Results**

**`cleanvision/issue_summary.csv`** - Overview of image quality issues:
- Total images processed
- Number of images with each type of issue
- Percentage breakdown

**`cleanvision/issues.csv`** - Per-image detailed scores:
- `is_blurry_issue`: Blurry/out-of-focus images
- `is_dark_issue`: Underexposed images  
- `is_light_issue`: Overexposed images
- `is_odd_aspect_ratio_issue`: Unusual image dimensions
- `is_exact_duplicates_issue`: Identical images

### **Cleanlab Results**

**`cleanlab/cleanlab_full_report.csv`** - Comprehensive analysis:
- `file_path`: Path to each image
- `class_name`: True class label
- `predicted_class`: Model's prediction
- `prediction_confidence`: Confidence score (0-1)
- `is_label_issue`: Potential labeling errors
- `is_outlier_issue`: Unusual/anomalous samples
- `is_near_duplicate_issue`: Very similar images

**`cleanlab/problematic_samples.csv`** - Most problematic files:
- Low confidence predictions
- Potential mislabeled samples
- Recommended for manual review

**`cleanlab/cleanlab_summary.json`** - Quick statistics:
```json
{
  "total_samples": 50000,
  "issues_found": {
    "is_label_issue": {"count": 127, "percentage": 0.25},
    "is_outlier_issue": {"count": 89, "percentage": 0.18}
  },
  "confidence_stats": {
    "mean": 0.834,
    "median": 0.892,
    "std": 0.156
  }
}
```

---

## 🎯 **Real-World Use Cases**

### **1. Dataset Cleaning Pipeline**

```bash
# Step 1: Initial quality assessment
python universal_data_quality_analyzer.py \
  --data raw_dataset/train \
  --output initial_assessment

# Step 2: Review problematic_samples.csv and remove bad samples

# Step 3: Re-analyze cleaned dataset
python universal_data_quality_analyzer.py \
  --data cleaned_dataset/train \
  --output final_assessment
```

### **2. Model Training Data Selection**

```python
import pandas as pd

# Load analysis results
results = pd.read_csv("data_quality_reports/cleanlab/cleanlab_full_report.csv")

# Filter high-quality samples for training
high_quality = results[
    (results['prediction_confidence'] > 0.7) &
    (~results['is_label_issue']) &
    (~results['is_outlier_issue'])
]

# Use high_quality['file_path'] for training
```

### **3. Comparative Dataset Analysis**

```bash
# Analyze multiple datasets with same settings
for dataset in dataset_A dataset_B dataset_C; do
  python universal_data_quality_analyzer.py \
    --data $dataset/train \
    --output ${dataset}_analysis \
    --preset balanced
done

# Compare quality across datasets
```

### **4. Hardware-Optimized Analysis**

```bash
# For GTX 1050 (2GB VRAM)
python universal_data_quality_analyzer.py \
  --data my_dataset/train \
  --preset lightweight \
  --batch-size 8

# For RTX 3080 (10GB VRAM)  
python universal_data_quality_analyzer.py \
  --data my_dataset/train \
  --preset accurate \
  --batch-size 64
```

---

## 🔧 **Advanced Configuration**

### **Custom Analysis Script**

For repeated analysis with specific settings, create a custom script:

```bash
#!/bin/bash
# analyze_medical_data.sh

python universal_data_quality_analyzer.py \
  --data "$1" \
  --output "${1}_medical_analysis" \
  --backbone resnet50 \
  --normalization imagenet \
  --preset balanced \
  --cv-folds 5 \
  --force-overwrite

echo "Medical dataset analysis complete for $1"
```

Usage:
```bash
chmod +x analyze_medical_data.sh
./analyze_medical_data.sh data/chest_xray/train
./analyze_medical_data.sh data/skin_lesion/train
```

### **Batch Processing Multiple Datasets**

```python
# batch_analyze.py
import subprocess
import sys
from pathlib import Path

datasets = [
    "data/dataset1/train",
    "data/dataset2/train", 
    "data/dataset3/train"
]

for dataset_path in datasets:
    dataset_name = Path(dataset_path).parent.name
    output_dir = f"batch_analysis/{dataset_name}"
    
    cmd = [
        sys.executable,
        "universal_data_quality_analyzer.py",
        "--data", dataset_path,
        "--output", output_dir,
        "--preset", "balanced"
    ]
    
    print(f"Analyzing {dataset_name}...")
    subprocess.run(cmd)
    print(f"✅ {dataset_name} complete")
```

### **Integration with Existing Workflows**

```python
# integrate_with_training.py
import pandas as pd
from pathlib import Path

def filter_high_quality_samples(analysis_dir, confidence_threshold=0.8):
    """Filter dataset based on quality analysis results"""
    
    results_path = Path(analysis_dir) / "cleanlab" / "cleanlab_full_report.csv"
    
    if not results_path.exists():
        print("Analysis results not found. Run quality analysis first.")
        return None
    
    results = pd.read_csv(results_path)
    
    # Define quality criteria
    high_quality = results[
        (results['prediction_confidence'] >= confidence_threshold) &
        (~results.get('is_label_issue', False)) &
        (~results.get('is_outlier_issue', False))
    ]
    
    print(f"Original samples: {len(results)}")
    print(f"High quality samples: {len(high_quality)} ({len(high_quality)/len(results)*100:.1f}%)")
    
    return high_quality['file_path'].tolist()

# Usage in training script
quality_file_paths = filter_high_quality_samples("data_quality_reports")
# Use quality_file_paths to create filtered dataset
```

---

## 🛠️ **Troubleshooting**

### **Common Issues & Solutions**

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **GPU Memory Error** | CUDA out of memory | Use `--preset lightweight` or `--batch-size 8` |
| **Model Download Fails** | Connection/download errors | Re-run `setup_offline.py`, check internet |
| **CleanVision Crashes** | CleanVision import/runtime errors | Add `--skip-cleanvision` flag |
| **CleanLab Fails** | ImportError or runtime errors | Check `basic_quality_metrics.csv` instead |
| **No Images Found** | Empty dataset error | Check dataset structure: `data/train/class1/*.jpg` |
| **Permission Denied** | File access errors | Close CSV files, check folder permissions |
| **Slow Analysis** | Very slow processing | Reduce `--max-samples` or use `--preset lightweight` |

### **Hardware-Specific Optimizations**

#### **Low-End Hardware (2-4GB GPU/8GB RAM)**
```bash
python universal_data_quality_analyzer.py \
  --preset lightweight \
  --batch-size 8 \
  --max-samples 5000 \
  --cv-folds 2
```

#### **High-End Hardware (12GB+ GPU/32GB+ RAM)**
```bash
python universal_data_quality_analyzer.py \
  --preset accurate \
  --batch-size 128 \
  --cv-folds 5
```

#### **CPU-Only Systems**
```bash
python universal_data_quality_analyzer.py \
  --preset lightweight \
  --batch-size 4 \
  --max-samples 1000
```

### **Windows-Specific Notes**

- **Multiprocessing**: Always use `--num-workers 0` (default)
- **Path separators**: Use forward slashes `/` or escape backslashes `\\`
- **Long paths**: Enable long path support in Windows 10/11
- **Antivirus**: Exclude the analyzer directory from real-time scanning

---

## 📖 **Dataset Setup Guide**

### **Supported Dataset Structures**

#### **Standard ImageFolder**
```
dataset/
├── train/
│   ├── cats/
│   │   ├── cat1.jpg
│   │   └── cat2.jpg
│   └── dogs/
│       ├── dog1.jpg
│       └── dog2.jpg
```

#### **Multi-Split Dataset**
```
dataset/
├── train/
│   ├── class1/
│   └── class2/
├── validation/
│   ├── class1/
│   └── class2/
└── test/
    ├── class1/
    └── class2/
```

#### **Nested Class Structure**
```
dataset/
├── train/
│   ├── animals/
│   │   ├── cats/
│   │   └── dogs/
│   └── vehicles/
│       ├── cars/
│       └── trucks/
```

### **Dataset Conversion Scripts**

#### **From CIFAR-10 Pickle to ImageFolder**
```python
# convert_cifar10.py
import pickle
import numpy as np
from PIL import Image
from pathlib import Path

def convert_cifar10(cifar_dir, output_dir):
    # Load CIFAR-10 data
    meta = pickle.load(open(f"{cifar_dir}/batches.meta", 'rb'))
    classes = [b.decode('utf-8') for b in meta['label_names']]
    
    # Create output structure
    for split in ['train', 'test']:
        for class_name in classes:
            Path(f"{output_dir}/{split}/{class_name}").mkdir(parents=True, exist_ok=True)
    
    # Convert training data
    for i in range(1, 6):
        batch = pickle.load(open(f"{cifar_dir}/data_batch_{i}", 'rb'))
        images = batch['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = batch['labels']
        
        for idx, (img, label) in enumerate(zip(images, labels)):
            class_name = classes[label]
            img_path = f"{output_dir}/train/{class_name}/batch{i}_{idx:05d}.png"
            Image.fromarray(img).save(img_path)

# Usage
convert_cifar10("data/cifar-10-batches-py", "data/cifar10_imagefolder")
```

#### **From CSV Annotations to ImageFolder**
```python
# convert_csv_dataset.py
import pandas as pd
import shutil
from pathlib import Path

def convert_csv_to_imagefolder(csv_path, images_dir, output_dir):
    df = pd.read_csv(csv_path)
    
    # Create class directories
    classes = df['class'].unique()
    for class_name in classes:
        Path(f"{output_dir}/train/{class_name}").mkdir(parents=True, exist_ok=True)
    
    # Copy images to class directories
    for _, row in df.iterrows():
        src_path = Path(images_dir) / row['filename']
        dst_path = Path(output_dir) / "train" / row['class'] / row['filename']
        shutil.copy2(src_path, dst_path)

# Usage
# CSV should have columns: filename, class
convert_csv_to_imagefolder("annotations.csv", "images/", "organized_dataset/")
```

---

## 🔬 **Technical Details**

### **Feature Extraction Process**

1. **Image Preprocessing**:
   - Resize to 224×224 pixels
   - Convert to RGB (handles grayscale)
   - Normalize using dataset-specific statistics
   - Tensor conversion

2. **Model Inference**:
   - Load pretrained vision transformer or CNN
   - Extract features from penultimate layer
   - Typical dimensions: 192-768 features

3. **Cross-Validation**:
   - Stratified K-fold to maintain class balance
   - Train logistic regression on features
   - Generate out-of-fold predictions

### **Quality Assessment Methods**

#### **CleanVision (Image Quality)**
- **Blur detection**: Laplacian variance analysis
- **Brightness**: Histogram-based luminance analysis  
- **Aspect ratio**: Geometric analysis
- **Duplicates**: Perceptual hashing comparison

#### **Cleanlab (Label Quality)**
- **Label errors**: Confident Learning algorithm
- **Outliers**: Feature space density analysis
- **Near-duplicates**: Cosine similarity in feature space
- **Confidence scoring**: Prediction probability analysis

### **Performance Characteristics**

| Dataset Size | Hardware | Expected Time | Memory Usage |
|--------------|----------|---------------|--------------|
| 1K images | GTX 1050 | 2-5 minutes | 1-2GB |
| 10K images | GTX 1060 | 15-30 minutes | 2-4GB |
| 50K images | GTX 3070 | 30-60 minutes | 4-8GB |
| 100K images | RTX 3080 | 1-2 hours | 8-12GB |

---

## 📚 **References & Credits**

### **Core Libraries**
- **[CleanVision](https://github.com/cleanlab/cleanvision)**: Computer vision data quality analysis
- **[Cleanlab](https://github.com/cleanlab/cleanlab)**: Label quality and data-centric AI
- **[timm](https://github.com/rwightman/pytorch-image-models)**: PyTorch image models
- **[PyTorch](https://pytorch.org/)**: Deep learning framework

### **Supported Models**
- **Vision Transformers**: ViT-Tiny, ViT-Small, ViT-Base
- **CNNs**: ResNet-50, EfficientNet-B0, ConvNeXt-Tiny
- **All models**: Pretrained on ImageNet-1k/21k

### **Research Papers**
- Confident Learning: [Northcutt et al., 2021](https://jair.org/index.php/jair/article/view/12125)
- Vision Transformer: [Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929)
- EfficientNet: [Tan & Le, 2019](https://arxiv.org/abs/1905.11946)

---

## 📄 **License**

This template is released under the MIT License. See [LICENSE](LICENSE) for details.

## 🤝 **Contributing**

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📞 **Support**

- 📖 **Documentation**: This README and inline code comments
- 🐛 **Issues**: GitHub Issues for bug reports
- 💡 **Feature Requests**: GitHub Discussions
- 📧 **Contact**: [Your contact information]

---

**🎉 Happy analyzing! This template helps you build robust, high-quality datasets for better machine learning models.**