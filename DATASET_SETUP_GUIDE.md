# 📁 Dataset Setup Guide

Comprehensive guide for preparing various image datasets for quality analysis with the Universal Data Quality Analyzer.

## 📋 **Table of Contents**

1. [Standard Dataset Structure](#standard-dataset-structure)
2. [Common Dataset Conversions](#common-dataset-conversions)
3. [Dataset-Specific Setup](#dataset-specific-setup)
4. [Validation & Testing](#validation--testing)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)

---

## 🏗️ **Standard Dataset Structure**

The analyzer expects the **ImageFolder** structure used by PyTorch:

### **Basic Structure**
```
your_dataset/
├── train/                    # Main directory (can be any name)
│   ├── class1/              # Class directories
│   │   ├── image1.jpg       # Image files
│   │   ├── image2.png
│   │   ├── image3.jpeg
│   │   └── ...
│   ├── class2/
│   │   ├── image1.jpg
│   │   ├── image2.bmp
│   │   └── ...
│   ├── class3/
│   │   └── ...
│   └── classN/
│       └── ...
```

### **Supported File Formats**
- ✅ **JPEG/JPG** - Most common format
- ✅ **PNG** - Lossless compression, transparency support
- ✅ **BMP** - Uncompressed bitmap
- ✅ **WebP** - Modern web format
- ✅ **TIFF/TIF** - High-quality format
- ✅ **Mixed formats** - Can mix different formats in same dataset

### **File Naming Conventions**
- 🎯 **Any names work** - No specific naming requirements
- 🎯 **Special characters** - Avoid spaces and special characters in filenames
- 🎯 **Extensions** - Case-insensitive (.jpg, .JPG, .Jpg all work)
- 🎯 **Subdirectories** - Only immediate subdirectories are considered classes

---

## 🔄 **Common Dataset Conversions**

### **1. CIFAR-10/100 from Pickle to ImageFolder**

CIFAR datasets come as Python pickle files. Convert them to ImageFolder:

```python
#!/usr/bin/env python3
"""Convert CIFAR-10/100 from pickle to ImageFolder structure"""

import pickle
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def convert_cifar10(cifar_dir, output_dir):
    """Convert CIFAR-10 from pickle to ImageFolder"""
    
    cifar_path = Path(cifar_dir)
    output_path = Path(output_dir)
    
    # Load class names
    meta = pickle.load(open(cifar_path / "batches.meta", 'rb'), encoding='bytes')
    class_names = [name.decode('utf-8') for name in meta[b'label_names']]
    
    print(f"Converting CIFAR-10: {len(class_names)} classes")
    print(f"Classes: {class_names}")
    
    # Create directory structure
    for split in ['train', 'test']:
        for class_name in class_names:
            (output_path / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Convert training data (5 batches)
    print("Converting training data...")
    train_images, train_labels = [], []
    
    for i in range(1, 6):
        batch_file = cifar_path / f"data_batch_{i}"
        batch = pickle.load(open(batch_file, 'rb'), encoding='bytes')
        train_images.append(batch[b'data'])
        train_labels.extend(batch[b'labels'])
    
    train_images = np.concatenate(train_images, axis=0)
    train_images = train_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    for idx, (img, label) in enumerate(tqdm(zip(train_images, train_labels), desc="Saving train")):
        class_name = class_names[label]
        img_pil = Image.fromarray(img)
        img_path = output_path / "train" / class_name / f"train_{idx:05d}.png"
        img_pil.save(img_path)
    
    # Convert test data
    print("Converting test data...")
    test_batch = pickle.load(open(cifar_path / "test_batch", 'rb'), encoding='bytes')
    test_images = test_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_labels = test_batch[b'labels']
    
    for idx, (img, label) in enumerate(tqdm(zip(test_images, test_labels), desc="Saving test")):
        class_name = class_names[label]
        img_pil = Image.fromarray(img)
        img_path = output_path / "test" / class_name / f"test_{idx:05d}.png"
        img_pil.save(img_path)
    
    print(f"✅ CIFAR-10 converted successfully to {output_path}")
    print(f"📊 Training samples: {len(train_labels)}")
    print(f"📊 Test samples: {len(test_labels)}")

def convert_cifar100(cifar_dir, output_dir):
    """Convert CIFAR-100 from pickle to ImageFolder"""
    
    cifar_path = Path(cifar_dir)
    output_path = Path(output_dir)
    
    # Load class names
    meta = pickle.load(open(cifar_path / "meta", 'rb'), encoding='bytes')
    fine_label_names = [name.decode('utf-8') for name in meta[b'fine_label_names']]
    
    print(f"Converting CIFAR-100: {len(fine_label_names)} classes")
    
    # Create directory structure
    for split in ['train', 'test']:
        for class_name in fine_label_names:
            (output_path / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Convert training data
    print("Converting training data...")
    train_batch = pickle.load(open(cifar_path / "train", 'rb'), encoding='bytes')
    train_images = train_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    train_labels = train_batch[b'fine_labels']
    
    for idx, (img, label) in enumerate(tqdm(zip(train_images, train_labels), desc="Saving train")):
        class_name = fine_label_names[label]
        img_pil = Image.fromarray(img)
        img_path = output_path / "train" / class_name / f"train_{idx:05d}.png"
        img_pil.save(img_path)
    
    # Convert test data
    print("Converting test data...")
    test_batch = pickle.load(open(cifar_path / "test", 'rb'), encoding='bytes')
    test_images = test_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_labels = test_batch[b'fine_labels']
    
    for idx, (img, label) in enumerate(tqdm(zip(test_images, test_labels), desc="Saving test")):
        class_name = fine_label_names[label]
        img_pil = Image.fromarray(img)
        img_path = output_path / "test" / class_name / f"test_{idx:05d}.png"
        img_pil.save(img_path)
    
    print(f"✅ CIFAR-100 converted successfully to {output_path}")

# Usage examples
if __name__ == "__main__":
    # Convert CIFAR-10
    convert_cifar10("data/cifar-10-batches-py", "data/cifar10_imagefolder")
    
    # Convert CIFAR-100  
    convert_cifar100("data/cifar-100-python", "data/cifar100_imagefolder")
```

**Usage:**
```bash
python convert_cifar.py
```

### **2. CSV/Annotation Files to ImageFolder**

Many datasets come with CSV annotation files. Convert them:

```python
#!/usr/bin/env python3
"""Convert CSV-annotated dataset to ImageFolder structure"""

import pandas as pd
import shutil
from pathlib import Path
from tqdm import tqdm

def convert_csv_dataset(csv_path, images_dir, output_dir, 
                       filename_col='filename', label_col='class', 
                       split_col=None, copy_files=True):
    """
    Convert CSV-annotated dataset to ImageFolder structure
    
    Args:
        csv_path: Path to CSV file with annotations
        images_dir: Directory containing all images
        output_dir: Output directory for ImageFolder structure
        filename_col: Column name for image filenames
        label_col: Column name for class labels
        split_col: Optional column name for train/val/test split
        copy_files: If True, copy files; if False, create symlinks
    """
    
    # Load annotations
    df = pd.read_csv(csv_path)
    print(f"📊 Loaded {len(df)} annotations from {csv_path}")
    
    # Validate required columns
    if filename_col not in df.columns:
        raise ValueError(f"Column '{filename_col}' not found in CSV")
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found in CSV")
    
    # Get unique classes and splits
    classes = sorted(df[label_col].unique())
    splits = ['train'] if split_col is None else sorted(df[split_col].unique())
    
    print(f"🏷️  Classes ({len(classes)}): {classes}")
    print(f"📂 Splits: {splits}")
    
    # Create directory structure
    output_path = Path(output_dir)
    for split in splits:
        for class_name in classes:
            (output_path / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Process each row
    images_path = Path(images_dir)
    success_count = 0
    error_count = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Converting files"):
        try:
            filename = row[filename_col]
            class_name = row[label_col]
            split = 'train' if split_col is None else row[split_col]
            
            # Source and destination paths
            src_path = images_path / filename
            dst_path = output_path / split / class_name / filename
            
            # Check if source exists
            if not src_path.exists():
                print(f"⚠️  Source file not found: {src_path}")
                error_count += 1
                continue
            
            # Copy or link file
            if copy_files:
                shutil.copy2(src_path, dst_path)
            else:
                if dst_path.exists():
                    dst_path.unlink()
                dst_path.symlink_to(src_path.absolute())
            
            success_count += 1
            
        except Exception as e:
            print(f"❌ Error processing {row.get(filename_col, 'unknown')}: {e}")
            error_count += 1
    
    print(f"✅ Conversion complete!")
    print(f"📊 Success: {success_count}, Errors: {error_count}")
    
    # Print dataset statistics
    print("\n📈 Dataset statistics:")
    for split in splits:
        split_path = output_path / split
        if split_path.exists():
            total_files = sum(len(list(class_dir.glob('*'))) 
                            for class_dir in split_path.iterdir() 
                            if class_dir.is_dir())
            print(f"   {split}: {total_files} files")

# Specialized converters for common formats

def convert_imagenet_style(annotations_dir, images_dir, output_dir):
    """Convert ImageNet-style dataset (separate annotation files per split)"""
    
    annotations_path = Path(annotations_dir)
    
    for split_file in annotations_path.glob("*.txt"):
        split_name = split_file.stem  # train, val, test
        
        print(f"Processing {split_name} split...")
        
        with open(split_file, 'r') as f:
            lines = f.readlines()
        
        # Create split dataframe
        data = []
        for line in lines:
            parts = line.strip().split()
            filename = parts[0]
            class_id = int(parts[1])
            data.append({'filename': filename, 'class_id': class_id})
        
        split_df = pd.DataFrame(data)
        split_df['split'] = split_name
        
        # Convert this split
        convert_csv_dataset(
            csv_path=None,  # We'll handle DataFrame directly
            images_dir=images_dir,
            output_dir=output_dir,
            filename_col='filename',
            label_col='class_id',
            split_col='split'
        )

def convert_coco_style(annotations_file, images_dir, output_dir):
    """Convert COCO-style annotations to ImageFolder (for classification)"""
    
    import json
    
    # Load COCO annotations
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create mapping from category_id to category_name
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Create mapping from image_id to filename
    images = {img['id']: img['file_name'] for img in coco_data['images']}
    
    # Process annotations (assumes single label per image)
    data = []
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        category_id = ann['category_id']
        
        data.append({
            'filename': images[image_id],
            'class': categories[category_id]
        })
    
    # Convert to DataFrame and process
    df = pd.DataFrame(data)
    
    # Remove duplicates (keep first occurrence)
    df = df.drop_duplicates(subset=['filename'], keep='first')
    
    # Save temporary CSV and convert
    temp_csv = "temp_coco_annotations.csv"
    df.to_csv(temp_csv, index=False)
    
    convert_csv_dataset(
        csv_path=temp_csv,
        images_dir=images_dir,
        output_dir=output_dir,
        filename_col='filename',
        label_col='class'
    )
    
    # Clean up temporary file
    Path(temp_csv).unlink()

# Usage examples
if __name__ == "__main__":
    # Example 1: Simple CSV with filename and class columns
    convert_csv_dataset(
        csv_path="annotations.csv",
        images_dir="images/",
        output_dir="organized_dataset/",
        filename_col="image_name",
        label_col="category"
    )
    
    # Example 2: CSV with train/val/test split column
    convert_csv_dataset(
        csv_path="annotations_with_splits.csv", 
        images_dir="all_images/",
        output_dir="split_dataset/",
        filename_col="filename",
        label_col="label",
        split_col="split"
    )
```

### **3. Flat Directory to ImageFolder**

Convert a flat directory with filename-encoded labels:

```python
#!/usr/bin/env python3
"""Convert flat directory to ImageFolder based on filename patterns"""

import re
from pathlib import Path
from tqdm import tqdm
import shutil

def convert_flat_directory(input_dir, output_dir, pattern_type="prefix"):
    """
    Convert flat directory to ImageFolder structure
    
    pattern_type options:
    - "prefix": class_image.jpg -> class/class_image.jpg
    - "suffix": image_class.jpg -> class/image_class.jpg  
    - "regex": custom regex pattern
    """
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
    image_files = [f for f in input_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    print(f"📁 Found {len(image_files)} images in {input_dir}")
    
    # Extract classes based on pattern
    classes = set()
    file_class_map = {}
    
    for img_file in image_files:
        filename = img_file.stem  # filename without extension
        
        if pattern_type == "prefix":
            # Format: class_whatever.jpg
            if '_' in filename:
                class_name = filename.split('_')[0]
            else:
                class_name = "unknown"
                
        elif pattern_type == "suffix":
            # Format: whatever_class.jpg
            if '_' in filename:
                class_name = filename.split('_')[-1]
            else:
                class_name = "unknown"
                
        elif pattern_type == "regex":
            # Custom regex pattern (modify as needed)
            # Example: extract class from filename like "img_class_001.jpg"
            match = re.search(r'_([a-zA-Z]+)_', filename)
            class_name = match.group(1) if match else "unknown"
            
        else:
            raise ValueError(f"Unknown pattern_type: {pattern_type}")
        
        classes.add(class_name)
        file_class_map[img_file] = class_name
    
    print(f"🏷️  Detected classes: {sorted(classes)}")
    
    # Create class directories
    train_dir = output_path / "train"
    for class_name in classes:
        (train_dir / class_name).mkdir(parents=True, exist_ok=True)
    
    # Copy files to appropriate directories
    for img_file, class_name in tqdm(file_class_map.items(), desc="Organizing files"):
        dst_path = train_dir / class_name / img_file.name
        shutil.copy2(img_file, dst_path)
    
    print(f"✅ Organization complete: {output_path}")
    
    # Print statistics
    for class_name in sorted(classes):
        class_dir = train_dir / class_name
        count = len(list(class_dir.glob('*')))
        print(f"   {class_name}: {count} images")

# Usage examples
if __name__ == "__main__":
    # Convert based on filename prefix
    convert_flat_directory(
        input_dir="flat_images/",
        output_dir="organized_by_prefix/",
        pattern_type="prefix"
    )
```

---

## 📊 **Dataset-Specific Setup**

### **CIFAR-10/100**

```bash
# Download CIFAR-10
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzf cifar-10-python.tar.gz

# Convert to ImageFolder
python convert_cifar.py

# Analyze
python universal_data_quality_analyzer.py \
  --data data/cifar10_imagefolder/train \
  --normalization cifar10 \
  --preset lightweight
```

### **ImageNet**

```bash
# Assuming ImageNet is already downloaded
# Structure: ImageNet/train/n01440764/*.JPEG

python universal_data_quality_analyzer.py \
  --data ImageNet/train \
  --normalization imagenet \
  --preset accurate \
  --max-samples 50000  # Limit for faster analysis
```

### **Medical Images (Chest X-Ray)**

```bash
# Common structure: chest_xray/train/{NORMAL,PNEUMONIA}/*.jpeg

python universal_data_quality_analyzer.py \
  --data chest_xray/train \
  --backbone resnet50 \
  --normalization imagenet \
  --preset balanced
```

### **Satellite/Aerial Images**

```bash
# EuroSAT or similar
python universal_data_quality_analyzer.py \
  --data eurosat/train \
  --backbone efficientnet_b0 \
  --normalization imagenet \
  --preset balanced
```

### **Fashion/Product Images**

```bash
# Fashion-MNIST converted to images or product catalogs
python universal_data_quality_analyzer.py \
  --data fashion_dataset/train \
  --backbone vit_small_patch16_224 \
  --normalization imagenet \
  --preset balanced
```

---

## ✅ **Validation & Testing**

### **Quick Dataset Validation Script**

```python
#!/usr/bin/env python3
"""Validate dataset structure before analysis"""

from pathlib import Path
import pandas as pd

def validate_dataset(dataset_path):
    """Validate ImageFolder dataset structure"""
    
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return False
    
    # Check for class directories
    class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    if len(class_dirs) == 0:
        print(f"❌ No class directories found in {dataset_path}")
        return False
    
    print(f"✅ Found {len(class_dirs)} class directories")
    
    # Check each class directory
    total_images = 0
    class_stats = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
    
    for class_dir in class_dirs:
        if not class_dir.is_dir():
            continue
            
        images = [f for f in class_dir.iterdir() 
                 if f.suffix.lower() in image_extensions]
        
        class_stats.append({
            'class_name': class_dir.name,
            'image_count': len(images),
            'directory_path': str(class_dir)
        })
        
        total_images += len(images)
    
    # Print statistics
    stats_df = pd.DataFrame(class_stats)
    print(f"\n📊 Dataset Statistics:")
    print(f"Total images: {total_images}")
    print(f"Number of classes: {len(class_dirs)}")
    print(f"Average images per class: {total_images / len(class_dirs):.1f}")
    
    print(f"\n📈 Class Distribution:")
    stats_df = stats_df.sort_values('image_count', ascending=False)
    for _, row in stats_df.iterrows():
        percentage = (row['image_count'] / total_images) * 100
        print(f"   {row['class_name']}: {row['image_count']} ({percentage:.1f}%)")
    
    # Check for potential issues
    print(f"\n🔍 Potential Issues:")
    
    # Empty classes
    empty_classes = stats_df[stats_df['image_count'] == 0]
    if len(empty_classes) > 0:
        print(f"   ⚠️  Empty classes: {list(empty_classes['class_name'])}")
    
    # Very small classes
    small_classes = stats_df[stats_df['image_count'] < 10]
    if len(small_classes) > 0:
        print(f"   ⚠️  Classes with <10 images: {list(small_classes['class_name'])}")
    
    # Imbalanced classes
    max_count = stats_df['image_count'].max()
    min_count = stats_df['image_count'].min()
    if min_count > 0 and (max_count / min_count) > 10:
        print(f"   ⚠️  High class imbalance: {max_count / min_count:.1f}x difference")
    
    # Class name issues
    problematic_names = []
    for class_name in stats_df['class_name']:
        if ' ' in class_name or any(char in class_name for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']):
            problematic_names.append(class_name)
    
    if problematic_names:
        print(f"   ⚠️  Problematic class names: {problematic_names}")
    
    return True

def test_analyzer_compatibility(dataset_path):
    """Test if dataset is compatible with the analyzer"""
    
    print(f"\n🧪 Testing analyzer compatibility...")
    
    try:
        # Try to create a small dataset instance
        from universal_data_quality_analyzer import UniversalImageDataset
        
        test_dataset = UniversalImageDataset(
            root=dataset_path,
            img_size=224,
            normalization="imagenet",
            max_samples=5  # Just test with 5 samples
        )
        
        print(f"✅ Dataset successfully loaded: {len(test_dataset)} samples")
        print(f"✅ Classes detected: {test_dataset.classes}")
        
        # Test loading a sample
        sample = test_dataset[0]
        print(f"✅ Sample loading successful: {sample[0].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python validate_dataset.py <dataset_path>")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    
    print("🔍 DATASET VALIDATION")
    print("=" * 50)
    
    if validate_dataset(dataset_path):
        test_analyzer_compatibility(dataset_path)
    
    print("\n✅ Validation complete!")
```

**Usage:**
```bash
python validate_dataset.py data/my_dataset/train
```

---

## 🎯 **Best Practices**

### **Directory Organization**

```
project/
├── data/
│   ├── raw/                     # Original downloaded data
│   ├── processed/               # Converted to ImageFolder
│   │   ├── train/
│   │   ├── validation/
│   │   └── test/
│   └── quality_reports/         # Analysis results
├── scripts/
│   ├── convert_dataset.py       # Dataset conversion
│   ├── validate_dataset.py      # Validation
│   └── analyze_quality.py       # Quality analysis
└── models/                      # Trained models
```

### **Naming Conventions**

#### **✅ Good Class Names**
- `airplane`, `automobile`, `bird`
- `normal`, `abnormal` 
- `cat`, `dog`, `horse`
- `class_0`, `class_1`, `class_2`

#### **❌ Avoid**
- `class with spaces`
- `class/with/slashes`
- `class:with:colons`
- `class*with*special`

### **Image Quality Guidelines**

#### **Recommended Specifications**
- **Resolution**: 224×224 pixels minimum (will be resized)
- **Format**: JPEG or PNG preferred
- **File size**: 10KB - 10MB per image
- **Color space**: RGB (grayscale will be converted)

#### **Quality Checklist**
- ✅ Images are not corrupted
- ✅ Consistent image quality within classes
- ✅ No duplicate files
- ✅ Meaningful class distinctions
- ✅ Sufficient samples per class (>100 recommended)

### **Performance Optimization**

#### **For Large Datasets**
```bash
# Use sample limiting for initial assessment
python universal_data_quality_analyzer.py \
  --data large_dataset/train \
  --max-samples 10000 \
  --preset lightweight

# Then run full analysis if needed
python universal_data_quality_analyzer.py \
  --data large_dataset/train \
  --preset balanced
```

#### **For Memory-Constrained Systems**
```bash
# Reduce batch size and use lightweight model
python universal_data_quality_analyzer.py \
  --data dataset/train \
  --preset lightweight \
  --batch-size 8 \
  --cv-folds 2
```

---

## 🛠️ **Troubleshooting**

### **Common Setup Issues**

#### **Permission Errors**
```bash
# Fix file permissions
chmod -R 755 dataset/
chown -R $USER:$USER dataset/
```

#### **Symlink Issues (Windows)**
```python
# Use copy instead of symlink in conversion scripts
shutil.copy2(src_path, dst_path)  # Instead of symlink
```

#### **Large File Handling**
```python
# For very large images, add resizing during conversion
from PIL import Image

def safe_resize_and_save(image_array, output_path, max_size=1024):
    img = Image.fromarray(image_array)
    
    # Resize if too large
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    img.save(output_path, optimize=True)
```

### **Dataset-Specific Issues**

#### **CIFAR Conversion Fails**
```python
# Check Python version compatibility
import pickle
import sys
print(f"Python version: {sys.version}")

# Try different encoding
pickle.load(open(file_path, 'rb'), encoding='latin1')  # Instead of 'bytes'
```

#### **CSV Reading Issues**
```python
# Handle different CSV formats
df = pd.read_csv(csv_path, encoding='utf-8')        # Standard
df = pd.read_csv(csv_path, encoding='latin1')       # Legacy
df = pd.read_csv(csv_path, sep=';')                 # Different separator
df = pd.read_csv(csv_path, quotechar='"')           # Quote handling
```

#### **Image Loading Errors**
```python
# Robust image loading
def safe_load_image(image_path):
    try:
        img = Image.open(image_path)
        img = img.convert('RGB')  # Ensure RGB
        return img
    except Exception as e:
        print(f"Failed to load {image_path}: {e}")
        return None
```

### **Memory Management**

#### **For Very Large Datasets**
```python
# Process in chunks
def convert_large_dataset_chunks(input_dir, output_dir, chunk_size=1000):
    all_files = list(Path(input_dir).rglob('*.jpg'))
    
    for i in range(0, len(all_files), chunk_size):
        chunk = all_files[i:i+chunk_size]
        print(f"Processing chunk {i//chunk_size + 1}/{len(all_files)//chunk_size + 1}")
        
        for file_path in chunk:
            # Process individual file
            process_single_file(file_path, output_dir)
        
        # Optional: garbage collection
        import gc
        gc.collect()
```

---

## 📚 **Additional Resources**

### **Dataset Sources**
- **[Papers With Code Datasets](https://paperswithcode.com/datasets)** - Research datasets
- **[Kaggle Datasets](https://www.kaggle.com/datasets)** - Competition datasets  
- **[Google Dataset Search](https://datasetsearch.research.google.com/)** - Dataset discovery
- **[Hugging Face Datasets](https://huggingface.co/datasets)** - ML datasets

### **Conversion Tools**
- **[Roboflow](https://roboflow.com/)** - Computer vision dataset management
- **[Label Studio](https://labelstud.io/)** - Data labeling and conversion
- **[CVAT](https://github.com/openvinotoolkit/cvat)** - Computer vision annotation tool

### **Format Specifications**
- **[ImageFolder PyTorch Docs](https://pytorch.org/vision/stable/datasets.html#imagefolder)**
- **[COCO Format](https://cocodataset.org/#format-data)**
- **[Pascal VOC Format](http://host.robots.ox.ac.uk/pascal/VOC/)**

---

**🎉 Your dataset is now ready for quality analysis! Run the Universal Data Quality Analyzer to identify and fix data issues before training your models.**