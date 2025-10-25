#!/usr/bin/env python3
"""
Quick Analysis Templates
Pre-configured scripts for common analysis scenarios
"""

import subprocess
import sys
from pathlib import Path

def run_analysis(data_path, config_name, output_path=None):
    """Run analysis with predefined configuration"""
    
    if output_path is None:
        dataset_name = Path(data_path).name
        output_path = f"quality_reports_{dataset_name}_{config_name}"
    
    configs = {
        "quick_check": [
            "--preset", "lightweight",
            "--skip-cleanlab",
            "--batch-size", "16",
            "--cv-folds", "2"
        ],
        
        "medical_images": [
            "--backbone", "resnet50",
            "--preset", "balanced", 
            "--normalization", "imagenet",
            "--cv-folds", "3"
        ],
        
        "cifar_analysis": [
            "--preset", "lightweight",
            "--normalization", "cifar10",
            "--batch-size", "32",
            "--cv-folds", "3"
        ],
        
        "large_dataset": [
            "--preset", "balanced",
            "--max-samples", "10000",
            "--cv-folds", "5",
            "--batch-size", "64"
        ],
        
        "high_accuracy": [
            "--preset", "accurate",
            "--backbone", "vit_base_patch16_224",
            "--cv-folds", "5",
            "--batch-size", "32"
        ],
        
        "low_memory": [
            "--preset", "lightweight",
            "--batch-size", "8",
            "--max-samples", "5000",
            "--cv-folds", "2"
        ]
    }
    
    if config_name not in configs:
        print(f"❌ Unknown config: {config_name}")
        print(f"Available configs: {list(configs.keys())}")
        return False
    
    cmd = [
        sys.executable,
        "universal_data_quality_analyzer.py",
        "--data", data_path,
        "--output", output_path
    ] + configs[config_name]
    
    print(f"🚀 Running {config_name} analysis on {data_path}")
    print(f"📁 Output: {output_path}")
    print(f"⚙️  Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ Analysis complete: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Analysis failed: {e}")
        return False

def batch_analyze_datasets(datasets, config_name="quick_check"):
    """Analyze multiple datasets with the same configuration"""
    
    results = {}
    
    for dataset_path in datasets:
        dataset_name = Path(dataset_path).name
        print(f"\n{'='*60}")
        print(f"Analyzing: {dataset_name}")
        print(f"{'='*60}")
        
        success = run_analysis(dataset_path, config_name)
        results[dataset_name] = success
    
    # Print summary
    print(f"\n{'='*60}")
    print("BATCH ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    for dataset_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{dataset_name}: {status}")
    
    success_count = sum(results.values())
    total_count = len(results)
    print(f"\nOverall: {success_count}/{total_count} datasets analyzed successfully")

def compare_datasets(dataset_paths, config_name="balanced"):
    """Compare multiple datasets using the same analysis settings"""
    
    import pandas as pd
    import json
    
    comparison_results = []
    
    for dataset_path in dataset_paths:
        dataset_name = Path(dataset_path).name
        output_dir = f"comparison_{dataset_name}"
        
        print(f"Analyzing {dataset_name} for comparison...")
        
        if run_analysis(dataset_path, config_name, output_dir):
            # Load metadata
            metadata_path = Path(output_dir) / "analysis_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                comparison_results.append({
                    'dataset_name': dataset_name,
                    'total_samples': metadata['dataset_info']['total_samples'],
                    'num_classes': metadata['dataset_info']['num_classes'],
                    'cv_accuracy': metadata['results_summary']['cv_accuracy'],
                    'feature_dim': metadata['analysis_config']['backbone_model']
                })
    
    if comparison_results:
        # Create comparison report
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df.to_csv("dataset_comparison.csv", index=False)
        
        print(f"\n📊 DATASET COMPARISON")
        print(f"{'='*50}")
        print(comparison_df.to_string(index=False))
        print(f"\n💾 Detailed comparison saved to: dataset_comparison.csv")

# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick Analysis Templates")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Single analysis
    single_parser = subparsers.add_parser('analyze', help='Analyze single dataset')
    single_parser.add_argument('data_path', help='Path to dataset')
    single_parser.add_argument('config', choices=[
        'quick_check', 'medical_images', 'cifar_analysis', 
        'large_dataset', 'high_accuracy', 'low_memory'
    ], help='Analysis configuration')
    single_parser.add_argument('--output', help='Output directory')
    
    # Batch analysis
    batch_parser = subparsers.add_parser('batch', help='Batch analyze multiple datasets')
    batch_parser.add_argument('datasets', nargs='+', help='Paths to datasets')
    batch_parser.add_argument('--config', default='quick_check', choices=[
        'quick_check', 'medical_images', 'cifar_analysis',
        'large_dataset', 'high_accuracy', 'low_memory'
    ], help='Analysis configuration')
    
    # Comparison
    compare_parser = subparsers.add_parser('compare', help='Compare multiple datasets')
    compare_parser.add_argument('datasets', nargs='+', help='Paths to datasets')
    compare_parser.add_argument('--config', default='balanced', choices=[
        'quick_check', 'medical_images', 'cifar_analysis',
        'large_dataset', 'high_accuracy', 'low_memory'
    ], help='Analysis configuration')
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        run_analysis(args.data_path, args.config, args.output)
    elif args.command == 'batch':
        batch_analyze_datasets(args.datasets, args.config)
    elif args.command == 'compare':
        compare_datasets(args.datasets, args.config)
    else:
        parser.print_help()

# Example usage:
"""
# Quick analysis of a single dataset
python quick_templates.py analyze data/my_dataset/train quick_check

# Medical image analysis
python quick_templates.py analyze data/chest_xray/train medical_images --output medical_analysis

# Batch analysis of multiple datasets
python quick_templates.py batch data/dataset1/train data/dataset2/train data/dataset3/train --config quick_check

# Compare datasets
python quick_templates.py compare data/cifar10/train data/cifar100/train --config cifar_analysis
"""