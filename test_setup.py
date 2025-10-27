#!/usr/bin/env python3
"""Test script to verify all dependencies are working"""

def test_imports():
    print("Testing imports...")
    
    try:
        import sklearn
        print("✅ scikit-learn:", sklearn.__version__)
    except ImportError as e:
        print("❌ scikit-learn:", e)
    
    try:
        import cleanlab
        print("✅ cleanlab:", cleanlab.__version__)
    except ImportError as e:
        print("❌ cleanlab:", e)
    
    try:
        import cleanvision
        print("✅ cleanvision: installed")
    except ImportError as e:
        print("❌ cleanvision:", e)
    
    try:
        import torch
        print("✅ torch:", torch.__version__)
    except ImportError as e:
        print("❌ torch:", e)
    
    try:
        import timm
        print("✅ timm:", timm.__version__)
    except ImportError as e:
        print("❌ timm:", e)
    
    try:
        import pandas as pd
        print("✅ pandas:", pd.__version__)
    except ImportError as e:
        print("❌ pandas:", e)

if __name__ == "__main__":
    test_imports()
