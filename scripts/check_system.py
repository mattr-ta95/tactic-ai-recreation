#!/usr/bin/env python3
"""
System check script - verify all dependencies are installed
"""

import sys

def check_import(module_name, package_name=None):
    """Check if a module can be imported"""
    if package_name is None:
        package_name = module_name
    
    try:
        __import__(module_name)
        print(f"✅ {package_name:20s} - OK")
        return True
    except ImportError as e:
        print(f"❌ {package_name:20s} - MISSING ({e})")
        return False

print("=" * 60)
print("TacticAI System Check")
print("=" * 60)

all_ok = True

print("\n1. Core Dependencies:")
all_ok &= check_import("torch", "PyTorch")
all_ok &= check_import("numpy", "NumPy")
all_ok &= check_import("pandas", "Pandas")

print("\n2. Graph ML:")
all_ok &= check_import("torch_geometric", "PyTorch Geometric")
all_ok &= check_import("torch_scatter", "torch-scatter")
all_ok &= check_import("torch_sparse", "torch-sparse")

print("\n3. Soccer Data Tools:")
all_ok &= check_import("statsbombpy", "StatsBomb")
all_ok &= check_import("mplsoccer", "mplsoccer")

print("\n4. Visualization:")
all_ok &= check_import("matplotlib", "Matplotlib")
all_ok &= check_import("seaborn", "Seaborn")

print("\n5. ML Utilities:")
all_ok &= check_import("sklearn", "scikit-learn")

print("\n6. Optional Tools:")
check_import("pytorch_lightning", "PyTorch Lightning")
check_import("wandb", "Weights & Biases")
check_import("fastapi", "FastAPI")
check_import("streamlit", "Streamlit")

print("\n" + "=" * 60)

if all_ok:
    print("✅ ALL CORE DEPENDENCIES INSTALLED!")
    print("\nYou're ready to start!")
    print("\nNext steps:")
    print("  1. python scripts/download_data.py")
    print("  2. python scripts/visualize_sample.py")
    print("  3. python scripts/train_baseline.py")
else:
    print("❌ SOME DEPENDENCIES MISSING")
    print("\nPlease install missing packages:")
    print("  pip install -r requirements.txt")
    print("\nFor PyTorch Geometric:")
    print("  pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html")

print("=" * 60)

# Check GPU availability
print("\n7. GPU Check:")
try:
    import torch
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
    elif torch.backends.mps.is_available():
        print("✅ Apple Silicon GPU (MPS) available")
        print("   Training will use Metal Performance Shaders for acceleration")
    else:
        print("⚠️  No GPU available - will use CPU (slower)")
except:
    print("❌ Could not check GPU")

print("\n" + "=" * 60)

sys.exit(0 if all_ok else 1)
