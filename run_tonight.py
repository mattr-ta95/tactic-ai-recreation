#!/usr/bin/env python3
"""
TacticAI - Complete Evening Setup Script (Python Version)
Run this to execute the entire setup automatically
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"🔵 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, shell=True, capture_output=False)
        print(f"✅ {description} - SUCCESS!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED!")
        print(f"Error: {e}")
        return False

def main():
    print("="*60)
    print("🚀 TacticAI Project - Evening Setup")
    print("="*60)
    print()
    
    # Check if in correct directory
    if not Path("README.md").exists():
        print("❌ Error: Not in project directory")
        print("Please run this from the tacticai-project folder")
        sys.exit(1)
    
    # Step 1: System check
    if not run_command("python3 scripts/check_system.py", "Step 1/4: Checking System"):
        print("\n⚠️  Please install dependencies:")
        print("pip install -r requirements.txt")
        sys.exit(1)

    # Step 2: Download data
    if not run_command("python3 scripts/download_data.py --num-matches 50",
                      "Step 2/4: Downloading Data (may take 5 minutes)"):
        sys.exit(1)

    # Step 3: Visualize
    if not run_command("python3 scripts/visualize_sample.py",
                      "Step 3/4: Creating Visualizations"):
        sys.exit(1)

    # Step 4: Train
    if not run_command("python3 scripts/train_baseline.py",
                      "Step 4/4: Training Baseline Model (may take 5 minutes)"):
        sys.exit(1)
    
    # Success!
    print("\n" + "="*60)
    print("🎉 SUCCESS! All steps complete!")
    print("="*60)
    print()
    print("Results:")
    print("  📊 Data: data/processed/corners.pkl")
    print("  📈 Model: models/checkpoints/best_model.pth")
    print("  🎨 Visualizations: visualizations/")
    print("  📝 Progress: See PROGRESS.md")
    print()
    print("Next steps:")
    print("  1. Review visualizations in visualizations/")
    print("  2. Check training results in models/checkpoints/")
    print("  3. Update PROGRESS.md with your results")
    print("  4. Tomorrow: Add receiver labels and evaluation metrics")
    print()
    print("="*60)

if __name__ == "__main__":
    main()
