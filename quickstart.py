#!/usr/bin/env python3
"""
TacticAI Quick Start Script (Python version)
Run this to get started immediately on any platform!
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60 + "\n")

def run_command(cmd, description=""):
    """Run a command and return success status"""
    if description:
        print(f"\n{description}")
    print(f"Running: {cmd}\n")
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0

def main():
    print_header("TacticAI Project - Quick Start")
    
    # Get script directory
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    
    print(f"Current directory: {os.getcwd()}\n")
    
    # Step 1: Check Python
    print_header("Step 1/4: Checking Python installation")
    python_version = sys.version
    print(f"✅ Python version: {python_version}")
    
    if sys.version_info < (3, 10):
        print("⚠️  Warning: Python 3.10+ recommended")
    
    # Step 2: Environment Setup
    print_header("Step 2/4: Environment Setup")
    print("It's recommended to use a virtual environment.")
    print("\nOptions:")
    print("  A) Conda: conda create -n tacticai python=3.10")
    print("  B) Venv: python -m venv venv && source venv/bin/activate")
    print("  C) Install directly (not recommended)")
    
    env_ready = input("\nHave you activated your environment? (y/n): ").lower()
    
    if env_ready != 'y':
        print("\nPlease set up your environment first:")
        print("  1. Create environment (conda or venv)")
        print("  2. Activate it")
        print("  3. Run this script again")
        return
    
    # Step 3: Check Dependencies
    print_header("Step 3/4: Checking Dependencies")
    
    # Try to run check_system.py
    check_result = run_command(
        f"{sys.executable} scripts/check_system.py",
        "Checking installed packages..."
    )
    
    if not check_result:
        print("\n⚠️  Some dependencies are missing.")
        install = input("Install dependencies now? (y/n): ").lower()
        
        if install == 'y':
            print("\nInstalling core dependencies...")
            if not run_command(f"{sys.executable} -m pip install -r requirements.txt"):
                print("❌ Failed to install dependencies")
                return
            
            print("\nInstalling PyTorch Geometric...")
            pyg_cmd = (
                f"{sys.executable} -m pip install torch-geometric torch-scatter "
                f"torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html"
            )
            run_command(pyg_cmd)
            
            print("\n✅ Dependencies installed!")
        else:
            print("\nPlease install dependencies manually:")
            print("  pip install -r requirements.txt")
            return
    
    # Step 4: Download Data
    print_header("Step 4/4: Download Data")
    
    download = input("Download StatsBomb data (50 matches, ~5 minutes)? (y/n): ").lower()
    
    if download == 'y':
        print("\nDownloading data...")
        if run_command(
            f"{sys.executable} scripts/download_data.py --num-matches 50",
            "This may take a few minutes..."
        ):
            print("\n✅ Data downloaded successfully!")
            
            # Optional: Visualize
            viz = input("\nCreate corner visualizations? (y/n): ").lower()
            if viz == 'y':
                run_command(
                    f"{sys.executable} scripts/visualize_sample.py",
                    "Creating visualizations..."
                )
            
            # Optional: Train
            train = input("\nTrain baseline model (~5 minutes)? (y/n): ").lower()
            if train == 'y':
                run_command(
                    f"{sys.executable} scripts/train_baseline.py",
                    "Training model..."
                )
    
    # Final Summary
    print_header("Setup Complete! 🎉")
    
    print("Your TacticAI project is ready!\n")
    print("Next steps:")
    print("  1. Explore data: python scripts/visualize_sample.py")
    print("  2. Train model: python scripts/train_baseline.py")
    print("  3. Check progress: cat PROGRESS.md (or type PROGRESS.md on Windows)")
    print("\nFor detailed guidance:")
    print("  - README.md (project overview)")
    print("  - ../START_HERE.md (getting started guide)")
    print("\nHappy coding! 🚀⚽🤖\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
