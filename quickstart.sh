#!/bin/bash

# TacticAI Quick Start Script
# Run this to get started immediately!

set -e  # Exit on error

echo "============================================================"
echo "TacticAI Project - Quick Start"
echo "============================================================"
echo ""

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Current directory: $(pwd)"
echo ""

# Step 1: Check System
echo "Step 1/4: Checking system dependencies..."
echo "============================================================"
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "❌ Python not found! Please install Python 3.10+"
    exit 1
fi

echo "✅ Found Python: $($PYTHON --version)"
echo ""

# Step 2: Check/Create Environment
echo "Step 2/4: Environment setup..."
echo "============================================================"
echo "You can:"
echo "  A) Create a conda environment: conda create -n tacticai python=3.10"
echo "  B) Use a virtual environment: python -m venv venv && source venv/bin/activate"
echo "  C) Install directly (not recommended): pip install -r requirements.txt"
echo ""
read -p "Have you set up your environment? (y/n): " ENV_READY

if [[ "$ENV_READY" != "y" ]]; then
    echo "Please set up your environment first, then run this script again."
    exit 0
fi

# Step 3: Check if dependencies are installed
echo ""
echo "Step 3/4: Checking dependencies..."
echo "============================================================"
$PYTHON scripts/check_system.py

if [ $? -ne 0 ]; then
    echo ""
    echo "Some dependencies are missing. Install them with:"
    echo "  pip install -r requirements.txt"
    echo ""
    read -p "Install now? (y/n): " INSTALL

    if [[ "$INSTALL" == "y" ]]; then
        echo "Installing dependencies..."
        pip install -r requirements.txt
        
        echo ""
        echo "Installing PyTorch Geometric..."
        pip install torch-geometric torch-scatter torch-sparse torch-cluster \
            -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
    else
        echo "Please install dependencies manually, then run this script again."
        exit 0
    fi
fi

# Step 4: Download data
echo ""
echo "Step 4/4: Download data..."
echo "============================================================"
read -p "Download StatsBomb data (50 matches)? (y/n): " DOWNLOAD

if [[ "$DOWNLOAD" == "y" ]]; then
    echo "Downloading data..."
    $PYTHON scripts/download_data.py --num-matches 50
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Data downloaded successfully!"
        
        # Optional: Visualize
        echo ""
        read -p "Create visualizations? (y/n): " VIZ
        if [[ "$VIZ" == "y" ]]; then
            echo "Creating visualizations..."
            $PYTHON scripts/visualize_sample.py
        fi
        
        # Optional: Train
        echo ""
        read -p "Train baseline model? (y/n): " TRAIN
        if [[ "$TRAIN" == "y" ]]; then
            echo "Training baseline model..."
            $PYTHON scripts/train_baseline.py
        fi
    fi
fi

echo ""
echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Visualize data: python scripts/visualize_sample.py"
echo "  2. Train model: python scripts/train_baseline.py"
echo "  3. Check progress: cat PROGRESS.md"
echo ""
echo "For detailed guidance, read:"
echo "  - README.md (project overview)"
echo "  - START_HERE.md (in parent directory)"
echo ""
echo "Happy coding! 🚀⚽🤖"
