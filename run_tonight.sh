#!/bin/bash
# TacticAI - Complete Evening Setup Script
# Run this to execute the entire setup automatically

set -e  # Exit on error

echo "=========================================="
echo "🚀 TacticAI Project - Evening Setup"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "README.md" ]; then
    echo -e "${RED}❌ Error: Not in project directory${NC}"
    echo "Please run this from the tacticai-project folder"
    exit 1
fi

echo -e "${BLUE}Step 1/4: Checking System...${NC}"
python scripts/check_system.py
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ System check failed. Please install dependencies:${NC}"
    echo "pip install -r requirements.txt"
    exit 1
fi
echo -e "${GREEN}✅ System check passed!${NC}"
echo ""

echo -e "${BLUE}Step 2/4: Downloading Data (this may take 5 minutes)...${NC}"
python scripts/download_data.py --num-matches 50
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Data download failed${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Data downloaded!${NC}"
echo ""

echo -e "${BLUE}Step 3/4: Creating Visualizations...${NC}"
python scripts/visualize_sample.py
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Visualization failed${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Visualizations created!${NC}"
echo ""

echo -e "${BLUE}Step 4/4: Training Baseline Model (this may take 5 minutes)...${NC}"
python scripts/train_baseline.py
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Training failed${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Model trained!${NC}"
echo ""

echo "=========================================="
echo -e "${GREEN}🎉 SUCCESS! All steps complete!${NC}"
echo "=========================================="
echo ""
echo "Results:"
echo "  📊 Data: data/processed/corners.pkl"
echo "  📈 Model: models/checkpoints/best_model.pth"
echo "  🎨 Visualizations: visualizations/"
echo "  📝 Progress: See PROGRESS.md"
echo ""
echo "Next steps:"
echo "  1. Review visualizations in visualizations/"
echo "  2. Check training results in models/checkpoints/"
echo "  3. Update PROGRESS.md with your results"
echo "  4. Tomorrow: Add receiver labels and evaluation metrics"
echo ""
echo "=========================================="
