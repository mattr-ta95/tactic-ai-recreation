# Next Steps: Moving Forward with Reframed Approach

**Date:** October 22, 2025  
**Status:** Ready to implement and train

## ✅ Completed Reframing

1. **Documentation Updated**
   - README.md reframed with realistic goals
   - PROGRESS.md adjusted with achievable targets
   - REFRAMED_APPROACH.md created with full context

2. **Receiver Labeling Fixed**
   - `_find_receiver()` now uses pass_recipient_id matching (ground truth)
   - Fallback to pass_end_location distance (heuristic)
   - Better data quality (excludes bad labels)

3. **Shot-to-Corner Linking Created**
   - `corner_linker.py` module for linking shots to corner passes
   - Extracts receiver labels from corner passes
   - Links freeze frame positions (shots) to receiver labels (corners)

## 🚀 Immediate Next Steps

### Step 1: Update Data Pipeline
```bash
# Modify download_data.py or create new script to:
# 1. Load shots with freeze frames
# 2. Load full events for corner linking
# 3. Link shots to corners
# 4. Save linked dataset
```

**Action:** Update `scripts/download_data.py` to include shot-to-corner linking step

### Step 2: Test Receiver Labeling
```python
# Test the new receiver labeling
from src.data.processor import CornerKickProcessor
from src.data.corner_linker import link_shots_to_corners
import pandas as pd

# Load data
shots = pd.read_pickle('data/processed/shots_freeze.pkl')
events = pd.read_csv('data/raw/events_pl_2022.csv')

# Link shots to corners
shots_linked = link_shots_to_corners(shots, events)

# Process a sample
processor = CornerKickProcessor()
sample = shots_linked.iloc[0]
graph = processor.corner_to_graph(sample)

print(f"Has receiver label: {hasattr(graph, 'y')}")
```

### Step 3: Create Complete Dataset
- Filter shots that have receiver labels
- Split into train/val/test (by match_id, not randomly)
- Save processed dataset

### Step 4: Train Baseline Model
```bash
python scripts/train_baseline.py
```

**Expected Results:**
- ~650 examples with receiver labels
- Baseline training should complete
- Initial accuracy: 30-45% (will improve with tuning)

### Step 5: Evaluate and Iterate
- Calculate top-1, top-3, top-5 accuracy
- Visualize predictions
- Identify failure cases
- Plan Phase 2 improvements

## 📋 Implementation Checklist

- [x] Reframe project documentation
- [x] Fix receiver labeling method
- [x] Create shot-to-corner linking module
- [ ] Update download script to link shots/corners
- [ ] Test receiver labeling on sample data
- [ ] Create train/val/test splits
- [ ] Train baseline model
- [ ] Evaluate initial performance
- [ ] Document results

## 🎯 Success Criteria (Realistic)

**Phase 1 Complete When:**
- ✅ Model trains without errors
- ✅ Receiver accuracy > 35% (better than random 4.5%)
- ✅ Loss decreases over epochs
- ✅ Can make predictions on new data

**Phase 1 Success:**
- 🎯 Receiver accuracy: 45-55%
- 🎯 Top-3 accuracy: 70%+
- 🎯 Demonstrates methodology works

## 💡 Key Reminders

1. **Focus on learning** - methodology > exact replication
2. **Quality over quantity** - exclude bad data rather than use it
3. **Realistic expectations** - 45-55% is good with limited data
4. **Document limitations** - be honest about data constraints
5. **Iterate and improve** - start simple, enhance gradually

## 🔧 Quick Commands

```bash
# Test receiver labeling
python -c "from src.data.processor import CornerKickProcessor; print('Processor imported successfully')"

# Test corner linking
python -c "from src.data.corner_linker import link_shots_to_corners; print('Linker imported successfully')"

# Check data availability
python -c "import pandas as pd; shots = pd.read_pickle('data/processed/shots_freeze.pkl'); print(f'Shots: {len(shots)}')"
```

---

**Remember:** This is a proof-of-concept. The value is in learning, demonstrating methodology, and building a portfolio project. Perfect replication isn't the goal - understanding and application is! 🚀⚽


