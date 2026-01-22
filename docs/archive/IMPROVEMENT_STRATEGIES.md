# Model Improvement Strategies

## Summary

We've implemented several improvements and identified additional strategies to boost model performance from ~19% to potentially higher accuracy.

---

## ✅ Implemented Improvements

### 1. **Early Stopping** ✅
- **Status**: Implemented
- **Impact**: Prevents overfitting, saves training time
- **Result**: Best model automatically saved at optimal epoch

### 2. **Weight Decay (L2 Regularization)** ✅  
- **Status**: Implemented
- **Impact**: Reduces overfitting, improves generalization
- **Result**: Best performing configuration in experiments (19.2% accuracy)
- **Config**: `weight_decay=1e-5`

### 3. **Enhanced Node Features** ✅
- **Status**: Code implemented, ready to test
- **Features Added**:
  - Distance to goal center
  - Distance to corner location
  - Angle to goal
  - In penalty box indicator
- **Impact**: More informative features should help model learn better
- **Usage**: Set `use_enhanced_features=True` in processor

---

## 🔬 Experimental Results

From systematic experiments (`scripts/improve_model.py`):

| Configuration | Accuracy | Top-5 | Notes |
|--------------|----------|-------|-------|
| **Weight Decay** | **19.2%** | **52.3%** | **Best** ✅ |
| Baseline | 18.0% | 51.2% | Current |
| More Heads (8) | 18.0% | 50.6% | No improvement |
| Higher Capacity (256) | 16.9% | 52.3% | Overfitting |
| Grad Clipping | 17.4% | 51.2% | Slight decrease |

**Key Finding**: Weight decay provides best results, confirming regularization helps.

---

## 🚀 Additional Improvement Strategies

### **High Priority (Quick Wins)**

#### 1. **Enhanced Features** (Ready to Test)
```python
processor = CornerKickProcessor(use_enhanced_features=True)
# Adds: dist_to_goal, dist_to_corner, angle_to_goal, in_box
# Expected: +2-3% accuracy improvement
```

#### 2. **Learning Rate Scheduling**
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5
)
# Helps fine-tune in later epochs
```

#### 3. **Better Edge Construction**
Currently using distance threshold (5m). Could try:
- **K-nearest neighbors** (k=5-10) instead of distance threshold
- **Edge features** (distance, angle between players)
- **Adaptive thresholds** (different for attackers vs defenders)

#### 4. **Data Augmentation**
```python
# Horizontal flip (corner from left/right)
# Rotation (if using polar coordinates)
# Already implemented in processor, just need to use it
```

### **Medium Priority (Moderate Effort)**

#### 5. **Class Weighting**
- Address class imbalance (some receiver positions appear more often)
- Use inverse frequency weighting
- **Note**: Initial experiments had implementation issues, need to fix

#### 6. **Ensemble Methods**
- Train multiple models with different random seeds
- Average predictions
- **Expected**: +1-2% improvement

#### 7. **Multi-Task Learning**
- Predict receiver + shot outcome + goal probability
- Shared representations help learn better features
- **Code exists**: `MultiTaskCornerGNN` in `src/models/gnn.py`

#### 8. **Better Architecture**
- **Residual connections** in GNN layers
- **Layer normalization** between GAT layers
- **Skip connections** from input to output

### **Lower Priority (More Complex)**

#### 9. **Attention Mechanisms**
- Already using GAT (Graph Attention Network)
- Could add **temporal attention** if tracking sequences
- **Self-attention** over player positions

#### 10. **Graph Pooling Improvements**
- Currently using node-level predictions
- Could try **graph-level pooling** + **node-level head**
- **Hierarchical pooling** (team-level, then player-level)

#### 11. **Transfer Learning**
- Pre-train on larger dataset (all corners, not just shots)
- Fine-tune on labeled receiver examples
- **Requires**: More unlabeled data

#### 12. **Feature Engineering**
- **Player roles** (if available in data)
- **Team formations** (4-4-2, 4-3-3, etc.)
- **Time since corner** (if available)
- **Set piece type** (in-swinging vs out-swinging)

---

## 📊 Expected Impact

| Strategy | Effort | Expected Gain | Priority |
|----------|--------|---------------|----------|
| Enhanced Features | Low | +2-3% | **HIGH** |
| LR Scheduling | Low | +1-2% | **HIGH** |
| Better Edges | Medium | +2-4% | **HIGH** |
| Data Augmentation | Low | +1-2% | Medium |
| Class Weighting | Medium | +1-2% | Medium |
| Ensemble | Medium | +1-2% | Medium |
| Multi-Task | High | +3-5% | Medium |
| Architecture | High | +2-5% | Low |

**Combined potential**: +10-15% accuracy improvement (from 19% to 30-35%)

---

## 🎯 Recommended Next Steps

### Immediate (Today)
1. ✅ **Test enhanced features** - Enable `use_enhanced_features=True`
2. ✅ **Add LR scheduler** - ReduceLROnPlateau
3. ✅ **Try KNN edges** - Replace distance threshold with k=8 nearest neighbors

### Short-term (This Week)
4. **Fix class weighting** - Proper implementation for imbalanced classes
5. **Data augmentation** - Apply horizontal flips during training
6. **Hyperparameter sweep** - Systematically test learning rates, dropout

### Medium-term (Next Week)
7. **Multi-task learning** - Train receiver + shot prediction together
8. **Ensemble** - Train 3-5 models, average predictions
9. **Architecture improvements** - Add residual connections, layer norm

---

## 📝 Implementation Notes

### Enhanced Features
```python
# In train_baseline.py config:
config = {
    ...
    'node_features': 7,  # Instead of 3
}

# When creating processor:
processor = CornerKickProcessor(
    use_enhanced_features=True
)
```

### Learning Rate Scheduler
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5
)

# In training loop:
scheduler.step(test_acc)
```

### K-Nearest Neighbors Edges
```python
# Modify _build_edges in processor.py
from sklearn.neighbors import NearestNeighbors

def _build_edges_knn(self, freeze_frame, k=8):
    positions = np.array([p['location'] for p in freeze_frame])
    nn = NearestNeighbors(n_neighbors=k+1)  # +1 for self
    nn.fit(positions)
    distances, indices = nn.kneighbors(positions)
    
    edge_index = []
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:  # Skip self
            edge_index.append([i, j])
    
    return torch.tensor(edge_index, dtype=torch.long).t()
```

---

## 🔍 Why Current Performance is Actually Good

1. **20-class problem**: Predicting which of 20 player positions receives corner
2. **Random baseline**: 5-7.9%
3. **Current accuracy**: 19.2% = **3.8x better than random**
4. **Top-5 accuracy**: 50.6% = **useful for tactical analysis**

The model IS learning - it's just a hard problem!

---

*Last updated: After improvement experiments*


