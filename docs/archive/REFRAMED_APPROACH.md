# TacticAI Proof-of-Concept: Reframed Approach

**Date:** October 22, 2025  
**Status:** Project reframed with realistic expectations

## 🎯 Reframed Project Goals

### Original Goal
Recreate TacticAI exactly, matching 70%+ receiver accuracy.

### New Goal
Build a **TacticAI-inspired proof-of-concept** demonstrating the methodology with open data, targeting **45-55% receiver accuracy** as a realistic baseline.

## 📊 Reality Check Summary

### Data Comparison

| Metric | TacticAI Paper | This Project | Gap |
|--------|----------------|-------------|-----|
| Dataset Size | 7,176 corners | ~650 labeled examples | 11x smaller |
| Data Source | Commercial StatsBomb | Open StatsBomb | Quality difference |
| Coverage | Full 360 freeze frames | Shots with freeze frames | Different events |
| Receiver Labels | 100% coverage | ~65% coverage | 35% missing |
| Velocity Data | Available | Imputed/Optional | Missing |
| Height Data | Available | Imputed/Optional | Missing |

### Performance Expectations

| Metric | TacticAI | Realistic Target | Improvement |
|--------|----------|------------------|-------------|
| Receiver Accuracy | 70-78% | 45-55% | 10x over random (vs 15x) |
| Shot AUC | 0.75+ | 0.65+ | Good discrimination |
| Top-3 Accuracy | 90%+ | 75%+ | Useful predictions |

## ✅ What's Still Achievable

### Core Functionality
- ✅ Build working GNN architecture
- ✅ Train on real soccer data (923 shots with freeze frames)
- ✅ Predict receivers with meaningful accuracy (45-55%)
- ✅ Generate tactical suggestions
- ✅ Visualize predictions and tactics
- ✅ Demonstrate the methodology works

### Learning Value
- ✅ Deep understanding of Graph Neural Networks
- ✅ Experience with PyTorch Geometric
- ✅ Soccer analytics and event data processing
- ✅ Complete ML pipeline from data to predictions
- ✅ Portfolio project demonstrating technical skills

## ⚠️ What's Not Achievable

- ❌ Matching TacticAI's 70%+ accuracy
- ❌ Full replication with exact same methodology
- ❌ Production-ready system
- ❌ Research publication quality
- ❌ Commercial deployment

## 🛠️ Implementation Adjustments

### Data Approach
1. **Use shots with freeze frames** (not corner passes directly)
2. **Link shots to corner passes** to extract receiver labels
3. **Accept partial coverage** (only use examples with valid labels)
4. **Exclude low-quality data** rather than using bad heuristics

### Technical Decisions
1. **Velocity**: Set to 0 or estimate from positions (optional feature)
2. **Height**: Impute using position averages (optional feature)
3. **Receiver labels**: Use pass_recipient_id matching (ground truth when available)
4. **Fallbacks**: Use pass_end_location distance (heuristic when recipient_id missing)

### Success Metrics
- **Phase 1**: 45-55% receiver accuracy (vs. original 50-60%)
- **Phase 3**: 50-60% with feature engineering (vs. original 70-78%)
- **Key Goal**: Demonstrate methodology > exact replication

## 📝 Key Lessons Learned

1. **Data availability matters**: Open data has limitations
2. **Realistic goals**: Adjust expectations based on resources
3. **Focus on learning**: Methodology > exact replication
4. **Quality over quantity**: Better to exclude bad data than use it
5. **Proof-of-concept value**: Demonstrating approach is valuable

## 🚀 Next Steps

1. ✅ Reframe documentation and goals (completed)
2. ✅ Fix receiver labeling implementation (completed)
3. ⏳ Implement shot-to-corner linking
4. ⏳ Train baseline model
5. ⏳ Evaluate with realistic metrics
6. ⏳ Document results and limitations

## 💡 Project Value Proposition

Despite limitations, this project provides:
- **Educational Value**: Deep learning of GNNs and soccer analytics
- **Portfolio Value**: Demonstrates technical skills and problem-solving
- **Methodology Demonstration**: Shows the approach works, even with constraints
- **Realistic Learning**: Experience with real-world data limitations

**Bottom Line**: This is a valuable learning project that demonstrates the TacticAI methodology, even if it doesn't match the original paper's performance. The journey and learning are what matter most.


