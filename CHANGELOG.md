# Changelog

## [0.3.0] - 2026-03-17

### Added
- `scripts/download_all_statsbomb.py` - Bulk download pipeline for all StatsBomb open data with `--validate` and `--download` modes
- Support for 67 competition-seasons (~2,700 matches) from StatsBomb open data
- Validation results saved to `data/validation_results.json` for reproducibility

### Changed
- `scripts/prepare_training_data.py` now prefers `corners.pkl` over events CSV, avoiding column mismatch across competitions
- `scripts/train_baseline.py` batch size reduced from 32 to 16 for better memory stability on MPS/CPU
- Dataset grew from 1,839 to 11,511 labeled corners (6x increase)

### Fixed
- `pd.notna()` bug in `src/data/processor.py` that failed on list/array values from multi-league data (graph creation success: 33% -> 100%)

### Performance
- Receiver prediction accuracy: **68.2%** (was 45.6%)
- Top-3 accuracy: 94.0%
- Top-5 accuracy: 98.5%

## [0.2.0] - 2026-03-03

### Fixed
- Replaced `eval()` with `ast.literal_eval` for security (download_data.py, config.py)
- Augmentation now recomputes all derived features after coordinate flips
- Horizontal flip edge angle formula corrected
- Penalty box constants unified to (102, 18-62)
- Global random seeds (42) + sorted match IDs for reproducibility
- All 3 model `forward()` signatures accept `edge_attr=None`
- `/corners/count` route moved before `/corners/{id}` to prevent shadowing
- Removed `sys.path.insert` hacks from main.py and train_baseline.py

### Added
- Test suite: 52 tests across models, processor, API, and config
- Cleaned requirements.txt with missing dependencies added

### Performance
- Receiver prediction accuracy: 45.6% (10x over random baseline)

## [0.1.0] - 2026-01-22

### Added
- Initial implementation of TacticAI corner kick analysis system
- GAT, GCN, and MultiTask GNN architectures
- FastAPI REST API with prediction, optimization, and sensitivity endpoints
- Streamlit dashboard with optimizer, explorer, and custom corner pages
- StatsBomb data download pipeline for 4 competitions (WC 2018/2022, Euro 2020/2024)
- Gradient-based tactical position optimizer
