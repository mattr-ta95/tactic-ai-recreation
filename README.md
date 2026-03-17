# TacticAI

A recreation of Google DeepMind's TacticAI system using Graph Neural Networks to analyze and optimize football corner kick tactics. This educational implementation demonstrates the TacticAI methodology with StatsBomb open data.

## What It Does

TacticAI uses **Graph Neural Networks** to:
1. **Predict receivers** - Identify which player will receive the corner kick
2. **Optimize positioning** - Suggest improved attacker positions to maximize scoring chances
3. **Analyze sensitivity** - Determine which player movements have the most tactical impact

### Key Innovation

Unlike traditional ML that treats player positions as independent coordinates, TacticAI models **relationships between players** explicitly through graph structure - players are nodes, spatial relationships are edges.

## Performance

| Metric | Result |
|--------|--------|
| Receiver Prediction Accuracy | **68.2%** |
| Top-3 Accuracy | **94.0%** |
| Top-5 Accuracy | **98.5%** |
| Random Baseline | 4.5% (1/22 players) |
| Improvement Over Random | **15x** |
| Dataset Size | 11,511 labeled corners (67 competitions) |

*Trained on all StatsBomb open data (67 competition-seasons, ~2,700 matches). The original TacticAI achieved 70%+ with commercial StatsBomb 360 data.*

## Quick Start

### 1. Setup Environment

```bash
conda create -n tacticai python=3.10
conda activate tacticai
pip install -r requirements.txt

# PyTorch Geometric (version-sensitive)
pip install torch-geometric torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### 2. Download Data

```bash
# Download all StatsBomb open data (~2,700 matches, 67 competitions)
python scripts/download_all_statsbomb.py --validate --download

# Or download a smaller subset for quick testing
python scripts/download_data.py --num-matches 100
```

### 3. Prepare Training Data

```bash
python scripts/prepare_training_data.py
python scripts/generate_synthetic_corners.py --combine
```

### 4. Train Model

```bash
PYTHONPATH=src python scripts/train_baseline.py
```

### 5. Launch API & Dashboard

```bash
# Terminal 1: Start API server
python3 -m uvicorn src.api.main:app --reload --port 8000

# Terminal 2: Start dashboard
python3 -m streamlit run dashboard/app.py --server.port 8501
```

Then open http://localhost:8501 in your browser.

## Project Structure

```
tacticai-project/
├── src/
│   ├── api/                 # FastAPI REST API
│   │   ├── main.py          # App entry point
│   │   ├── routes/          # API endpoints
│   │   └── schemas.py       # Pydantic models
│   ├── data/                # Data processing
│   │   └── processor.py     # Corner -> Graph conversion
│   ├── models/              # GNN architectures
│   │   └── gnn.py           # GAT, GCN models
│   └── tactical/            # Optimization system
│       ├── optimizer.py     # Position optimization
│       └── visualization.py # Pitch plots
├── dashboard/               # Streamlit UI
│   ├── app.py               # Main dashboard
│   └── pages/               # Multi-page app
│       ├── 01_optimizer.py  # Position optimizer
│       ├── 02_explorer.py   # Dataset browser
│       └── 03_custom_corner.py  # Custom scenarios
├── scripts/                 # CLI utilities
│   ├── download_all_statsbomb.py  # Bulk download all open data
│   ├── download_data.py     # Fetch StatsBomb data (subset)
│   ├── prepare_training_data.py   # Link shots to corners
│   ├── generate_synthetic_corners.py  # Synthetic data augmentation
│   ├── train_baseline.py    # Train models
│   └── visualize_sample.py  # Sample visualizations
├── data/                    # Data directory (not in git)
├── models/checkpoints/      # Saved models (not in git)
└── docs/                    # Documentation
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | System status |
| `/api/v1/corners` | GET | List corner kicks |
| `/api/v1/corners/{id}` | GET | Get corner details |
| `/api/v1/predict` | POST | Predict receivers |
| `/api/v1/optimize` | POST | Optimize positions |
| `/api/v1/analyze/sensitivity` | POST | Sensitivity analysis |

## Dashboard Pages

- **Home** - Project overview and quick start
- **Optimizer** - Run position optimization on corners
- **Explorer** - Browse and visualize training data
- **Custom Corner** - Create and analyze custom scenarios

## Documentation

- [Quick Start Guide](docs/QUICKSTART.md) - Get running in 15 minutes
- [Project Plan](docs/PROJECT_PLAN.md) - Development roadmap
- [Technical Reference](docs/TECHNICAL_REFERENCE.md) - Architecture details
- [Progress Log](PROGRESS.md) - Development history

## Development

### Run Tests
```bash
pytest tests/
```

### Verify System
```bash
python scripts/check_system.py
```

### Train with Custom Config
```bash
python scripts/train_baseline.py --hidden-dim 256 --num-layers 6
```

## Tech Stack

- **PyTorch 2.0+** - Deep learning
- **PyTorch Geometric** - Graph neural networks
- **FastAPI** - REST API
- **Streamlit** - Dashboard UI
- **StatsBomb** - Open soccer data
- **mplsoccer** - Pitch visualizations

## References

- [TacticAI Paper](https://www.nature.com/articles/s41467-024-45965-x) (Nature Communications, 2024)
- [StatsBomb Open Data](https://github.com/statsbomb/open-data)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io)

## License

MIT License. See [LICENSE](LICENSE) for details.

This is an educational recreation. Original TacticAI by Google DeepMind.
Data from StatsBomb under their open data license.
