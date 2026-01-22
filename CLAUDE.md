# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TacticAI is a recreation of Google DeepMind's geometric deep learning system for soccer tactical analysis. The system uses Graph Neural Networks (GNNs) to analyze corner kicks by representing players as nodes and their relationships as edges, predicting receivers and optimizing player positioning.

**Key Innovation**: Models player relationships explicitly through graph structure rather than treating positions as independent coordinates.

**Current Performance**: 45.6% receiver prediction accuracy (10x improvement over random baseline).

## Quick Start

### Environment Setup
```bash
conda create -n tacticai python=3.10
conda activate tacticai
pip install -r requirements.txt

# PyTorch Geometric (version-sensitive)
pip install torch-geometric torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### Run API & Dashboard
```bash
# Terminal 1: Start API server
python3 -m uvicorn src.api.main:app --reload --port 8000

# Terminal 2: Start dashboard
python3 -m streamlit run dashboard/app.py --server.port 8501
```

### Train Model
```bash
python scripts/download_data.py --num-matches 100
python scripts/train_baseline.py
```

## Project Structure

```
tacticai-project/
├── src/
│   ├── api/                     # FastAPI REST API
│   │   ├── main.py              # App entry point with lifespan handler
│   │   ├── state.py             # Shared application state
│   │   ├── schemas.py           # Pydantic request/response models
│   │   ├── utils.py             # Graph serialization helpers
│   │   └── routes/
│   │       ├── corners.py       # /corners endpoints
│   │       ├── predict.py       # /predict endpoint
│   │       └── optimize.py      # /optimize, /analyze endpoints
│   ├── data/
│   │   └── processor.py         # Corner -> Graph conversion
│   ├── models/
│   │   └── gnn.py               # GNN architectures (GAT, GCN)
│   └── tactical/
│       ├── optimizer.py         # Position optimization
│       └── visualization.py     # Pitch visualizations
├── dashboard/                   # Streamlit UI
│   ├── app.py                   # Main dashboard page
│   └── pages/
│       ├── 01_optimizer.py      # Position optimizer
│       ├── 02_explorer.py       # Dataset browser
│       └── 03_custom_corner.py  # Custom scenario builder
├── scripts/
│   ├── download_data.py         # Fetch StatsBomb data
│   ├── train_baseline.py        # Train GNN model
│   └── visualize_sample.py      # Sample visualizations
├── data/                        # Data directory (gitignored)
├── models/checkpoints/          # Saved models (gitignored)
└── docs/                        # Documentation
```

## API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | System status (model loaded, dataset size, device) |
| `/api/v1/corners` | GET | List corners with pagination (`skip`, `limit`) |
| `/api/v1/corners/{id}` | GET | Get corner details, optionally with graph (`include_graph=true`) |
| `/api/v1/predict` | POST | Predict receiver probabilities |
| `/api/v1/predict/top` | POST | Get top N most likely receivers |
| `/api/v1/optimize` | POST | Optimize attacker positions |
| `/api/v1/optimize/quick` | POST | Quick optimization with defaults |
| `/api/v1/analyze/sensitivity` | POST | Analyze player position sensitivity |

### Request Examples

**Predict receiver for corner #42:**
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"corner_id": 42}'
```

**Optimize positions:**
```bash
curl -X POST "http://localhost:8000/api/v1/optimize" \
  -H "Content-Type: application/json" \
  -d '{"corner_id": 42, "target_receiver": 5, "num_iterations": 50}'
```

**Custom corner setup:**
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "corner_setup": {
      "players": [
        {"x": 110, "y": 35, "is_attacker": true},
        {"x": 108, "y": 40, "is_attacker": true},
        {"x": 105, "y": 38, "is_attacker": false}
      ],
      "corner_location": "right"
    }
  }'
```

### API State Management

The API uses a shared state module (`src/api/state.py`) to avoid circular imports:

```python
from api.state import get_state

state = get_state()
model = state['model']        # Trained GNN model
dataset = state['dataset']    # List of PyG Data objects
processor = state['processor'] # CornerKickProcessor instance
optimizer = state['optimizer'] # TacticalOptimizer instance
device = state['device']       # 'cuda', 'mps', or 'cpu'
```

## Dashboard Pages

1. **Home** (`app.py`) - Overview, quick start instructions
2. **Optimizer** (`01_optimizer.py`) - Run position optimization on existing corners
3. **Explorer** (`02_explorer.py`) - Browse dataset, visualize corners, filter by attributes
4. **Custom Corner** (`03_custom_corner.py`) - Create custom scenarios, run predictions

## Architecture

### Graph Representation

**Core Concept**: Corner kicks -> PyTorch Geometric graphs
- **Nodes**: Players with features (14 dimensions including position, team, role)
- **Edges**: Connect players within distance threshold (default 5m)
- **Target**: Receiver label (index of player who receives ball)

### Models (`src/models/gnn.py`)

1. **GATCornerNet** (default): Multi-head attention with residual connections
2. **SimpleCornerGNN**: Basic GCN baseline
3. **MultiTaskCornerGNN**: Predicts receiver + shot/goal probabilities

Factory: `get_model(model_type='gat'|'simple'|'multitask', **kwargs)`

### Tactical Optimizer (`src/tactical/optimizer.py`)

Gradient-based optimization that:
1. Takes a corner setup and target receiver
2. Optimizes attacker positions to maximize receiver probability
3. Respects constraints (pitch boundaries, minimum player spacing)
4. Returns position changes with visualizations

## Common Operations

### Load Model Manually
```python
import torch
from src.models.gnn import get_model

checkpoint = torch.load('models/checkpoints/best_model.pth', map_location='cpu')
model = get_model('gat', **checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Create Graph from Positions
```python
from src.data.processor import CornerKickProcessor

processor = CornerKickProcessor()
graph = processor.positions_to_graph(
    positions=[(110, 35, True), (108, 40, True), (105, 38, False)],
    corner_location='right'
)
```

### Run Optimization
```python
from src.tactical.optimizer import TacticalOptimizer

optimizer = TacticalOptimizer(model, device='cpu')
result = optimizer.optimize_positions(
    graph,
    target_receiver=5,
    num_iterations=50,
    learning_rate=0.1
)
print(f"Improvement: {result.improvement_percentage:.1f}%")
```

## Troubleshooting

### "command not found: uvicorn"
```bash
# Use python module syntax
python3 -m uvicorn src.api.main:app --reload --port 8000
```

### "command not found: streamlit"
```bash
# Use python module syntax
python3 -m streamlit run dashboard/app.py --server.port 8501
```

### "No corners available in dataset"
1. Ensure API server is running on port 8000
2. Check data exists: `ls data/processed/training_shots*.pkl`
3. Restart API server after code changes

### "Circular import" errors
The API uses `src/api/state.py` to hold shared state. Always import `get_state` from `state.py`, not from `main.py`.

### "CUDA out of memory"
Reduce batch size in `scripts/train_baseline.py` or use CPU:
```bash
export TACTICAI_DEVICE=cpu
python3 -m uvicorn src.api.main:app --port 8000
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TACTICAI_CHECKPOINT` | Path to model checkpoint | `models/checkpoints/best_model.pth` |
| `TACTICAI_DATA` | Path to training data | `data/processed/training_shots_combined.pkl` |

## File Locations

**Source Code**
- `src/api/` - FastAPI REST API
- `src/data/processor.py` - Graph construction
- `src/models/gnn.py` - GNN architectures
- `src/tactical/` - Optimization and visualization

**Data** (gitignored)
- `data/processed/training_shots_combined.pkl` - Combined training data
- `data/processed/shots_freeze.pkl` - StatsBomb shots with freeze frames

**Models** (gitignored)
- `models/checkpoints/best_model.pth` - Trained model weights
- `models/checkpoints/training_history.json` - Training metrics

**Dashboard**
- `dashboard/app.py` - Main page
- `dashboard/pages/` - Additional pages

## Dependencies

**Core**
- PyTorch 2.0+, PyTorch Geometric 2.3+
- FastAPI, Uvicorn
- Streamlit

**Data & Visualization**
- statsbombpy, mplsoccer
- pandas, numpy, matplotlib

See `requirements.txt` for full list.
