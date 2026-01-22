"""TacticAI REST API - FastAPI application."""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add src to path for imports
_src_path = Path(__file__).parent.parent
sys.path.insert(0, str(_src_path))

from api.schemas import HealthResponse
from api.state import _state, get_state


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model and data on startup, cleanup on shutdown."""
    # Import here to avoid circular imports
    from models.gnn import get_model
    from data.processor import CornerKickProcessor
    from tactical.optimizer import TacticalOptimizer
    from tactical.visualization import TacticalVisualizer

    # Detect device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    _state['device'] = device
    print(f"TacticAI API using device: {device}")

    # Load model from checkpoint
    checkpoint_path = os.environ.get(
        'TACTICAI_CHECKPOINT',
        'models/checkpoints/best_model.pth'
    )

    if os.path.exists(checkpoint_path):
        print(f"Loading model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Extract config from checkpoint
        config = checkpoint.get('config', {
            'model_type': 'gat',
            'node_features': 14,
            'hidden_dim': 128,
            'num_layers': 4,
            'dropout': 0.2,
        })
        _state['config'] = config

        # Build model
        model = get_model(
            config.get('model_type', 'gat'),
            node_features=config.get('node_features', 14),
            hidden_dim=config.get('hidden_dim', 128),
            num_layers=config.get('num_layers', 4),
            dropout=config.get('dropout', 0.2),
        )

        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()
        _state['model'] = model
        print(f"  Model type: {config.get('model_type', 'gat')}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    else:
        print(f"Warning: No checkpoint found at {checkpoint_path}")

    # Initialize processor
    config = _state.get('config', {})
    _state['processor'] = CornerKickProcessor(
        distance_threshold=config.get('distance_threshold', 5.0),
        normalize_positions=True,
        use_enhanced_features=config.get('use_enhanced_features', True),
        use_role_features=config.get('use_role_features', True),
        use_positional_context=config.get('use_positional_context', True),
    )

    # Initialize optimizer and visualizer
    if _state['model']:
        _state['optimizer'] = TacticalOptimizer(_state['model'], device=device)
        _state['visualizer'] = TacticalVisualizer()

    # Load dataset
    import pandas as pd
    data_paths = [
        os.environ.get('TACTICAI_DATA', ''),
        'data/processed/training_shots_combined.pkl',
        'data/processed/training_shots.pkl',
        'data/processed/shots_freeze.pkl',
    ]

    for data_path in data_paths:
        if data_path and os.path.exists(data_path):
            print(f"Loading dataset from: {data_path}")
            corners_df = pd.read_pickle(data_path)
            _state['dataset'] = _state['processor'].create_dataset(corners_df)
            # Filter for labeled examples
            _state['dataset'] = [g for g in _state['dataset']
                                 if hasattr(g, 'y') and g.y is not None]
            print(f"  Loaded {len(_state['dataset'])} labeled corners")
            break
    else:
        print("Warning: No dataset found")

    print("TacticAI API ready!")

    yield  # Application runs

    # Cleanup
    print("Shutting down TacticAI API...")
    _state.clear()


# Create FastAPI app
app = FastAPI(
    title="TacticAI API",
    description="REST API for corner kick tactical analysis using Graph Neural Networks",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint (defined here, not in routes)
@app.get("/api/v1/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    """
    Check API health and status.

    Returns model loading status, dataset size, and compute device.
    """
    return HealthResponse(
        status="healthy" if _state['model'] else "degraded",
        model_loaded=_state['model'] is not None,
        dataset_size=len(_state['dataset']),
        device=_state['device'],
    )


# Import and include routers
from api.routes.corners import router as corners_router
from api.routes.predict import router as predict_router
from api.routes.optimize import router as optimize_router

app.include_router(corners_router, prefix="/api/v1", tags=["data"])
app.include_router(predict_router, prefix="/api/v1", tags=["prediction"])
app.include_router(optimize_router, prefix="/api/v1", tags=["optimization"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
