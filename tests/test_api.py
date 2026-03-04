"""Tests for FastAPI endpoints."""

import pytest
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.testclient import TestClient
from torch_geometric.data import Data


@pytest.fixture
def mock_dataset():
    """Create a small dataset for API testing."""
    graphs = []
    for i in range(5):
        num_nodes = 10
        x = torch.rand(num_nodes, 14)
        x[:5, 2] = 1.0
        x[5:, 2] = 0.0
        ei = torch.randint(0, num_nodes, (2, 20))
        ea = torch.randn(20, 3)
        g = Data(x=x, edge_index=ei, edge_attr=ea)
        g.y = torch.tensor([i % 5], dtype=torch.long)
        g.match_id = i
        g.num_players = num_nodes
        graphs.append(g)
    return graphs


@pytest.fixture
def mock_model():
    """Simple model that returns valid logits."""
    from models.gnn import GATCornerNet
    return GATCornerNet(node_features=14, hidden_dim=16, num_layers=2, heads=2, dropout=0.1)


@pytest.fixture
def client(mock_dataset, mock_model):
    """Test client with mock state, bypassing the real lifespan handler."""
    from api.state import _state

    # Build a test app with a no-op lifespan that injects our mocks
    @asynccontextmanager
    async def _test_lifespan(app: FastAPI):
        _state['model'] = mock_model
        _state['dataset'] = mock_dataset
        _state['device'] = 'cpu'
        _state['config'] = {'model_type': 'gat', 'node_features': 14}
        _state['processor'] = None
        _state['optimizer'] = None
        _state['visualizer'] = None
        yield
        _state.clear()
        # Re-init defaults so other tests aren't affected
        _state.update({
            'model': None, 'config': None, 'processor': None,
            'optimizer': None, 'visualizer': None, 'dataset': [], 'device': 'cpu',
        })

    # Create a fresh app with our lifespan
    from fastapi.middleware.cors import CORSMiddleware
    from api.schemas import HealthResponse

    test_app = FastAPI(lifespan=_test_lifespan)
    test_app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_credentials=True,
        allow_methods=["*"], allow_headers=["*"],
    )

    @test_app.get("/api/v1/health", response_model=HealthResponse)
    async def health_check():
        return HealthResponse(
            status="healthy" if _state['model'] else "degraded",
            model_loaded=_state['model'] is not None,
            dataset_size=len(_state['dataset']),
            device=_state['device'],
        )

    from api.routes.corners import router as corners_router
    from api.routes.predict import router as predict_router
    from api.routes.optimize import router as optimize_router

    test_app.include_router(corners_router, prefix="/api/v1")
    test_app.include_router(predict_router, prefix="/api/v1")
    test_app.include_router(optimize_router, prefix="/api/v1")

    with TestClient(test_app) as tc:
        yield tc


class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_health_model_loaded(self, client):
        data = client.get("/api/v1/health").json()
        assert data["model_loaded"] is True

    def test_health_dataset_size(self, client):
        data = client.get("/api/v1/health").json()
        assert data["dataset_size"] == 5


class TestCornersEndpoints:

    def test_list_corners(self, client):
        resp = client.get("/api/v1/corners")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 5

    def test_list_corners_with_limit(self, client):
        resp = client.get("/api/v1/corners?limit=2")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_get_corner_by_id(self, client):
        resp = client.get("/api/v1/corners/0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["corner_id"] == 0
        assert data["num_players"] == 10

    def test_get_corner_not_found(self, client):
        resp = client.get("/api/v1/corners/999")
        assert resp.status_code == 404

    def test_corners_count_reachable(self, client):
        """Fix 7: /corners/count should NOT be caught by /corners/{id}."""
        resp = client.get("/api/v1/corners/count")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert data["total"] == 5


class TestPredictEndpoint:

    def test_predict_with_corner_id(self, client):
        resp = client.post("/api/v1/predict", json={"corner_id": 0})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "predictions" in data
        assert len(data["predictions"]) == 10

    def test_predict_invalid_corner_id(self, client):
        resp = client.post("/api/v1/predict", json={"corner_id": 999})
        assert resp.status_code == 404

    def test_predict_probabilities_sum_to_one(self, client):
        resp = client.post("/api/v1/predict", json={"corner_id": 0})
        data = resp.json()
        total_prob = sum(p["probability"] for p in data["predictions"])
        assert abs(total_prob - 1.0) < 0.05  # allow small rounding error
