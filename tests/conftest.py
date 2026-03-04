"""Shared fixtures for TacticAI tests."""

import sys
from pathlib import Path

# Ensure src/ is on sys.path before any project imports.
# Also clear any stale 'data' module from sys.modules so that our
# src/data package takes precedence over a pip-installed 'data' package.
_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
for _mod in list(sys.modules):
    if _mod == "data" or _mod.startswith("data."):
        del sys.modules[_mod]

import pytest
import torch
from torch_geometric.data import Data

from models.gnn import get_model, GATCornerNet, SimpleCornerGNN, MultiTaskCornerGNN
from data.processor import CornerKickProcessor


# ---------------------------------------------------------------------------
# Graph fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_freeze_frame():
    """Minimal freeze frame with 6 players (3 attackers, 3 defenders)."""
    return [
        {"location": [110, 35], "teammate": True, "player": {"id": 1}, "position": {"id": 22, "name": "CF"}},
        {"location": [108, 40], "teammate": True, "player": {"id": 2}, "position": {"id": 15, "name": "CM"}},
        {"location": [112, 45], "teammate": True, "player": {"id": 3}, "position": {"id": 5, "name": "CB"}},
        {"location": [109, 36], "teammate": False, "player": {"id": 4}, "position": {"id": 3, "name": "CB"}},
        {"location": [107, 42], "teammate": False, "player": {"id": 5}, "position": {"id": 1, "name": "GK"}},
        {"location": [111, 44], "teammate": False, "player": {"id": 6}, "position": {"id": 9, "name": "DM"}},
    ]


@pytest.fixture
def basic_processor():
    """Processor with only basic features (3 dims)."""
    return CornerKickProcessor(distance_threshold=15.0, normalize_positions=True)


@pytest.fixture
def enhanced_processor():
    """Processor with all feature flags enabled (14 dims)."""
    return CornerKickProcessor(
        distance_threshold=15.0,
        normalize_positions=True,
        use_enhanced_features=True,
        use_role_features=True,
        use_positional_context=True,
    )


@pytest.fixture
def sample_graph_3feat():
    """Graph with 3-dim node features, 10 nodes."""
    num_nodes = 10
    x = torch.rand(num_nodes, 3)
    x[:5, 2] = 1.0   # attackers
    x[5:, 2] = 0.0   # defenders
    edge_index = torch.randint(0, num_nodes, (2, 30))
    edge_attr = torch.randn(30, 3)
    batch = torch.zeros(num_nodes, dtype=torch.long)
    g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    g.y = torch.tensor([2], dtype=torch.long)
    g.match_id = 1
    g.num_players = num_nodes
    g.corner_location = [120.0, 0.0]
    return g


@pytest.fixture
def sample_graph_14feat():
    """Graph with 14-dim node features, 10 nodes."""
    num_nodes = 10
    x = torch.rand(num_nodes, 14)
    x[:5, 2] = 1.0
    x[5:, 2] = 0.0
    edge_index = torch.randint(0, num_nodes, (2, 30))
    edge_attr = torch.randn(30, 3)
    g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    g.y = torch.tensor([2], dtype=torch.long)
    g.match_id = 1
    g.num_players = num_nodes
    g.corner_location = [120.0, 0.0]
    return g


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gat_model_3feat():
    """GAT model for 3-feature input."""
    return GATCornerNet(node_features=3, hidden_dim=16, num_layers=2, heads=2, dropout=0.1)


@pytest.fixture
def gat_model_14feat():
    """GAT model for 14-feature input."""
    return GATCornerNet(node_features=14, hidden_dim=16, num_layers=2, heads=2, dropout=0.1)


@pytest.fixture
def simple_model():
    """Simple GCN model."""
    return SimpleCornerGNN(node_features=3, hidden_dim=16, num_layers=2, dropout=0.1)


@pytest.fixture
def multitask_model():
    """MultiTask GCN model."""
    return MultiTaskCornerGNN(node_features=3, hidden_dim=16, num_layers=2, dropout=0.1)
