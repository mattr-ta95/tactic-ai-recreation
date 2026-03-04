"""Tests for GNN model architectures."""

import pytest
import torch
from torch_geometric.data import Data

from models.gnn import get_model, GATCornerNet, SimpleCornerGNN, MultiTaskCornerGNN


class TestSimpleCornerGNN:
    """Tests for the SimpleCornerGNN baseline model."""

    def test_forward_shape(self, simple_model, sample_graph_3feat):
        """Output should be [num_nodes] logits."""
        g = sample_graph_3feat
        out = simple_model(g.x, g.edge_index, batch=torch.zeros(g.num_nodes, dtype=torch.long))
        assert out.shape == (g.num_nodes,)

    def test_accepts_edge_attr(self, simple_model, sample_graph_3feat):
        """Should accept edge_attr kwarg without crashing (even if unused)."""
        g = sample_graph_3feat
        batch = torch.zeros(g.num_nodes, dtype=torch.long)
        out = simple_model(g.x, g.edge_index, batch=batch, edge_attr=g.edge_attr)
        assert out.shape == (g.num_nodes,)

    def test_returns_tensor(self, simple_model, sample_graph_3feat):
        """Return type should be a plain tensor, not a dict."""
        g = sample_graph_3feat
        out = simple_model(g.x, g.edge_index)
        assert isinstance(out, torch.Tensor)


class TestGATCornerNet:
    """Tests for the GAT model with attention."""

    def test_forward_shape_3feat(self, gat_model_3feat, sample_graph_3feat):
        g = sample_graph_3feat
        out = gat_model_3feat(g.x, g.edge_index)
        assert out.shape == (g.num_nodes,)

    def test_forward_shape_14feat(self, gat_model_14feat, sample_graph_14feat):
        g = sample_graph_14feat
        out = gat_model_14feat(g.x, g.edge_index)
        assert out.shape == (g.num_nodes,)

    def test_with_edge_features(self):
        """GAT with edge_dim should use edge features in attention."""
        model = GATCornerNet(node_features=3, hidden_dim=16, num_layers=2, heads=2, edge_dim=3)
        x = torch.randn(8, 3)
        ei = torch.randint(0, 8, (2, 20))
        ea = torch.randn(20, 3)
        out = model(x, ei, edge_attr=ea)
        assert out.shape == (8,)

    def test_returns_tensor(self, gat_model_3feat, sample_graph_3feat):
        g = sample_graph_3feat
        out = gat_model_3feat(g.x, g.edge_index)
        assert isinstance(out, torch.Tensor)


class TestMultiTaskCornerGNN:
    """Tests for the multi-task model."""

    def test_returns_dict(self, multitask_model, sample_graph_3feat):
        """MultiTask model should return a dict with receiver/shot/goal keys."""
        g = sample_graph_3feat
        batch = torch.zeros(g.num_nodes, dtype=torch.long)
        out = multitask_model(g.x, g.edge_index, batch)
        assert isinstance(out, dict)
        assert "receiver" in out
        assert "shot" in out
        assert "goal" in out

    def test_receiver_shape(self, multitask_model, sample_graph_3feat):
        g = sample_graph_3feat
        batch = torch.zeros(g.num_nodes, dtype=torch.long)
        out = multitask_model(g.x, g.edge_index, batch)
        assert out["receiver"].shape == (g.num_nodes,)

    def test_accepts_edge_attr(self, multitask_model, sample_graph_3feat):
        """Should accept edge_attr kwarg without crashing."""
        g = sample_graph_3feat
        batch = torch.zeros(g.num_nodes, dtype=torch.long)
        out = multitask_model(g.x, g.edge_index, batch, edge_attr=g.edge_attr)
        assert "receiver" in out

    def test_batch_none_default(self, multitask_model, sample_graph_3feat):
        """batch=None should not crash (auto-creates single-graph batch)."""
        g = sample_graph_3feat
        out = multitask_model(g.x, g.edge_index)
        assert "receiver" in out


class TestGetModelFactory:
    """Tests for the model factory function."""

    def test_simple(self):
        m = get_model("simple", node_features=3, hidden_dim=16, num_layers=2)
        assert isinstance(m, SimpleCornerGNN)

    def test_gat(self):
        m = get_model("gat", node_features=3, hidden_dim=16, num_layers=2)
        assert isinstance(m, GATCornerNet)

    def test_multitask(self):
        m = get_model("multitask", node_features=3, hidden_dim=16, num_layers=2)
        assert isinstance(m, MultiTaskCornerGNN)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown model type"):
            get_model("nonexistent")
