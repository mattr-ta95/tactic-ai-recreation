"""Tests for data processing and augmentation."""

import math
import pytest
import torch
import pandas as pd
from torch_geometric.data import Data

from data.processor import CornerKickProcessor, augment_graph, get_data_statistics


class TestCornerKickProcessor:
    """Tests for graph construction from freeze frames."""

    def test_basic_graph_creation(self, basic_processor, sample_freeze_frame):
        """Basic processor should create graph with 3 features per node."""
        row = pd.Series({
            "freeze_frame_parsed": sample_freeze_frame,
            "match_id": 1,
        })
        graph = basic_processor.corner_to_graph(row)
        assert graph.x.shape[0] == len(sample_freeze_frame)
        assert graph.x.shape[1] == 3

    def test_enhanced_graph_creation(self, enhanced_processor, sample_freeze_frame):
        """Enhanced processor should create graph with 14 features per node."""
        row = pd.Series({
            "freeze_frame_parsed": sample_freeze_frame,
            "match_id": 1,
        })
        graph = enhanced_processor.corner_to_graph(row)
        assert graph.x.shape[0] == len(sample_freeze_frame)
        assert graph.x.shape[1] == 14

    def test_edge_creation(self, basic_processor, sample_freeze_frame):
        """Edges should be created between nearby players."""
        row = pd.Series({
            "freeze_frame_parsed": sample_freeze_frame,
            "match_id": 1,
        })
        graph = basic_processor.corner_to_graph(row)
        assert graph.edge_index.shape[0] == 2
        assert graph.edge_index.shape[1] > 0

    def test_edge_attr_shape(self, basic_processor, sample_freeze_frame):
        """Edge attributes should have 3 features (dist, angle, same_team)."""
        row = pd.Series({
            "freeze_frame_parsed": sample_freeze_frame,
            "match_id": 1,
        })
        graph = basic_processor.corner_to_graph(row)
        assert graph.edge_attr.shape[1] == 3

    def test_corner_location_stored(self, enhanced_processor, sample_freeze_frame):
        """Graph should store corner_location for augmentation."""
        row = pd.Series({
            "freeze_frame_parsed": sample_freeze_frame,
            "match_id": 1,
        })
        graph = enhanced_processor.corner_to_graph(row)
        assert hasattr(graph, "corner_location")

    def test_penalty_box_uses_correct_bounds(self, enhanced_processor, sample_freeze_frame):
        """in_box feature (index 6) should use 102/18/62 bounds."""
        # Player at (110, 35) => normalized => should be in box
        row = pd.Series({
            "freeze_frame_parsed": sample_freeze_frame,
            "match_id": 1,
        })
        graph = enhanced_processor.corner_to_graph(row)
        # Player 0 is at (110, 35) which is in the box
        assert graph.x[0, 6].item() == 1.0
        # Player at x=107 is in box if y in [18,62]; player 1 at (108, 40) => in box
        assert graph.x[1, 6].item() == 1.0

    def test_normalized_positions_range(self, basic_processor, sample_freeze_frame):
        """Normalized x, y should be in [0, 1]."""
        row = pd.Series({
            "freeze_frame_parsed": sample_freeze_frame,
            "match_id": 1,
        })
        graph = basic_processor.corner_to_graph(row)
        assert graph.x[:, 0].min() >= 0.0
        assert graph.x[:, 0].max() <= 1.0
        assert graph.x[:, 1].min() >= 0.0
        assert graph.x[:, 1].max() <= 1.0


class TestAugmentation:
    """Tests for augment_graph, validating fixes 2-4."""

    def test_horizontal_flip_x_coordinate(self, sample_graph_14feat):
        """Horizontal flip should invert x: new_x = 1 - old_x."""
        orig_x = sample_graph_14feat.x[:, 0].clone()
        aug = augment_graph(sample_graph_14feat, "horizontal")
        expected = 1.0 - orig_x
        assert torch.allclose(aug.x[:, 0], expected, atol=1e-5)

    def test_horizontal_flip_preserves_y(self, sample_graph_14feat):
        """Horizontal flip should NOT change y."""
        orig_y = sample_graph_14feat.x[:, 1].clone()
        aug = augment_graph(sample_graph_14feat, "horizontal")
        assert torch.allclose(aug.x[:, 1], orig_y, atol=1e-5)

    def test_vertical_flip_y_coordinate(self, sample_graph_14feat):
        """Vertical flip should invert y: new_y = 1 - old_y."""
        orig_y = sample_graph_14feat.x[:, 1].clone()
        aug = augment_graph(sample_graph_14feat, "vertical")
        expected = 1.0 - orig_y
        assert torch.allclose(aug.x[:, 1], expected, atol=1e-5)

    def test_augmentation_recomputes_dist_to_goal(self, sample_graph_14feat):
        """After flip, dist_to_goal (feat 3) should be recomputed, not just copied."""
        g = sample_graph_14feat
        # Set a known position and dist_to_goal
        g.x[0, 0] = 0.5  # x = 60m
        g.x[0, 1] = 0.5  # y = 40m
        g.x[0, 3] = 0.0  # wrong dist_to_goal -- should be recomputed

        aug = augment_graph(g, "horizontal")
        # After horizontal flip: x = 1-0.5 = 0.5 (60m), so dist to (120,40) = 60m
        # normalized: 60/120 = 0.5
        assert abs(aug.x[0, 3].item() - 0.5) < 0.01

    def test_augmentation_recomputes_in_box(self, sample_graph_14feat):
        """After flip, in_box (feat 6) should reflect new position."""
        g = sample_graph_14feat
        # Place player inside box: x=110/120, y=30/80
        g.x[0, 0] = 110.0 / 120.0
        g.x[0, 1] = 30.0 / 80.0
        g.x[0, 6] = 1.0  # currently in box

        # Horizontal flip: new x = 1 - 110/120 = 10/120 ~= 0.083 => 10m (outside box)
        aug = augment_graph(g, "horizontal")
        assert aug.x[0, 6].item() == 0.0  # Should be outside box after flip

    def test_augmentation_recomputes_angle_to_goal(self, sample_graph_14feat):
        """After flip, angle_to_goal (feat 5) should be recomputed from new positions."""
        g = sample_graph_14feat
        g.x[0, 0] = 0.5  # x = 60m
        g.x[0, 1] = 0.5  # y = 40m

        aug = augment_graph(g, "horizontal")
        # After flip x is still 60m (0.5), y still 40m (0.5)
        # angle_to_goal = atan2(40-40, 120-60) / pi = atan2(0, 60) / pi = 0.0
        assert abs(aug.x[0, 5].item()) < 0.01

    def test_horizontal_flip_edge_angle_formula(self):
        """Horizontal flip: edge angle should use sign*1-angle, not -angle."""
        g = Data(
            x=torch.rand(4, 3),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t(),
            edge_attr=torch.tensor([[0.1, 0.5, 1.0], [0.1, -0.3, 1.0]]),
        )
        g.corner_location = [120.0, 0.0]
        aug = augment_graph(g, "horizontal")
        # For angle 0.5 (positive): result should be 1.0 - 0.5 = 0.5
        assert abs(aug.edge_attr[0, 1].item() - 0.5) < 1e-5
        # For angle -0.3 (negative): result should be -1.0 - (-0.3) = -0.7
        assert abs(aug.edge_attr[1, 1].item() - (-0.7)) < 1e-5

    def test_clone_independence(self, sample_graph_3feat):
        """Augmented graph should not modify the original."""
        orig_x = sample_graph_3feat.x.clone()
        _ = augment_graph(sample_graph_3feat, "horizontal")
        assert torch.equal(sample_graph_3feat.x, orig_x)


class TestDataStatistics:
    """Tests for dataset statistics helper."""

    def test_returns_expected_keys(self, sample_graph_3feat):
        stats = get_data_statistics([sample_graph_3feat])
        assert "num_graphs" in stats
        assert "avg_players" in stats
        assert "avg_edges" in stats
