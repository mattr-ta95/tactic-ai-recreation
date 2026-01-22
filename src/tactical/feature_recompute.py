"""Feature recomputation utilities for tactical optimization."""

import torch
import math
from typing import Tuple, Optional


class FeatureRecomputer:
    """
    Recompute derived features after position changes.

    Feature layout (14 total):
    [0] x_norm - position x (LEARNABLE)
    [1] y_norm - position y (LEARNABLE)
    [2] is_teammate - team indicator (FIXED)
    [3] dist_to_goal - distance to goal center (RECOMPUTE)
    [4] dist_to_corner - distance to corner location (RECOMPUTE)
    [5] angle_to_goal - angle to goal (RECOMPUTE)
    [6] in_box - penalty box indicator (RECOMPUTE)
    [7] is_gk - goalkeeper role (FIXED)
    [8] is_def - defender role (FIXED)
    [9] is_mid - midfielder role (FIXED)
    [10] is_fwd - forward role (FIXED)
    [11] dist_to_nearest_teammate (RECOMPUTE)
    [12] dist_to_nearest_opponent (RECOMPUTE)
    [13] positional_depth (RECOMPUTE)
    """

    def __init__(
        self,
        pitch_length: float = 120.0,
        pitch_width: float = 80.0,
        goal_center: Optional[Tuple[float, float]] = None,
        corner_location: Optional[Tuple[float, float]] = None,
    ):
        """
        Initialize with pitch parameters.

        Args:
            pitch_length: Pitch length in meters (default 120)
            pitch_width: Pitch width in meters (default 80)
            goal_center: Goal center position (default: center of goal line)
            corner_location: Corner kick location (used for dist_to_corner)
        """
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.goal_center = goal_center or (pitch_length, pitch_width / 2)
        self.corner_location = corner_location

        # Penalty box dimensions (StatsBomb coordinates)
        self.penalty_box_x_min = 102.0  # 18 yards from goal
        self.penalty_box_y_min = 18.0   # 18 yards from each post
        self.penalty_box_y_max = 62.0

    def recompute_all(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Recompute all position-dependent features.

        Args:
            x: Current node features [num_nodes, num_features]
            positions: New positions [num_nodes, 2] (denormalized, in meters)

        Returns:
            Updated features tensor [num_nodes, num_features]
        """
        x = x.clone()
        num_features = x.shape[1]
        device = x.device

        # Update normalized positions (features 0-1)
        x[:, 0] = positions[:, 0] / self.pitch_length
        x[:, 1] = positions[:, 1] / self.pitch_width

        # Get team mask for positional context
        teammate_mask = x[:, 2] > 0.5  # is_teammate feature

        # Recompute enhanced features (3-6) if they exist
        if num_features > 3:
            # Feature 3: dist_to_goal
            goal_pos = torch.tensor(self.goal_center, device=device)
            dist_to_goal = torch.norm(positions - goal_pos, dim=1)
            x[:, 3] = dist_to_goal / self.pitch_length

        if num_features > 4 and self.corner_location is not None:
            # Feature 4: dist_to_corner
            corner_pos = torch.tensor(self.corner_location, device=device)
            dist_to_corner = torch.norm(positions - corner_pos, dim=1)
            x[:, 4] = dist_to_corner / self.pitch_length

        if num_features > 5:
            # Feature 5: angle_to_goal
            goal_pos = torch.tensor(self.goal_center, device=device)
            dx = goal_pos[0] - positions[:, 0]
            dy = goal_pos[1] - positions[:, 1]
            angle_to_goal = torch.atan2(dy, dx)
            x[:, 5] = angle_to_goal / math.pi

        if num_features > 6:
            # Feature 6: in_box
            in_box = (
                (positions[:, 0] >= self.penalty_box_x_min) &
                (positions[:, 1] >= self.penalty_box_y_min) &
                (positions[:, 1] <= self.penalty_box_y_max)
            ).float()
            x[:, 6] = in_box

        # Recompute positional context (11-13) if they exist
        if num_features > 13:
            dist_teammate, dist_opponent, depth = self._compute_positional_context(
                positions, teammate_mask
            )
            x[:, 11] = dist_teammate
            x[:, 12] = dist_opponent
            x[:, 13] = depth

        return x

    def _compute_positional_context(
        self,
        positions: torch.Tensor,
        teammate_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute contextual features (nearest teammate/opponent, depth).

        Args:
            positions: All player positions [N, 2]
            teammate_mask: Boolean mask for teammates [N]

        Returns:
            (dist_to_nearest_teammate, dist_to_nearest_opponent, positional_depth)
        """
        num_players = positions.shape[0]
        device = positions.device

        dist_to_nearest_teammate = torch.ones(num_players, device=device)
        dist_to_nearest_opponent = torch.ones(num_players, device=device)

        # Compute pairwise distances
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # [N, N, 2]
        distances = torch.norm(diff, dim=2)  # [N, N]

        # Set diagonal to large value to exclude self
        distances = distances + torch.eye(num_players, device=device) * 1000.0

        for i in range(num_players):
            is_teammate = teammate_mask[i]

            # Find teammates and opponents
            same_team_mask = teammate_mask == is_teammate
            same_team_mask[i] = False  # Exclude self

            diff_team_mask = teammate_mask != is_teammate

            # Distance to nearest teammate
            if same_team_mask.any():
                teammate_dists = distances[i, same_team_mask]
                dist_to_nearest_teammate[i] = teammate_dists.min() / self.pitch_length
            else:
                dist_to_nearest_teammate[i] = 1.0

            # Distance to nearest opponent
            if diff_team_mask.any():
                opponent_dists = distances[i, diff_team_mask]
                dist_to_nearest_opponent[i] = opponent_dists.min() / self.pitch_length
            else:
                dist_to_nearest_opponent[i] = 1.0

        # Positional depth: player x relative to team centroid
        positional_depth = torch.zeros(num_players, device=device)
        for is_attacker in [True, False]:
            team_mask = teammate_mask == is_attacker
            if team_mask.sum() > 0:
                team_positions = positions[team_mask]
                centroid_x = team_positions[:, 0].mean()
                depth = (positions[team_mask, 0] - centroid_x) / self.pitch_length
                positional_depth[team_mask] = depth

        return dist_to_nearest_teammate, dist_to_nearest_opponent, positional_depth

    def update_edge_attributes(
        self,
        edge_index: torch.Tensor,
        positions: torch.Tensor,
        teammate_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Recompute edge attributes based on new positions.

        Args:
            edge_index: Edge connectivity [2, num_edges]
            positions: Player positions [N, 2]
            teammate_mask: Boolean mask for teammates [N]

        Returns:
            Edge attributes [num_edges, 3]: (distance, angle, same_team)
        """
        num_edges = edge_index.shape[1]
        device = positions.device

        edge_attr = torch.zeros(num_edges, 3, device=device)

        for e in range(num_edges):
            i, j = edge_index[0, e].item(), edge_index[1, e].item()

            # Distance (normalized)
            dist = torch.norm(positions[j] - positions[i])
            edge_attr[e, 0] = dist / self.pitch_length

            # Angle
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            angle = torch.atan2(dy, dx)
            edge_attr[e, 1] = angle / math.pi

            # Same team
            edge_attr[e, 2] = float(teammate_mask[i] == teammate_mask[j])

        return edge_attr
