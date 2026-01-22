"""Position constraints for tactical optimization."""

import torch
import torch.nn.functional as F
from typing import Optional


class PositionConstraints:
    """
    Constraint functions for position optimization.

    Enforces:
    1. Pitch bounds (players stay on field)
    2. Minimum spacing (players don't overlap)
    3. Movement limits (optional - max distance from original)
    """

    def __init__(
        self,
        pitch_length: float = 120.0,
        pitch_width: float = 80.0,
        min_spacing: float = 1.0,
        max_movement: Optional[float] = None,
        boundary_margin: float = 0.5,
    ):
        """
        Initialize constraint parameters.

        Args:
            pitch_length: StatsBomb pitch length (120m)
            pitch_width: StatsBomb pitch width (80m)
            min_spacing: Minimum distance between players in meters
            max_movement: Maximum distance a player can move from original position
            boundary_margin: Margin from pitch edge (keeps players inside)
        """
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.min_spacing = min_spacing
        self.max_movement = max_movement
        self.boundary_margin = boundary_margin

    def compute_penalty(
        self,
        positions: torch.Tensor,
        original_positions: torch.Tensor,
        all_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute total constraint penalty (differentiable).

        Args:
            positions: Current attacker positions [num_attackers, 2]
            original_positions: Starting positions for movement limits
            all_positions: All player positions [num_players, 2]

        Returns:
            Scalar penalty term to add to loss
        """
        penalty = torch.tensor(0.0, device=positions.device)

        # Pitch bounds penalty
        penalty = penalty + self.pitch_bounds_penalty(positions)

        # Spacing penalty (attackers vs all players)
        penalty = penalty + self.spacing_penalty(positions, all_positions)

        # Movement penalty (optional)
        if self.max_movement is not None:
            penalty = penalty + self.movement_penalty(positions, original_positions)

        return penalty

    def pitch_bounds_penalty(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Quadratic penalty for positions outside pitch.

        Args:
            positions: Player positions [N, 2]

        Returns:
            Scalar penalty
        """
        # X boundary violations
        x_low = F.relu(self.boundary_margin - positions[:, 0])
        x_high = F.relu(positions[:, 0] - (self.pitch_length - self.boundary_margin))

        # Y boundary violations
        y_low = F.relu(self.boundary_margin - positions[:, 1])
        y_high = F.relu(positions[:, 1] - (self.pitch_width - self.boundary_margin))

        return (x_low ** 2 + x_high ** 2 + y_low ** 2 + y_high ** 2).sum()

    def spacing_penalty(
        self,
        attacker_positions: torch.Tensor,
        all_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Penalty for players closer than min_spacing.

        Args:
            attacker_positions: Attacker positions [A, 2]
            all_positions: All player positions [N, 2]

        Returns:
            Scalar penalty
        """
        # Compute pairwise distances from attackers to all players
        diff = attacker_positions.unsqueeze(1) - all_positions.unsqueeze(0)  # [A, N, 2]
        distances = torch.norm(diff, dim=2)  # [A, N]

        # Penalty for distances below threshold (excluding self-distances ~0)
        violations = F.relu(self.min_spacing - distances)
        # Mask out self-distances (very small values)
        mask = distances > 0.01
        violations = violations * mask.float()

        return (violations ** 2).sum()

    def movement_penalty(
        self,
        positions: torch.Tensor,
        original_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Penalty for moving too far from original position.

        Args:
            positions: Current positions [N, 2]
            original_positions: Original positions [N, 2]

        Returns:
            Scalar penalty
        """
        if self.max_movement is None:
            return torch.tensor(0.0, device=positions.device)

        movement = torch.norm(positions - original_positions, dim=1)
        violations = F.relu(movement - self.max_movement)
        return (violations ** 2).sum()

    def project_to_feasible(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Hard projection: clamp positions to pitch bounds.

        Applied after gradient step to ensure valid positions.

        Args:
            positions: Player positions [N, 2]

        Returns:
            Clamped positions [N, 2]
        """
        positions = positions.clone()
        positions[:, 0] = torch.clamp(
            positions[:, 0],
            self.boundary_margin,
            self.pitch_length - self.boundary_margin
        )
        positions[:, 1] = torch.clamp(
            positions[:, 1],
            self.boundary_margin,
            self.pitch_width - self.boundary_margin
        )
        return positions
