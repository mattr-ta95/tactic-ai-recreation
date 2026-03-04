"""Tactical position optimizer for corner kicks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import copy

from .constraints import PositionConstraints
from .feature_recompute import FeatureRecomputer


@dataclass
class OptimizationResult:
    """Container for optimization results."""
    original_graph: Data
    optimized_graph: Data
    target_receiver: int
    original_probability: float
    optimized_probability: float
    position_changes: Dict[int, Tuple[float, float, float, float]]  # idx -> (orig_x, orig_y, new_x, new_y)
    optimization_history: List[float]
    probability_history: List[float]
    num_iterations: int
    converged: bool

    @property
    def improvement_percentage(self) -> float:
        """Relative improvement in receiver probability."""
        if self.original_probability > 0:
            return (self.optimized_probability - self.original_probability) / self.original_probability * 100
        return 0.0

    @property
    def absolute_improvement(self) -> float:
        """Absolute improvement in probability."""
        return self.optimized_probability - self.original_probability


class TacticalOptimizer:
    """
    Gradient-based optimizer for attacking player positions.

    Uses the trained GNN to backpropagate through receiver predictions
    and optimize attacker positions to maximize probability for a target receiver.
    """

    def __init__(
        self,
        model: nn.Module,
        pitch_length: float = 120.0,
        pitch_width: float = 80.0,
        distance_threshold: float = 5.0,
        device: str = 'cpu',
    ):
        """
        Initialize tactical optimizer.

        Args:
            model: Trained GATCornerNet or similar model
            pitch_length: StatsBomb pitch length
            pitch_width: StatsBomb pitch width
            distance_threshold: Edge distance threshold (meters)
            device: Torch device
        """
        self.model = model
        self.model.eval()
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.distance_threshold = distance_threshold
        self.device = device

        # Initialize helpers
        self.constraints = PositionConstraints(
            pitch_length=pitch_length,
            pitch_width=pitch_width,
        )
        self.feature_recomputer = FeatureRecomputer(
            pitch_length=pitch_length,
            pitch_width=pitch_width,
        )

    def optimize_positions(
        self,
        graph: Data,
        target_receiver: Optional[int] = None,
        num_iterations: int = 50,
        learning_rate: float = 0.1,
        constraint_penalty: float = 10.0,
        min_spacing: float = 1.0,
        max_movement: Optional[float] = None,
        convergence_threshold: float = 1e-4,
        verbose: bool = False,
    ) -> OptimizationResult:
        """
        Optimize attacker positions to maximize receiver probability.

        Args:
            graph: PyG Data object representing corner kick
            target_receiver: Index of target receiver (None = maximize best attacker)
            num_iterations: Optimization iterations
            learning_rate: Step size for position updates
            constraint_penalty: Penalty weight for constraint violations
            min_spacing: Minimum distance between players (meters)
            max_movement: Maximum movement from original position (None = unlimited)
            convergence_threshold: Stop if loss change is below this
            verbose: Print progress during optimization

        Returns:
            OptimizationResult with original/optimized graphs and metrics
        """
        # Clone graph to avoid modifying original
        graph = graph.clone()
        graph = graph.to(self.device)

        # Identify attackers (is_teammate = 1)
        attacker_mask = self._identify_attackers(graph)
        attacker_indices = attacker_mask.nonzero().squeeze(-1)
        num_attackers = attacker_indices.shape[0]

        if num_attackers == 0:
            raise ValueError("No attackers found in graph")

        # Extract positions (denormalize from [0,1] to meters)
        all_positions = graph.x[:, :2].clone()
        all_positions[:, 0] = all_positions[:, 0] * self.pitch_length
        all_positions[:, 1] = all_positions[:, 1] * self.pitch_width

        # Create learnable parameters for attacker positions only
        attacker_positions = all_positions[attacker_indices].clone().detach()
        attacker_positions.requires_grad_(True)

        # Store original positions
        original_positions = attacker_positions.clone().detach()
        original_all_positions = all_positions.clone().detach()

        # Update constraints with min_spacing and max_movement
        self.constraints.min_spacing = min_spacing
        self.constraints.max_movement = max_movement

        # Determine target receiver
        if target_receiver is None:
            # Find best attacker based on current predictions
            with torch.no_grad():
                probs = self._compute_probabilities(graph)
                attacker_probs = probs[attacker_mask]
                best_attacker_local = attacker_probs.argmax().item()
                target_receiver = attacker_indices[best_attacker_local].item()

        # Compute original probability
        with torch.no_grad():
            original_probs = self._compute_probabilities(graph)
            original_probability = original_probs[target_receiver].item()

        # Set up optimizer
        optimizer = torch.optim.Adam([attacker_positions], lr=learning_rate)

        # Optimization loop
        loss_history = []
        prob_history = [original_probability]
        best_loss = float('inf')
        best_positions = attacker_positions.clone().detach()
        converged = False

        for iteration in range(num_iterations):
            optimizer.zero_grad()

            # Reconstruct full position tensor
            current_positions = all_positions.clone()
            current_positions[attacker_indices] = attacker_positions

            # Update graph features with new positions
            updated_x = self.feature_recomputer.recompute_all(
                graph.x, current_positions
            )

            # Recompute edge attributes
            teammate_mask = graph.x[:, 2] > 0.5
            updated_edge_attr = self.feature_recomputer.update_edge_attributes(
                graph.edge_index, current_positions, teammate_mask
            )

            # Forward pass
            batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=self.device)
            logits = self.model(
                updated_x,
                graph.edge_index,
                batch,
                edge_attr=updated_edge_attr if hasattr(graph, 'edge_attr') else None
            )
            if isinstance(logits, dict):
                logits = logits['receiver']

            # Compute objective (negative log probability of target)
            probs = F.softmax(logits, dim=0)
            target_prob = probs[target_receiver]
            objective = -torch.log(target_prob + 1e-8)

            # Add constraint penalties
            penalty = self.constraints.compute_penalty(
                attacker_positions,
                original_positions,
                current_positions
            )

            loss = objective + constraint_penalty * penalty

            # Backward pass
            loss.backward()

            # Gradient step
            optimizer.step()

            # Project to feasible region
            with torch.no_grad():
                attacker_positions.data = self.constraints.project_to_feasible(
                    attacker_positions.data
                )

            # Track progress
            loss_val = loss.item()
            prob_val = target_prob.item()
            loss_history.append(loss_val)
            prob_history.append(prob_val)

            if loss_val < best_loss:
                best_loss = loss_val
                best_positions = attacker_positions.clone().detach()

            # Check convergence
            if len(loss_history) > 1 and abs(loss_history[-1] - loss_history[-2]) < convergence_threshold:
                converged = True
                if verbose:
                    print(f"Converged at iteration {iteration + 1}")
                break

            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{num_iterations}: "
                      f"Loss={loss_val:.4f}, P(target)={prob_val:.4f}")

        # Build optimized graph
        optimized_graph = self._build_optimized_graph(
            graph, best_positions, attacker_indices
        )

        # Compute final probability
        with torch.no_grad():
            optimized_probs = self._compute_probabilities(optimized_graph)
            optimized_probability = optimized_probs[target_receiver].item()

        # Compute position changes
        position_changes = {}
        for i, global_idx in enumerate(attacker_indices.tolist()):
            orig_x = original_positions[i, 0].item()
            orig_y = original_positions[i, 1].item()
            new_x = best_positions[i, 0].item()
            new_y = best_positions[i, 1].item()

            # Only include if position actually changed
            if abs(orig_x - new_x) > 0.1 or abs(orig_y - new_y) > 0.1:
                position_changes[global_idx] = (orig_x, orig_y, new_x, new_y)

        return OptimizationResult(
            original_graph=graph,
            optimized_graph=optimized_graph,
            target_receiver=target_receiver,
            original_probability=original_probability,
            optimized_probability=optimized_probability,
            position_changes=position_changes,
            optimization_history=loss_history,
            probability_history=prob_history,
            num_iterations=len(loss_history),
            converged=converged,
        )

    def _identify_attackers(self, graph: Data) -> torch.Tensor:
        """Return boolean mask for attacking players (teammate=1)."""
        return graph.x[:, 2] > 0.5

    def _compute_probabilities(self, graph: Data) -> torch.Tensor:
        """Compute receiver probabilities for all players."""
        batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=self.device)
        edge_attr = graph.edge_attr if hasattr(graph, 'edge_attr') else None

        logits = self.model(graph.x, graph.edge_index, batch, edge_attr=edge_attr)
        if isinstance(logits, dict):
            logits = logits['receiver']
        return F.softmax(logits, dim=0)

    def _build_optimized_graph(
        self,
        original_graph: Data,
        optimized_positions: torch.Tensor,
        attacker_indices: torch.Tensor,
    ) -> Data:
        """Build graph with optimized positions."""
        graph = original_graph.clone()

        # Get all positions
        all_positions = graph.x[:, :2].clone()
        all_positions[:, 0] = all_positions[:, 0] * self.pitch_length
        all_positions[:, 1] = all_positions[:, 1] * self.pitch_width

        # Update attacker positions
        all_positions[attacker_indices] = optimized_positions

        # Recompute features
        graph.x = self.feature_recomputer.recompute_all(graph.x, all_positions)

        # Recompute edge attributes
        teammate_mask = original_graph.x[:, 2] > 0.5
        graph.edge_attr = self.feature_recomputer.update_edge_attributes(
            graph.edge_index, all_positions, teammate_mask
        )

        return graph

    def analyze_sensitivity(
        self,
        graph: Data,
        target_receiver: int,
        num_samples: int = 100,
        perturbation_std: float = 2.0,
    ) -> Dict[int, float]:
        """
        Analyze which attackers have most impact on target's probability.

        Args:
            graph: Input graph
            target_receiver: Target receiver index
            num_samples: Number of random perturbations per player
            perturbation_std: Standard deviation of position perturbations (meters)

        Returns:
            Dict mapping attacker index to sensitivity score
        """
        graph = graph.clone().to(self.device)
        attacker_mask = self._identify_attackers(graph)
        attacker_indices = attacker_mask.nonzero().squeeze(-1)

        # Baseline probability
        with torch.no_grad():
            base_prob = self._compute_probabilities(graph)[target_receiver].item()

        sensitivity = {}

        for attacker_idx in attacker_indices.tolist():
            prob_changes = []

            for _ in range(num_samples):
                # Perturb this attacker's position
                perturbed_graph = graph.clone()
                perturbation = torch.randn(2, device=self.device) * perturbation_std

                # Get current position in meters
                current_x = perturbed_graph.x[attacker_idx, 0].item() * self.pitch_length
                current_y = perturbed_graph.x[attacker_idx, 1].item() * self.pitch_width

                # Apply perturbation with bounds
                new_x = max(0.5, min(119.5, current_x + perturbation[0].item()))
                new_y = max(0.5, min(79.5, current_y + perturbation[1].item()))

                # Update position
                perturbed_graph.x[attacker_idx, 0] = new_x / self.pitch_length
                perturbed_graph.x[attacker_idx, 1] = new_y / self.pitch_width

                # Compute new probability
                with torch.no_grad():
                    new_prob = self._compute_probabilities(perturbed_graph)[target_receiver].item()

                prob_changes.append(abs(new_prob - base_prob))

            # Sensitivity = average absolute change in probability
            sensitivity[attacker_idx] = sum(prob_changes) / len(prob_changes)

        return sensitivity
