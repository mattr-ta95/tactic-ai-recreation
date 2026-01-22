"""Visualization tools for tactical optimization results."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mplsoccer import Pitch
from typing import Optional, Tuple
import torch

from .optimizer import OptimizationResult


class TacticalVisualizer:
    """
    Visualization tools for position optimization results.

    Creates mplsoccer pitch plots showing:
    - Original vs optimized positions
    - Movement arrows
    - Probability changes
    """

    def __init__(
        self,
        pitch_type: str = 'statsbomb',
        pitch_color: str = '#22312b',
        line_color: str = '#c7d5cc',
        attacker_color: str = '#FF6B6B',
        defender_color: str = '#4ECDC4',
        target_color: str = '#FFD93D',
        arrow_color: str = '#FFFFFF',
    ):
        """
        Initialize visualization colors.

        Args:
            pitch_type: mplsoccer pitch type
            pitch_color: Background color
            line_color: Line color
            attacker_color: Color for attacking players
            defender_color: Color for defending players
            target_color: Color for target receiver
            arrow_color: Color for movement arrows
        """
        self.pitch_type = pitch_type
        self.pitch_color = pitch_color
        self.line_color = line_color
        self.attacker_color = attacker_color
        self.defender_color = defender_color
        self.target_color = target_color
        self.arrow_color = arrow_color

    def plot_optimization_result(
        self,
        result: OptimizationResult,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show_probabilities: bool = True,
        show_arrows: bool = True,
        figsize: Tuple[int, int] = (16, 8),
    ) -> plt.Figure:
        """
        Create side-by-side comparison of original vs optimized positions.

        Args:
            result: OptimizationResult from optimizer
            title: Optional title for the figure
            save_path: Path to save figure (None = don't save)
            show_probabilities: Show probability annotations
            show_arrows: Show movement arrows on optimized plot
            figsize: Figure size

        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        pitch = Pitch(
            pitch_type=self.pitch_type,
            pitch_color=self.pitch_color,
            line_color=self.line_color,
            goal_type='box',
        )

        # Original setup (left)
        pitch.draw(ax=ax1)
        self._plot_players(
            ax1, pitch, result.original_graph,
            target_receiver=result.target_receiver,
            show_probabilities=show_probabilities,
            title=f'Original Setup\nP(target receives) = {result.original_probability:.1%}'
        )

        # Optimized setup (right)
        pitch.draw(ax=ax2)
        self._plot_players(
            ax2, pitch, result.optimized_graph,
            target_receiver=result.target_receiver,
            show_probabilities=show_probabilities,
            title=f'Optimized Setup\nP(target receives) = {result.optimized_probability:.1%}'
        )

        # Draw movement arrows
        if show_arrows and result.position_changes:
            for player_idx, (ox, oy, nx, ny) in result.position_changes.items():
                # Draw arrow on optimized plot
                ax2.annotate(
                    '',
                    xy=(nx, ny),
                    xytext=(ox, oy),
                    arrowprops=dict(
                        arrowstyle='->',
                        color=self.arrow_color,
                        lw=2,
                        alpha=0.8,
                    ),
                    zorder=5
                )

        # Overall title
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        else:
            improvement = result.improvement_percentage
            fig.suptitle(
                f'Position Optimization Result: +{improvement:.1f}% improvement',
                fontsize=14,
                fontweight='bold'
            )

        # Add legend
        legend_elements = [
            mpatches.Patch(color=self.attacker_color, label='Attackers'),
            mpatches.Patch(color=self.defender_color, label='Defenders'),
            mpatches.Patch(color=self.target_color, label='Target Receiver'),
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Visualization saved to: {save_path}")

        return fig

    def _plot_players(
        self,
        ax,
        pitch: Pitch,
        graph,
        target_receiver: int,
        show_probabilities: bool = False,
        title: str = '',
    ):
        """Plot players on a pitch."""
        # Extract positions and teams
        x_positions = graph.x[:, 0].cpu().numpy() * 120  # Denormalize
        y_positions = graph.x[:, 1].cpu().numpy() * 80
        is_teammate = graph.x[:, 2].cpu().numpy() > 0.5

        # Plot defenders (opponents)
        defender_x = x_positions[~is_teammate]
        defender_y = y_positions[~is_teammate]
        pitch.scatter(
            defender_x, defender_y,
            ax=ax,
            s=300,
            c=self.defender_color,
            edgecolors='white',
            linewidth=2,
            zorder=3,
        )

        # Plot attackers (teammates)
        attacker_mask = is_teammate.copy()
        attacker_mask[target_receiver] = False  # Exclude target

        attacker_x = x_positions[attacker_mask]
        attacker_y = y_positions[attacker_mask]
        pitch.scatter(
            attacker_x, attacker_y,
            ax=ax,
            s=300,
            c=self.attacker_color,
            edgecolors='white',
            linewidth=2,
            zorder=3,
        )

        # Plot target receiver (highlighted)
        target_x = x_positions[target_receiver]
        target_y = y_positions[target_receiver]
        pitch.scatter(
            [target_x], [target_y],
            ax=ax,
            s=400,
            c=self.target_color,
            edgecolors='white',
            linewidth=3,
            marker='*',
            zorder=4,
        )

        # Add player indices
        for i in range(len(x_positions)):
            color = 'white' if is_teammate[i] else 'black'
            ax.annotate(
                str(i),
                (x_positions[i], y_positions[i]),
                ha='center',
                va='center',
                fontsize=8,
                fontweight='bold',
                color=color,
                zorder=6,
            )

        ax.set_title(title, fontsize=11)

    def plot_optimization_trajectory(
        self,
        result: OptimizationResult,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 4),
    ) -> plt.Figure:
        """
        Plot loss curve and probability improvement over iterations.

        Args:
            result: OptimizationResult from optimizer
            save_path: Path to save figure
            figsize: Figure size

        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        iterations = range(len(result.optimization_history))

        # Loss curve
        ax1.plot(iterations, result.optimization_history, 'b-', linewidth=2)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.set_title('Optimization Loss')
        ax1.grid(True, alpha=0.3)

        # Probability curve
        ax2.plot(range(len(result.probability_history)), result.probability_history, 'g-', linewidth=2)
        ax2.axhline(y=result.original_probability, color='r', linestyle='--', label='Original')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('P(target receives)')
        ax2.set_title('Target Receiver Probability')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_single_setup(
        self,
        graph,
        title: str = 'Corner Kick Setup',
        target_receiver: Optional[int] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
    ) -> plt.Figure:
        """
        Plot a single corner kick setup.

        Args:
            graph: PyG Data object
            title: Plot title
            target_receiver: Index of target receiver to highlight
            save_path: Path to save figure
            figsize: Figure size

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        pitch = Pitch(
            pitch_type=self.pitch_type,
            pitch_color=self.pitch_color,
            line_color=self.line_color,
            goal_type='box',
        )
        pitch.draw(ax=ax)

        # Extract positions and teams
        x_positions = graph.x[:, 0].cpu().numpy() * 120
        y_positions = graph.x[:, 1].cpu().numpy() * 80
        is_teammate = graph.x[:, 2].cpu().numpy() > 0.5

        # Plot defenders
        defender_mask = ~is_teammate
        pitch.scatter(
            x_positions[defender_mask],
            y_positions[defender_mask],
            ax=ax,
            s=300,
            c=self.defender_color,
            edgecolors='white',
            linewidth=2,
            zorder=3,
            label='Defenders',
        )

        # Plot attackers
        if target_receiver is not None:
            attacker_mask = is_teammate.copy()
            attacker_mask[target_receiver] = False
        else:
            attacker_mask = is_teammate

        pitch.scatter(
            x_positions[attacker_mask],
            y_positions[attacker_mask],
            ax=ax,
            s=300,
            c=self.attacker_color,
            edgecolors='white',
            linewidth=2,
            zorder=3,
            label='Attackers',
        )

        # Plot target receiver if specified
        if target_receiver is not None:
            pitch.scatter(
                [x_positions[target_receiver]],
                [y_positions[target_receiver]],
                ax=ax,
                s=400,
                c=self.target_color,
                edgecolors='white',
                linewidth=3,
                marker='*',
                zorder=4,
                label='Target',
            )

        # Add player indices
        for i in range(len(x_positions)):
            ax.annotate(
                str(i),
                (x_positions[i], y_positions[i]),
                ha='center',
                va='center',
                fontsize=8,
                fontweight='bold',
                color='white' if is_teammate[i] else 'black',
                zorder=6,
            )

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')

        return fig
