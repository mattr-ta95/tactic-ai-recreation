#!/usr/bin/env python3
"""
Tactical Analysis CLI for TacticAI Position Optimization.

Optimizes attacking player positions to maximize receiver probability.

Usage:
    # Optimize positions for a specific corner (by index in dataset)
    python scripts/tactical_analysis.py --input 42 --target-receiver 5

    # Optimize with custom settings
    python scripts/tactical_analysis.py --input 42 --iterations 100 --lr 0.05

    # Auto-select best target receiver
    python scripts/tactical_analysis.py --input 42

    # Load from specific checkpoint
    python scripts/tactical_analysis.py --input 42 --checkpoint models/checkpoints/best_model.pth

    # List available corners
    python scripts/tactical_analysis.py --list
"""

import argparse
import torch
import pandas as pd
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tactical.optimizer import TacticalOptimizer
from tactical.visualization import TacticalVisualizer
from models.gnn import get_model
from data.processor import CornerKickProcessor


def load_model(checkpoint_path: str, device: str = 'cpu'):
    """Load trained model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract config from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Default config
        config = {
            'model_type': 'gat',
            'node_features': 14,
            'hidden_dim': 128,
            'num_layers': 4,
            'dropout': 0.2,
        }

    # Build model
    model_kwargs = {
        'node_features': config.get('node_features', 14),
        'hidden_dim': config.get('hidden_dim', 128),
        'num_layers': config.get('num_layers', 4),
        'dropout': config.get('dropout', 0.2),
    }

    model = get_model(config.get('model_type', 'gat'), **model_kwargs)

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print(f"  Model type: {config.get('model_type', 'gat')}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, config


def load_dataset(processor: CornerKickProcessor, data_dir: str = 'data/processed'):
    """Load corner kick dataset."""
    # Try different data files
    for filename in ['training_shots_combined.pkl', 'training_shots.pkl', 'shots_freeze.pkl']:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            print(f"Loading data from: {filepath}")
            corners = pd.read_pickle(filepath)
            break
    else:
        raise FileNotFoundError(f"No data found in {data_dir}")

    # Convert to graphs
    print(f"Converting {len(corners)} corners to graphs...")
    dataset = processor.create_dataset(corners)

    # Filter for labeled examples
    dataset = [g for g in dataset if hasattr(g, 'y') and g.y is not None]
    print(f"  {len(dataset)} labeled graphs available")

    return dataset


def run_single_analysis(optimizer, visualizer, graph, args):
    """Run optimization on a single corner."""
    print("\n" + "=" * 70)
    print("TacticAI Tactical Analysis")
    print("=" * 70)

    # Run optimization
    print(f"\nOptimizing attacker positions...")
    print(f"  Iterations: {args.iterations}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Target receiver: {args.target_receiver or 'auto-select'}")

    result = optimizer.optimize_positions(
        graph,
        target_receiver=args.target_receiver,
        num_iterations=args.iterations,
        learning_rate=args.lr,
        constraint_penalty=args.constraint_penalty,
        min_spacing=args.min_spacing,
        verbose=args.verbose,
    )

    # Print results
    print(f"\n{'=' * 70}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"Target Receiver: Player {result.target_receiver}")
    print(f"\nProbability Change:")
    print(f"  Original:  {result.original_probability:.1%}")
    print(f"  Optimized: {result.optimized_probability:.1%}")
    print(f"  Change:    {result.absolute_improvement:+.1%} ({result.improvement_percentage:+.1f}%)")

    if result.position_changes:
        print(f"\nPosition Changes ({len(result.position_changes)} players moved):")
        for player_idx, (ox, oy, nx, ny) in result.position_changes.items():
            dx = nx - ox
            dy = ny - oy
            dist = (dx ** 2 + dy ** 2) ** 0.5
            print(f"  Player {player_idx}: ({ox:.1f}, {oy:.1f}) → ({nx:.1f}, {ny:.1f}) [{dist:.1f}m]")
    else:
        print("\nNo significant position changes (already optimal)")

    print(f"\nIterations: {result.num_iterations}")
    print(f"Converged: {result.converged}")

    # Generate visualization
    if not args.no_visualize:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Main comparison plot
        save_path = output_dir / f"corner_{args.input}_optimized.png"
        visualizer.plot_optimization_result(result, save_path=str(save_path))

        # Trajectory plot
        traj_path = output_dir / f"corner_{args.input}_trajectory.png"
        visualizer.plot_optimization_trajectory(result, save_path=str(traj_path))

    return result


def list_corners(dataset):
    """List available corners in dataset."""
    print("\n" + "=" * 70)
    print("Available Corners")
    print("=" * 70)

    for i, graph in enumerate(dataset[:20]):  # Show first 20
        num_players = graph.num_nodes
        num_attackers = (graph.x[:, 2] > 0.5).sum().item()
        num_defenders = num_players - num_attackers

        match_id = getattr(graph, 'match_id', 'N/A')
        print(f"  [{i:4d}] Players: {num_players} ({num_attackers}A / {num_defenders}D), Match: {match_id}")

    if len(dataset) > 20:
        print(f"  ... and {len(dataset) - 20} more corners")

    print(f"\nTotal: {len(dataset)} corners")


def main():
    parser = argparse.ArgumentParser(
        description='TacticAI Tactical Analysis - Position Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/tactical_analysis.py --input 0 --target-receiver 5
    python scripts/tactical_analysis.py --input 42 --iterations 100 --verbose
    python scripts/tactical_analysis.py --list
        """
    )

    # Input options
    parser.add_argument('--input', type=int, help='Corner index in dataset')
    parser.add_argument('--list', action='store_true', help='List available corners')

    # Optimization options
    parser.add_argument('--target-receiver', type=int, default=None,
                        help='Target receiver player index (auto-select if not specified)')
    parser.add_argument('--iterations', type=int, default=50, help='Optimization iterations')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--constraint-penalty', type=float, default=10.0,
                        help='Constraint penalty weight')
    parser.add_argument('--min-spacing', type=float, default=1.0,
                        help='Minimum player spacing in meters')

    # Model options
    parser.add_argument('--checkpoint', type=str, default='models/checkpoints/best_model.pth',
                        help='Model checkpoint path')

    # Output options
    parser.add_argument('--output-dir', type=str, default='visualizations/tactical',
                        help='Directory for visualization output')
    parser.add_argument('--no-visualize', action='store_true',
                        help='Skip visualization generation')
    parser.add_argument('--verbose', action='store_true', help='Print optimization progress')

    args = parser.parse_args()

    # Detect device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device}")

    # Load model
    model, config = load_model(args.checkpoint, device)

    # Initialize processor
    processor = CornerKickProcessor(
        distance_threshold=config.get('distance_threshold', 5.0),
        normalize_positions=True,
        use_enhanced_features=config.get('use_enhanced_features', True),
        use_role_features=config.get('use_role_features', True),
        use_positional_context=config.get('use_positional_context', True),
    )

    # Load dataset
    dataset = load_dataset(processor)

    # Handle --list flag
    if args.list:
        list_corners(dataset)
        return

    # Validate input
    if args.input is None:
        parser.error("--input is required (use --list to see available corners)")

    if args.input < 0 or args.input >= len(dataset):
        parser.error(f"Invalid corner index: {args.input}. Valid range: 0-{len(dataset)-1}")

    # Initialize optimizer and visualizer
    optimizer = TacticalOptimizer(model, device=device)
    visualizer = TacticalVisualizer()

    # Get graph
    graph = dataset[args.input].to(device)

    # Validate target receiver
    if args.target_receiver is not None:
        attacker_mask = graph.x[:, 2] > 0.5
        if args.target_receiver >= graph.num_nodes:
            parser.error(f"Invalid target receiver: {args.target_receiver}. Max: {graph.num_nodes - 1}")
        if not attacker_mask[args.target_receiver]:
            print(f"Warning: Player {args.target_receiver} is a defender, not an attacker")

    # Run analysis
    run_single_analysis(optimizer, visualizer, graph, args)


if __name__ == '__main__':
    main()
