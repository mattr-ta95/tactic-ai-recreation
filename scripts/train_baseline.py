#!/usr/bin/env python3
"""
Train baseline TacticAI model
Phase 1: Simple receiver prediction
"""

import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Add src to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.gnn import SimpleCornerGNN, get_model
from data.processor import CornerKickProcessor, get_data_statistics, augment_graph


class FocalLoss(torch.nn.Module):
    """
    Focal Loss for handling class imbalance in receiver prediction.

    Focal loss down-weights easy examples, focusing training on hard examples.
    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter (default 2.0). Higher = more focus on hard examples.
        alpha: Optional class weights
        label_smoothing: Label smoothing factor (default 0.0)
    """

    def __init__(self, gamma: float = 2.0, alpha=None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits of shape [batch_size, num_classes]
            targets: Target labels of shape [batch_size]

        Returns:
            Focal loss scalar
        """
        ce_loss = F.cross_entropy(
            inputs, targets,
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()


def train_epoch(model, loader, optimizer, device, label_smoothing=0.0,
                use_focal_loss=False, focal_gamma=2.0, use_edge_features=False):
    """Train for one epoch

    Args:
        model: The GNN model
        loader: DataLoader for training data
        optimizer: Optimizer
        device: Device to use
        label_smoothing: Label smoothing factor (0.0 = no smoothing)
        use_focal_loss: Whether to use focal loss instead of cross entropy
        focal_gamma: Gamma parameter for focal loss
        use_edge_features: Whether to pass edge_attr to model
    """
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    num_batches = 0

    # Initialize focal loss if enabled
    if use_focal_loss:
        focal_loss_fn = FocalLoss(gamma=focal_gamma, label_smoothing=label_smoothing)

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass with optional edge features
        if use_edge_features and hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
            out = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr)
        else:
            out = model(batch.x, batch.edge_index, batch.batch)

        # Use real receiver labels
        # batch.y contains receiver index for each graph
        # We need to create a mask for nodes that have labels
        if not hasattr(batch, 'y') or batch.y is None:
            continue

        # Get predictions per graph
        loss = 0
        correct = 0
        num_graphs = batch.num_graphs

        for i in range(num_graphs):
            # Get nodes for this graph
            mask = (batch.batch == i)
            graph_logits = out[mask]  # [num_nodes_in_graph]
            graph_label = batch.y[i] if batch.y.dim() > 0 else batch.y

            # Convert label to tensor if needed
            if isinstance(graph_label, torch.Tensor):
                label_tensor = graph_label.unsqueeze(0)
                label_val = graph_label.item()
            else:
                label_tensor = torch.tensor([int(graph_label)], device=graph_logits.device)
                label_val = int(graph_label)

            # Ensure label is within valid range
            if label_val >= len(graph_logits):
                continue

            # Compute loss (focal or cross entropy with label smoothing)
            if use_focal_loss:
                loss += focal_loss_fn(graph_logits.unsqueeze(0), label_tensor)
            else:
                loss += F.cross_entropy(
                    graph_logits.unsqueeze(0),
                    label_tensor,
                    label_smoothing=label_smoothing
                )

            # Accuracy
            pred = graph_logits.argmax().item()
            correct += 1.0 if (pred == label_val) else 0.0

        if num_graphs > 0:
            loss = loss / num_graphs
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            total_correct += correct
            total_samples += num_graphs
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    return avg_loss, accuracy


def evaluate(model, loader, device, use_edge_features=False):
    """Evaluate model with accuracy and top-k metrics

    Args:
        model: The GNN model
        loader: DataLoader for evaluation
        device: Device to use
        use_edge_features: Whether to pass edge_attr to model
    """
    model.eval()
    total_loss = 0
    total_correct = 0
    total_top3 = 0
    total_top5 = 0
    total_samples = 0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            # Forward pass with optional edge features
            if use_edge_features and hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
                out = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr)
            else:
                out = model(batch.x, batch.edge_index, batch.batch)

            if not hasattr(batch, 'y') or batch.y is None:
                continue

            loss = 0
            correct = 0
            top3_correct = 0
            top5_correct = 0
            num_graphs = batch.num_graphs

            for i in range(num_graphs):
                # Get nodes for this graph
                mask = (batch.batch == i)
                graph_logits = out[mask]
                graph_label = batch.y[i] if batch.y.dim() > 0 else batch.y

                if graph_label >= len(graph_logits):
                    continue

                # Loss
                loss += F.cross_entropy(graph_logits.unsqueeze(0), graph_label.unsqueeze(0))

                # Convert label to scalar for comparison
                if isinstance(graph_label, torch.Tensor):
                    label_val = graph_label.item()
                else:
                    label_val = int(graph_label)

                # Top-1 accuracy
                pred = graph_logits.argmax().item()
                correct += 1.0 if (pred == label_val) else 0.0

                # Top-3 accuracy
                if len(graph_logits) >= 3:
                    top3_preds = graph_logits.topk(min(3, len(graph_logits))).indices.cpu().tolist()
                    top3_correct += 1.0 if (label_val in top3_preds) else 0.0
                else:
                    top3_correct += 1.0 if (pred == label_val) else 0.0

                # Top-5 accuracy
                if len(graph_logits) >= 5:
                    top5_preds = graph_logits.topk(min(5, len(graph_logits))).indices.cpu().tolist()
                    top5_correct += 1.0 if (label_val in top5_preds) else 0.0
                else:
                    top5_correct += 1.0 if (pred == label_val) else 0.0

            if num_graphs > 0:
                total_loss += (loss / num_graphs).item()
                total_correct += correct
                total_top3 += top3_correct
                total_top5 += top5_correct
                total_samples += num_graphs
                num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    top3_acc = total_top3 / total_samples if total_samples > 0 else 0
    top5_acc = total_top5 / total_samples if total_samples > 0 else 0

    return avg_loss, accuracy, top3_acc, top5_acc


def main():
    print("=" * 70)
    print("TacticAI Baseline Training")
    print("=" * 70)
    
    # Configuration
    # Detect best available device (CUDA > MPS > CPU)
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    config = {
        'model_type': 'gat',
        # Feature dimensions: 7 (enhanced) + 4 (roles) + 3 (positional) = 14
        'node_features': 14,  # All node features
        'edge_dim': 3,
        'hidden_dim': 128,
        'num_layers': 4,
        'dropout': 0.2,
        'learning_rate': 0.0005,
        'batch_size': 32,
        'num_epochs': 50,
        'distance_threshold': 5.0,
        'early_stopping_patience': 10,
        'early_stopping_min_delta': 0.001,
        'weight_decay': 1e-5,
        # Feature flags - testing incremental improvements
        'use_enhanced_features': True,
        'use_role_features': True,  # Add GK/DEF/MID/FWD
        'use_positional_context': True,  # Re-enabled with role features
        'use_edge_features': False,  # Disabled for now
        'use_scheduler': True,
        'use_knn_edges': False,
        'knn_k': 8,
        'device': device,
        # Loss and regularization
        'label_smoothing': 0.1,
        'use_augmentation': True,
        'use_validation_split': True,
        'use_focal_loss': True,
        'focal_gamma': 2.0,
    }
    
    # Test with larger model to handle enhanced features
    # config['hidden_dim'] = 256  # Uncomment to test larger model
    # config['num_layers'] = 5    # Uncomment to test deeper model
    
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    device = torch.device(config['device'])
    print(f"\nUsing device: {device}")
    
    # Load data
    print("\n1. Loading data...")

    # Try to load prepared training data first (shots linked to corners with labels)
    combined_path = 'data/processed/training_shots_combined.pkl'  # Real + synthetic
    training_path = 'data/processed/training_shots.pkl'
    linked_path = 'data/processed/shots_linked_to_corners.pkl'
    shots_path = 'data/processed/shots_freeze.pkl'
    corners_path = 'data/processed/corners.pkl'

    if os.path.exists(combined_path):
        print(f"   Loading combined (real + synthetic) data from {combined_path}...")
        corners = pd.read_pickle(combined_path)
        if 'is_synthetic' in corners.columns:
            real_count = len(corners[corners['is_synthetic'] == False])
            synth_count = len(corners[corners['is_synthetic'] == True])
            print(f"   Loaded {len(corners)} examples (real: {real_count}, synthetic: {synth_count})")
        else:
            print(f"   Loaded {len(corners)} examples")
        data_type = "combined"
    elif os.path.exists(training_path):
        print(f"   Loading prepared training data from {training_path}...")
        corners = pd.read_pickle(training_path)
        print(f"   Loaded {len(corners)} labeled examples (shots linked to corners)")
        data_type = "training_shots"
    elif os.path.exists(linked_path):
        print(f"   Loading linked shots from {linked_path}...")
        corners = pd.read_pickle(linked_path)
        # Filter for usable examples
        corners = corners[
            (corners['is_from_corner'] == True) &
            (corners['corner_pass_recipient_id'].notna())
        ].copy()
        print(f"   Loaded {len(corners)} usable examples")
        data_type = "linked_shots"
    elif os.path.exists(shots_path):
        print(f"   ⚠️  Using shots without corner linking (no receiver labels)")
        print(f"   Run: python scripts/prepare_training_data.py for better labels")
        corners = pd.read_pickle(shots_path)
        print(f"   Loaded {len(corners)} shots with freeze frames")
        data_type = "shots"
    elif os.path.exists(corners_path):
        corners = pd.read_pickle(corners_path)
        if 'freeze_frame_parsed' not in corners.columns or corners['freeze_frame_parsed'].isna().all():
            print(f"\n❌ Corners don't have freeze frame data.")
            print("   Please run: python scripts/prepare_training_data.py")
            return
        else:
            print(f"   Loaded {len(corners)} corners")
            data_type = "corners"
    else:
        print(f"❌ No data files found")
        print("   Please run:")
        print("     1. python scripts/download_data.py")
        print("     2. python scripts/prepare_training_data.py")
        return

    print(f"   Using {data_type} for training")
    
    # Create graphs
    print("\n2. Converting corners to graphs...")
    # Create processor with Phase 2 feature flags
    processor = CornerKickProcessor(
        distance_threshold=config.get('distance_threshold', 5.0),
        normalize_positions=True,
        use_enhanced_features=config.get('use_enhanced_features', False),
        use_knn_edges=config.get('use_knn_edges', False),
        knn_k=config.get('knn_k', 8),
        use_role_features=config.get('use_role_features', False),
        use_positional_context=config.get('use_positional_context', False)
    )

    if config.get('use_enhanced_features', False):
        print("   ✅ Using enhanced features: distance to goal, corner, angle, penalty box")

    if config.get('use_role_features', False):
        print("   ✅ Using role features: GK/DEF/MID/FWD one-hot encoding")

    if config.get('use_positional_context', False):
        print("   ✅ Using positional context: dist to nearest teammate/opponent, depth")

    if config.get('use_knn_edges', False):
        print(f"   ✅ Using K-nearest neighbors edges (k={config.get('knn_k', 8)})")
    else:
        print(f"   Using distance-based edges (threshold={config.get('distance_threshold', 5.0)}m)")

    if config.get('use_edge_features', False):
        print(f"   ✅ Using edge features in model: distance, angle, same_team")
    
    dataset = processor.create_dataset(corners)
    
    if len(dataset) == 0:
        print("❌ No valid graphs created")
        return
    
    # Filter out graphs without receiver labels
    labeled_dataset = [g for g in dataset if hasattr(g, 'y') and g.y is not None]
    print(f"\n   Graphs with receiver labels: {len(labeled_dataset)}/{len(dataset)}")
    
    if len(labeled_dataset) == 0:
        print("❌ No graphs with receiver labels!")
        print("   The dataset needs receiver labels for training.")
        print("   Run: python scripts/prepare_training_data.py")
        return
    
    dataset = labeled_dataset  # Use only labeled examples
    
    # Dataset statistics
    stats = get_data_statistics(dataset)
    print(f"\nDataset statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Split dataset by match (prevent data leakage)
    print("\n3. Splitting dataset by match (preventing data leakage)...")

    # Group by match_id
    match_ids = [g.match_id for g in dataset]
    unique_matches = list(set(match_ids))

    if config.get('use_validation_split', True):
        # Split into train/val/test (60/20/20 by match)
        train_matches, temp_matches = train_test_split(
            unique_matches,
            test_size=0.4,
            random_state=42
        )
        val_matches, test_matches = train_test_split(
            temp_matches,
            test_size=0.5,
            random_state=42
        )

        train_data = [g for g in dataset if g.match_id in train_matches]
        val_data = [g for g in dataset if g.match_id in val_matches]
        test_data = [g for g in dataset if g.match_id in test_matches]

        print(f"   Train: {len(train_data)} graphs (from {len(train_matches)} matches)")
        print(f"   Val:   {len(val_data)} graphs (from {len(val_matches)} matches)")
        print(f"   Test:  {len(test_data)} graphs (from {len(test_matches)} matches)")
    else:
        # Original train/test split (80/20)
        train_matches, test_matches = train_test_split(
            unique_matches,
            test_size=0.2,
            random_state=42
        )

        train_data = [g for g in dataset if g.match_id in train_matches]
        val_data = None
        test_data = [g for g in dataset if g.match_id in test_matches]

        print(f"   Train: {len(train_data)} graphs (from {len(train_matches)} matches)")
        print(f"   Test:  {len(test_data)} graphs (from {len(test_matches)} matches)")

    # Data augmentation (fixed: now updates edge features correctly)
    if config.get('use_augmentation', False):
        print("\n   Applying data augmentation (horizontal flip)...")
        original_size = len(train_data)
        augmented_train = []
        for g in train_data:
            augmented_train.append(g)
            # Apply horizontal flip (most relevant for soccer - mirrors the pitch)
            aug_h = augment_graph(g, 'horizontal')
            augmented_train.append(aug_h)
        train_data = augmented_train
        print(f"   Augmented training set: {len(train_data)} graphs (2x from {original_size})")

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False)
    val_loader = None
    if val_data is not None:
        val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False)
    
    # Initialize model
    print("\n4. Initializing model...")
    model_kwargs = {
        'node_features': config['node_features'],
        'hidden_dim': config['hidden_dim'],
        'num_layers': config['num_layers'],
        'dropout': config['dropout']
    }

    # Add edge_dim for GAT models with edge features
    if config['model_type'] == 'gat' and config.get('use_edge_features', False):
        model_kwargs['edge_dim'] = config.get('edge_dim', 3)
        print(f"   Edge features enabled (edge_dim={model_kwargs['edge_dim']})")

    model = get_model(
        model_type=config['model_type'],
        **model_kwargs
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model: {config['model_type']}")
    print(f"   Parameters: {num_params:,}")
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0)
    )
    
    # Learning rate scheduler
    scheduler = None
    if config.get('use_scheduler', True):  # Enable by default
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
        )
        print(f"   Learning rate scheduler: ReduceLROnPlateau (factor=0.5, patience=5)")
    
    # Early stopping configuration
    early_stopping_patience = config.get('early_stopping_patience', 10)
    early_stopping_min_delta = config.get('early_stopping_min_delta', 0.001)
    
    # Training loop
    print("\n5. Training...")
    label_smoothing = config.get('label_smoothing', 0.0)
    use_focal_loss = config.get('use_focal_loss', False)
    focal_gamma = config.get('focal_gamma', 2.0)

    if use_focal_loss:
        print(f"   Using focal loss (gamma={focal_gamma}, label_smoothing={label_smoothing})")
    elif label_smoothing > 0:
        print(f"   Label smoothing: {label_smoothing}")
    print(f"   Early stopping: patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")

    # Determine which loader to use for early stopping
    if val_loader is not None:
        early_stop_loader = val_loader
        early_stop_name = "Val"
        print("   Using validation set for early stopping")
    else:
        early_stop_loader = test_loader
        early_stop_name = "Test"
        print("   Using test set for early stopping (no validation set)")

    print("-" * 80)
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {early_stop_name+' Loss':>10} | {early_stop_name+' Acc':>9} | {'Top-3':>6} | {'Top-5':>6}")
    print("-" * 80)

    best_accuracy = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    train_losses = []
    train_accs = []
    eval_losses = []
    eval_accs = []
    eval_top3_accs = []
    eval_top5_accs = []

    use_edge_features = config.get('use_edge_features', False)

    for epoch in range(1, config['num_epochs'] + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device,
            label_smoothing=label_smoothing,
            use_focal_loss=use_focal_loss,
            focal_gamma=focal_gamma,
            use_edge_features=use_edge_features
        )
        eval_loss, eval_acc, eval_top3, eval_top5 = evaluate(
            model, early_stop_loader, device, use_edge_features=use_edge_features
        )

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        eval_losses.append(eval_loss)
        eval_accs.append(eval_acc)
        eval_top3_accs.append(eval_top3)
        eval_top5_accs.append(eval_top5)

        # Check for improvement
        improved = False
        if eval_acc > best_accuracy + early_stopping_min_delta:
            best_accuracy = eval_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            improved = True

            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'eval_loss': eval_loss,
                'eval_acc': eval_acc,
                'eval_top3_acc': eval_top3,
                'eval_top5_acc': eval_top5,
                'config': config
            }, 'models/checkpoints/best_model.pth')
        else:
            epochs_without_improvement += 1

        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(eval_acc)
            current_lr = optimizer.param_groups[0]['lr']
            if current_lr < config['learning_rate'] * 0.5:  # Only print when LR is reduced
                lr_marker = f" [LR={current_lr:.6f}]"
            else:
                lr_marker = ""
        else:
            lr_marker = ""

        # Print epoch results
        improvement_marker = " *" if improved else ""
        print(f"{epoch:5d} | {train_loss:10.4f} | {train_acc:8.1%} | {eval_loss:10.4f} | {eval_acc:8.1%} | {eval_top3:5.1%} | {eval_top5:5.1%}{improvement_marker}{lr_marker}")

        # Early stopping check
        if epochs_without_improvement >= early_stopping_patience:
            print(f"\n⏹️  Early stopping triggered at epoch {epoch}")
            print(f"   No improvement for {early_stopping_patience} epochs")
            print(f"   Best {early_stop_name.lower()} accuracy: {best_accuracy:.1%} at epoch {best_epoch}")
            break

    print("-" * 80)

    # Final evaluation on test set (important when using validation for early stopping)
    if val_loader is not None:
        print("\n   Final evaluation on held-out test set...")
        # Load best model for final evaluation
        checkpoint = torch.load('models/checkpoints/best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        test_loss, test_acc, test_top3, test_top5 = evaluate(
            model, test_loader, device, use_edge_features=use_edge_features
        )
        print(f"   Test set results: Acc={test_acc:.1%}, Top-3={test_top3:.1%}, Top-5={test_top5:.1%}")
    else:
        # When no validation, the eval metrics ARE the test metrics
        test_acc = best_accuracy
        test_top3 = eval_top3_accs[best_epoch - 1] if best_epoch <= len(eval_top3_accs) else eval_top3_accs[-1]
        test_top5 = eval_top5_accs[best_epoch - 1] if best_epoch <= len(eval_top5_accs) else eval_top5_accs[-1]

    print(f"\n✅ Training complete!")
    print(f"   Best {early_stop_name.lower()} accuracy: {best_accuracy:.1%} (epoch {best_epoch})")
    print(f"   Test accuracy: {test_acc:.1%}")
    print(f"   Test top-3 accuracy: {test_top3:.1%}")
    print(f"   Test top-5 accuracy: {test_top5:.1%}")
    print(f"   Random baseline: {1.0/stats['avg_players']:.1%}")
    print(f"   Total epochs: {len(train_losses)}")
    print(f"   Model saved to: models/checkpoints/best_model.pth")

    # Save training history (convert numpy types to Python types for JSON)
    history = {
        'config': config,
        'train_losses': [float(x) for x in train_losses],
        'train_accs': [float(x) for x in train_accs],
        'eval_losses': [float(x) for x in eval_losses],
        'eval_accs': [float(x) for x in eval_accs],
        'eval_top3_accs': [float(x) for x in eval_top3_accs],
        'eval_top5_accs': [float(x) for x in eval_top5_accs],
        'eval_set': early_stop_name.lower(),
        'best_eval_accuracy': float(best_accuracy),
        'best_epoch': int(best_epoch),
        'final_test_accuracy': float(test_acc),
        'final_test_top3': float(test_top3),
        'final_test_top5': float(test_top5),
        'total_epochs': len(train_losses),
        'random_baseline': float(1.0/stats['avg_players']),
        'dataset_stats': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                         for k, v in stats.items()},
        'timestamp': datetime.now().isoformat()
    }

    with open('models/checkpoints/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"   Training history saved to: models/checkpoints/training_history.json")

    # Next steps
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)

    if best_accuracy >= 0.50:
        print("🎉 Phase 1 COMPLETE! Target accuracy achieved!")
        print("\nReady for Phase 2:")
        print("  1. Add edge features (distance, angle, marking)")
        print("  2. Add node features (height, position, role)")
        print("  3. Try GAT architecture for attention-based learning")
        print("  4. Target: 70-78% accuracy (matching TacticAI paper)")
    else:
        print("📊 Phase 1 in progress - accuracy below 50% target")
        print("\nTo improve:")
        print("  1. Download more data: python scripts/download_data.py --num-matches 100")
        print("  2. Increase model capacity: hidden_dim=128, num_layers=4")
        print("  3. Tune hyperparameters: learning_rate, dropout, distance_threshold")
        print("  4. Try GAT model: model_type='gat' in config")

    print("\nVisualization:")
    print("  - Create prediction visualizations (coming soon)")
    print("  - Analyze per-team performance")
    print("=" * 70)


if __name__ == "__main__":
    # Create necessary directories
    Path("models/checkpoints").mkdir(parents=True, exist_ok=True)
    
    main()
