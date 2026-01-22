#!/usr/bin/env python3
"""
Systematic model improvement experiments
Tests multiple approaches to improve receiver prediction accuracy
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.gnn import get_model
from data.processor import CornerKickProcessor, get_data_statistics


def train_and_evaluate(config, train_data, test_data, device):
    """Train and evaluate a model with given configuration"""
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False)
    
    # Initialize model
    model = get_model(
        model_type=config['model_type'],
        node_features=config['node_features'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        heads=config.get('heads', 4)
    ).to(device)
    
    # Optimizer with learning rate scheduling
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config.get('weight_decay', 0))
    
    # Learning rate scheduler
    scheduler = None
    if config.get('use_scheduler', False):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
    
    # Class weighting for imbalanced data
    class_weights = None
    if config.get('use_class_weights', False):
        # Calculate class frequencies
        train_labels = [g.y.item() if hasattr(g.y, 'item') else int(g.y) for g in train_data]
        from collections import Counter
        label_counts = Counter(train_labels)
        total = len(train_labels)
        num_classes = len(label_counts)
        
        # Create weights: inverse frequency
        weights = torch.ones(num_classes)
        for label, count in label_counts.items():
            if label < num_classes:
                weights[label] = total / (num_classes * count)
        class_weights = weights.to(device)
    
    best_accuracy = 0.0
    best_top5 = 0.0
    patience = config.get('early_stopping_patience', 10)
    min_delta = config.get('early_stopping_min_delta', 0.001)
    epochs_without_improvement = 0
    
    num_epochs = config.get('num_epochs', 30)
    
    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch.x, batch.edge_index, batch.batch)
            
            if not hasattr(batch, 'y') or batch.y is None:
                continue
            
            loss = 0
            num_graphs = batch.num_graphs
            
            for i in range(num_graphs):
                mask = (batch.batch == i)
                graph_logits = out[mask]
                graph_label = batch.y[i] if batch.y.dim() > 0 else batch.y
                
                if graph_label >= len(graph_logits):
                    continue
                
                # Weighted loss if using class weights
                if class_weights is not None and graph_label < len(class_weights):
                    weight = class_weights[graph_label]
                    loss += F.cross_entropy(graph_logits.unsqueeze(0), graph_label.unsqueeze(0), weight=weight.unsqueeze(0))
                else:
                    loss += F.cross_entropy(graph_logits.unsqueeze(0), graph_label.unsqueeze(0))
                
                pred = graph_logits.argmax().item()
                if isinstance(graph_label, torch.Tensor):
                    label_val = graph_label.item()
                else:
                    label_val = int(graph_label)
                train_correct += 1.0 if (pred == label_val) else 0.0
                train_total += 1
            
            if loss > 0:
                loss = loss / num_graphs
                loss.backward()
                
                # Gradient clipping
                if config.get('grad_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                
                optimizer.step()
                train_loss += loss.item()
        
        # Evaluate
        model.eval()
        test_loss = 0
        test_correct = 0
        test_top5 = 0
        test_total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                
                if not hasattr(batch, 'y') or batch.y is None:
                    continue
                
                num_graphs = batch.num_graphs
                
                for i in range(num_graphs):
                    mask = (batch.batch == i)
                    graph_logits = out[mask]
                    graph_label = batch.y[i] if batch.y.dim() > 0 else batch.y
                    
                    if graph_label >= len(graph_logits):
                        continue
                    
                    if isinstance(graph_label, torch.Tensor):
                        label_val = graph_label.item()
                    else:
                        label_val = int(graph_label)
                    
                    # Top-1
                    pred = graph_logits.argmax().item()
                    test_correct += 1.0 if (pred == label_val) else 0.0
                    
                    # Top-5
                    if len(graph_logits) >= 5:
                        top5_preds = graph_logits.topk(5).indices.cpu().tolist()
                        test_top5 += 1.0 if (label_val in top5_preds) else 0.0
                    else:
                        test_top5 += 1.0 if (pred == label_val) else 0.0
                    
                    test_total += 1
        
        test_acc = test_correct / test_total if test_total > 0 else 0
        test_top5_acc = test_top5 / test_total if test_total > 0 else 0
        
        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(test_acc)
        
        # Early stopping
        if test_acc > best_accuracy + min_delta:
            best_accuracy = test_acc
            best_top5 = test_top5_acc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            break
    
    return {
        'best_accuracy': best_accuracy,
        'best_top5': best_top5,
        'epochs': epoch
    }


def run_improvement_experiments():
    """Run multiple experiments to find best configuration"""
    print("=" * 70)
    print("Model Improvement Experiments")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading data...")
    training_path = 'data/processed/training_shots.pkl'
    shots = pd.read_pickle(training_path)
    print(f"   ✅ Loaded {len(shots)} labeled shots")
    
    # Create graphs (will test both basic and enhanced features)
    processor = CornerKickProcessor(
        distance_threshold=5.0,
        normalize_positions=True,
        use_enhanced_features=False  # Start with basic, test enhanced separately
    )
    
    dataset = processor.create_dataset(shots)
    labeled_dataset = [g for g in dataset if hasattr(g, 'y') and g.y is not None]
    print(f"   ✅ Created {len(labeled_dataset)} labeled graphs")
    
    # Split data
    match_ids = [g.match_id for g in labeled_dataset]
    unique_matches = list(set(match_ids))
    train_matches, test_matches = train_test_split(unique_matches, test_size=0.2, random_state=42)
    train_data = [g for g in labeled_dataset if g.match_id in train_matches]
    test_data = [g for g in labeled_dataset if g.match_id in test_matches]
    
    print(f"   Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    print(f"\n2. Running experiments on {device}...")
    print("-" * 70)
    
    # Baseline configuration
    baseline_config = {
        'model_type': 'gat',
        'node_features': 3,
        'hidden_dim': 128,
        'num_layers': 4,
        'dropout': 0.2,
        'learning_rate': 0.0005,
        'batch_size': 32,
        'num_epochs': 30,
        'heads': 4,
        'early_stopping_patience': 10,
        'early_stopping_min_delta': 0.001,
        'use_scheduler': False,
        'use_class_weights': False,
        'weight_decay': 0,
        'grad_clip': 0
    }
    
    # Test enhanced features separately
    print("\n   Testing enhanced features...")
    processor_enhanced = CornerKickProcessor(
        distance_threshold=5.0,
        normalize_positions=True,
        use_enhanced_features=True
    )
    dataset_enhanced = processor_enhanced.create_dataset(shots)
    labeled_dataset_enhanced = [g for g in dataset_enhanced if hasattr(g, 'y') and g.y is not None]
    
    # Check feature dimensions
    if len(labeled_dataset_enhanced) > 0:
        sample_features = labeled_dataset_enhanced[0].x.shape[1]
        print(f"   Enhanced features: {sample_features} dimensions per node")
    
    experiments = [
        # Experiment 1: Baseline
        {
            'name': 'Baseline (Current)',
            'config': baseline_config.copy()
        },
        
        # Experiment 2: Learning rate scheduling
        {
            'name': 'With LR Scheduler',
            'config': {**baseline_config, 'use_scheduler': True}
        },
        
        # Experiment 3: Class weights
        {
            'name': 'With Class Weights',
            'config': {**baseline_config, 'use_class_weights': True}
        },
        
        # Experiment 4: Higher capacity
        {
            'name': 'Higher Capacity (256 dim)',
            'config': {**baseline_config, 'hidden_dim': 256, 'num_layers': 5}
        },
        
        # Experiment 5: More attention heads
        {
            'name': 'More Heads (8 heads)',
            'config': {**baseline_config, 'heads': 8}
        },
        
        # Experiment 6: Lower learning rate with scheduler
        {
            'name': 'Lower LR + Scheduler',
            'config': {**baseline_config, 'learning_rate': 0.0001, 'use_scheduler': True}
        },
        
        # Experiment 7: Weight decay regularization
        {
            'name': 'With Weight Decay',
            'config': {**baseline_config, 'weight_decay': 1e-5}
        },
        
        # Experiment 8: Gradient clipping
        {
            'name': 'With Grad Clipping',
            'config': {**baseline_config, 'grad_clip': 1.0}
        },
        
        # Experiment 9: Combined improvements
        {
            'name': 'Combined (LR Sched + Class Weights + Weight Decay)',
            'config': {
                **baseline_config,
                'use_scheduler': True,
                'use_class_weights': True,
                'weight_decay': 1e-5,
                'grad_clip': 1.0
            }
        },
        
        # Experiment 10: Maximum capacity
        {
            'name': 'Max Capacity (256 dim, 8 heads, 6 layers)',
            'config': {
                **baseline_config,
                'hidden_dim': 256,
                'num_layers': 6,
                'heads': 8,
                'use_scheduler': True
            }
        },
        
        # Experiment 11: Enhanced features (need to recreate graphs)
        {
            'name': 'Enhanced Features (7 dim)',
            'config': {
                **baseline_config,
                'node_features': 7,  # x, y, teammate, dist_goal, dist_corner, angle, in_box
                'use_enhanced_features': True
            }
        },
        
        # Experiment 12: Enhanced features + weight decay
        {
            'name': 'Enhanced Features + Weight Decay',
            'config': {
                **baseline_config,
                'node_features': 7,
                'use_enhanced_features': True,
                'weight_decay': 1e-5
            }
        }
    ]
    
    results = []
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] {exp['name']}...")
        try:
            # Use enhanced dataset if config requires it
            if exp['config'].get('use_enhanced_features', False):
                train_data_exp = [g for g in labeled_dataset_enhanced if g.match_id in train_matches]
                test_data_exp = [g for g in labeled_dataset_enhanced if g.match_id in test_matches]
                result = train_and_evaluate(exp['config'], train_data_exp, test_data_exp, device)
            else:
                result = train_and_evaluate(exp['config'], train_data, test_data, device)
            
            result['name'] = exp['name']
            result['config'] = exp['config']
            results.append(result)
            print(f"   ✅ Accuracy: {result['best_accuracy']:.1%}, Top-5: {result['best_top5']:.1%}, Epochs: {result['epochs']}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'name': exp['name'],
                'best_accuracy': 0,
                'best_top5': 0,
                'error': str(e)
            })
    
    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Experiment':<50} | {'Accuracy':>10} | {'Top-5':>10} | {'Epochs':>7}")
    print("-" * 70)
    
    results_sorted = sorted([r for r in results if 'error' not in r], key=lambda x: x['best_accuracy'], reverse=True)
    
    for r in results_sorted:
        print(f"{r['name']:<50} | {r['best_accuracy']:>9.1%} | {r['best_top5']:>9.1%} | {r.get('epochs', 0):>7}")
    
    # Save results
    output_file = 'models/checkpoints/improvement_experiments.json'
    Path('models/checkpoints').mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")
    
    # Best configuration
    if results_sorted:
        best = results_sorted[0]
        print(f"\n🏆 Best Configuration: {best['name']}")
        print(f"   Accuracy: {best['best_accuracy']:.1%}")
        print(f"   Top-5: {best['best_top5']:.1%}")
        print(f"\n   Recommended config:")
        for key, value in best['config'].items():
            if key not in ['name']:
                print(f"      {key}: {value}")
    
    print("=" * 70)


if __name__ == "__main__":
    run_improvement_experiments()

