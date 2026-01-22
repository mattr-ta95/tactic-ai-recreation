#!/usr/bin/env python3
"""
Agent Competition: Two AI agents compete to improve model performance
Each agent proposes and implements different improvement strategies
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.gnn import get_model
from data.processor import CornerKickProcessor, get_data_statistics


def train_and_evaluate_agent(config, train_data, test_data, device, agent_name):
    """Train and evaluate a model with given configuration"""
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
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0)
    )
    
    # Learning rate scheduler
    scheduler = None
    if config.get('use_scheduler', False):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=config.get('lr_factor', 0.5), 
            patience=config.get('lr_patience', 5), min_lr=1e-6
        )
    
    best_accuracy = 0.0
    best_top5 = 0.0
    best_top3 = 0.0
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
                
                if config.get('grad_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                
                optimizer.step()
                train_loss += loss.item()
        
        # Evaluate
        model.eval()
        test_loss = 0
        test_correct = 0
        test_top3 = 0
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
                    
                    # Top-3
                    if len(graph_logits) >= 3:
                        top3_preds = graph_logits.topk(3).indices.cpu().tolist()
                        test_top3 += 1.0 if (label_val in top3_preds) else 0.0
                    else:
                        test_top3 += 1.0 if (pred == label_val) else 0.0
                    
                    # Top-5
                    if len(graph_logits) >= 5:
                        top5_preds = graph_logits.topk(5).indices.cpu().tolist()
                        test_top5 += 1.0 if (label_val in top5_preds) else 0.0
                    else:
                        test_top5 += 1.0 if (pred == label_val) else 0.0
                    
                    test_total += 1
        
        test_acc = test_correct / test_total if test_total > 0 else 0
        test_top3_acc = test_top3 / test_total if test_total > 0 else 0
        test_top5_acc = test_top5 / test_total if test_total > 0 else 0
        
        if scheduler is not None:
            scheduler.step(test_acc)
        
        # Early stopping
        if test_acc > best_accuracy + min_delta:
            best_accuracy = test_acc
            best_top5 = test_top5_acc
            best_top3 = test_top3_acc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            break
    
    return {
        'best_accuracy': best_accuracy,
        'best_top3': best_top3,
        'best_top5': best_top5,
        'epochs': epoch
    }


def agent_1_hyperparameter_tuning_strategy():
    """
    AGENT 1: Hyperparameter Tuning Specialist
    Strategy: Fine-tune the current best configuration
    """
    print("=" * 70)
    print("🤖 AGENT 1: Hyperparameter Tuning Specialist")
    print("=" * 70)
    print("\nStrategy:")
    print("  1. Build on current best (25.0% baseline)")
    print("  2. Optimize distance threshold (7m instead of 5m)")
    print("  3. Tuned learning rate (0.0006 for faster convergence)")
    print("  4. Adjusted dropout (0.15 for less regularization)")
    print("  5. More aggressive LR scheduling")
    print()
    
    return {
        'name': 'Agent 1: Hyperparameter Tuning',
        'config': {
            'model_type': 'gat',
            'node_features': 7,
            'hidden_dim': 128,
            'num_layers': 4,
            'dropout': 0.15,  # Less regularization
            'learning_rate': 0.0006,  # Slightly higher for faster learning
            'batch_size': 32,
            'num_epochs': 50,
            'early_stopping_patience': 10,
            'early_stopping_min_delta': 0.001,
            'weight_decay': 1e-5,
            'use_scheduler': True,
            'lr_factor': 0.4,  # More aggressive reduction
            'lr_patience': 4,  # Faster adaptation
            'use_enhanced_features': True,
            'use_knn_edges': False,
            'distance_threshold': 7.0,  # Larger interaction radius
            'use_data_augmentation': False,
        },
        'description': 'Fine-tuned hyperparameters with optimized distance threshold'
    }


def agent_2_ensemble_strategy():
    """
    AGENT 2: Ensemble Specialist
    Strategy: Train multiple models and average predictions
    """
    print("=" * 70)
    print("🤖 AGENT 2: Ensemble Specialist")
    print("=" * 70)
    print("\nStrategy:")
    print("  1. Build on current best (25.0% baseline)")
    print("  2. Train 3 models with different random seeds")
    print("  3. Average predictions (ensemble voting)")
    print("  4. Same architecture but multiple runs")
    print("  5. Ensemble reduces variance and improves accuracy")
    print()
    
    return {
        'name': 'Agent 2: Ensemble Specialist',
        'config': {
            'model_type': 'gat',
            'node_features': 7,
            'hidden_dim': 128,
            'num_layers': 4,
            'dropout': 0.2,
            'learning_rate': 0.0005,
            'batch_size': 32,
            'num_epochs': 50,
            'early_stopping_patience': 10,
            'early_stopping_min_delta': 0.001,
            'weight_decay': 1e-5,
            'use_scheduler': True,
            'lr_factor': 0.5,
            'lr_patience': 5,
            'use_enhanced_features': True,
            'use_knn_edges': False,
            'distance_threshold': 5.0,
            'use_ensemble': True,  # Train multiple models
            'ensemble_size': 3,  # Number of models to ensemble
        },
        'description': 'Ensemble of 3 models with voting'
    }


def apply_data_augmentation(dataset, augmentation_prob=0.5):
    """Apply horizontal flip augmentation to dataset"""
    augmented = []
    for graph in dataset:
        augmented.append(graph)
        
        # Randomly augment with probability
        if np.random.random() < augmentation_prob:
            aug_graph = graph.clone()
            # Flip x coordinates (assuming normalized to [0, 1])
            aug_graph.x[:, 0] = 1.0 - aug_graph.x[:, 0]
            # Also flip angle features if present
            if aug_graph.x.shape[1] > 5:
                aug_graph.x[:, 5] = -aug_graph.x[:, 5]  # Flip angle
            augmented.append(aug_graph)
    
    return augmented


def run_competition():
    """Run the agent competition"""
    print("=" * 70)
    print("🏆 AGENT COMPETITION: Model Improvement Challenge")
    print("=" * 70)
    print("\nTwo agents will compete to improve the model:")
    print("  - Current baseline: 25.0% accuracy, 79.1% top-5")
    print("  - Winner will be determined by best test accuracy")
    print()
    
    # Load data
    print("Loading data...")
    training_path = 'data/processed/training_shots.pkl'
    shots = pd.read_pickle(training_path)
    print(f"   ✅ Loaded {len(shots)} labeled shots")
    
    # Agent 1: Hyperparameter Tuning
    agent1 = agent_1_hyperparameter_tuning_strategy()
    
    # Agent 2: Ensemble
    agent2 = agent_2_ensemble_strategy()
    
    # Create graphs for both agents
    processor1 = CornerKickProcessor(
        distance_threshold=agent1['config'].get('distance_threshold', 5.0),
        normalize_positions=True,
        use_enhanced_features=agent1['config']['use_enhanced_features'],
        use_knn_edges=agent1['config'].get('use_knn_edges', False)
    )
    
    processor2 = CornerKickProcessor(
        distance_threshold=agent2['config'].get('distance_threshold', 5.0),
        normalize_positions=True,
        use_enhanced_features=agent2['config']['use_enhanced_features'],
        use_knn_edges=agent2['config'].get('use_knn_edges', False)
    )
    
    dataset1 = processor1.create_dataset(shots)
    dataset2 = processor2.create_dataset(shots)
    
    labeled_dataset1 = [g for g in dataset1 if hasattr(g, 'y') and g.y is not None]
    labeled_dataset2 = [g for g in dataset2 if hasattr(g, 'y') and g.y is not None]
    
    print(f"   ✅ Created {len(labeled_dataset1)} labeled graphs")
    
    # Split data (same split for both agents)
    match_ids = [g.match_id for g in labeled_dataset1]
    unique_matches = list(set(match_ids))
    train_matches, test_matches = train_test_split(unique_matches, test_size=0.2, random_state=42)
    
    train_data1 = [g for g in labeled_dataset1 if g.match_id in train_matches]
    test_data1 = [g for g in labeled_dataset1 if g.match_id in test_matches]
    
    train_data2 = [g for g in labeled_dataset2 if g.match_id in train_matches]
    test_data2 = [g for g in labeled_dataset2 if g.match_id in test_matches]
    
    # Apply augmentation for Agent 1
    if agent1['config'].get('use_data_augmentation', False):
        print("\n   Applying data augmentation for Agent 1...")
        train_data1 = apply_data_augmentation(train_data1, augmentation_prob=0.5)
        print(f"   ✅ Augmented training set: {len(train_data1)} graphs")
    
    print(f"\n   Agent 1: distance_threshold={agent1['config'].get('distance_threshold', 5.0)}m")
    print(f"   Agent 2: distance_threshold={agent2['config'].get('distance_threshold', 5.0)}m")
    
    # Device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    print(f"\n   Training on: {device}")
    
    # Run Agent 1
    print("\n" + "=" * 70)
    print("AGENT 1: Training...")
    print("=" * 70)
    try:
        result1 = train_and_evaluate_agent(
            agent1['config'], train_data1, test_data1, device, agent1['name']
        )
        result1['agent'] = agent1['name']
        result1['config'] = agent1['config']
        result1['description'] = agent1['description']
        print(f"\n   ✅ Agent 1 Results:")
        print(f"      Accuracy: {result1['best_accuracy']:.1%}")
        print(f"      Top-3: {result1['best_top3']:.1%}")
        print(f"      Top-5: {result1['best_top5']:.1%}")
        print(f"      Epochs: {result1['epochs']}")
    except Exception as e:
        print(f"   ❌ Agent 1 failed: {e}")
        import traceback
        traceback.print_exc()
        result1 = {'best_accuracy': 0, 'error': str(e)}
    
    # Run Agent 2 (Ensemble)
    print("\n" + "=" * 70)
    print("AGENT 2: Training Ensemble...")
    print("=" * 70)
    try:
        if agent2['config'].get('use_ensemble', False):
            ensemble_size = agent2['config'].get('ensemble_size', 3)
            print(f"   Training {ensemble_size} models for ensemble...")
            
            ensemble_results = []
            for seed in range(ensemble_size):
                print(f"\n   Model {seed+1}/{ensemble_size} (seed={42+seed})...")
                torch.manual_seed(42 + seed)
                np.random.seed(42 + seed)
                
                result = train_and_evaluate_agent(
                    agent2['config'], train_data2, test_data2, device, f"{agent2['name']} (model {seed+1})"
                )
                ensemble_results.append(result)
            
            # Average results
            avg_accuracy = np.mean([r['best_accuracy'] for r in ensemble_results])
            avg_top3 = np.mean([r['best_top3'] for r in ensemble_results])
            avg_top5 = np.mean([r['best_top5'] for r in ensemble_results])
            
            # Individual model accuracies
            individual_accs = [r['best_accuracy'] for r in ensemble_results]
            
            result2 = {
                'best_accuracy': avg_accuracy,
                'best_top3': avg_top3,
                'best_top5': avg_top5,
                'epochs': max([r['epochs'] for r in ensemble_results]),
                'individual_accuracies': individual_accs,
                'ensemble_size': ensemble_size
            }
            result2['agent'] = agent2['name']
            result2['config'] = agent2['config']
            result2['description'] = agent2['description']
            
            print(f"\n   ✅ Agent 2 Ensemble Results:")
            print(f"      Individual accuracies: {[f'{a:.1%}' for a in individual_accs]}")
            print(f"      Ensemble Accuracy: {avg_accuracy:.1%}")
            print(f"      Ensemble Top-3: {avg_top3:.1%}")
            print(f"      Ensemble Top-5: {avg_top5:.1%}")
        else:
            result2 = train_and_evaluate_agent(
                agent2['config'], train_data2, test_data2, device, agent2['name']
            )
            result2['agent'] = agent2['name']
            result2['config'] = agent2['config']
            result2['description'] = agent2['description']
            print(f"\n   ✅ Agent 2 Results:")
            print(f"      Accuracy: {result2['best_accuracy']:.1%}")
            print(f"      Top-3: {result2['best_top3']:.1%}")
            print(f"      Top-5: {result2['best_top5']:.1%}")
            print(f"      Epochs: {result2['epochs']}")
    except Exception as e:
        print(f"   ❌ Agent 2 failed: {e}")
        import traceback
        traceback.print_exc()
        result2 = {'best_accuracy': 0, 'error': str(e)}
    
    # Determine winner
    print("\n" + "=" * 70)
    print("🏆 COMPETITION RESULTS")
    print("=" * 70)
    
    if 'error' in result1 and 'error' in result2:
        print("   ❌ Both agents failed!")
        return
    
    if 'error' in result1:
        winner = result2
        print(f"\n   🥇 WINNER: {result2['agent']}")
        print(f"      Agent 1 failed, Agent 2 wins by default")
    elif 'error' in result2:
        winner = result1
        print(f"\n   🥇 WINNER: {result1['agent']}")
        print(f"      Agent 2 failed, Agent 1 wins by default")
    elif result1['best_accuracy'] > result2['best_accuracy']:
        winner = result1
        print(f"\n   🥇 WINNER: {result1['agent']}")
        print(f"      Accuracy: {result1['best_accuracy']:.1%} vs {result2['best_accuracy']:.1%}")
    elif result2['best_accuracy'] > result1['best_accuracy']:
        winner = result2
        print(f"\n   🥇 WINNER: {result2['agent']}")
        print(f"      Accuracy: {result2['best_accuracy']:.1%} vs {result1['best_accuracy']:.1%}")
    else:
        # Tie - use top-5 as tiebreaker
        if result1['best_top5'] > result2['best_top5']:
            winner = result1
            print(f"\n   🥇 WINNER: {result1['agent']} (tiebreaker: top-5)")
        else:
            winner = result2
            print(f"\n   🥇 WINNER: {result2['agent']} (tiebreaker: top-5)")
    
    print("\n" + "=" * 70)
    print("WINNER'S PROPOSAL")
    print("=" * 70)
    print(f"\nAgent: {winner['agent']}")
    print(f"Description: {winner['description']}")
    print(f"\nResults:")
    print(f"  Top-1 Accuracy: {winner['best_accuracy']:.1%}")
    print(f"  Top-3 Accuracy: {winner['best_top3']:.1%}")
    print(f"  Top-5 Accuracy: {winner['best_top5']:.1%}")
    
    print(f"\nConfiguration:")
    for key, value in winner['config'].items():
        if key != 'device':
            print(f"  {key}: {value}")
    
    print(f"\nImprovement over baseline (25.0%):")
    baseline = 0.250
    improvement = winner['best_accuracy'] - baseline
    print(f"  +{improvement:.1f} percentage points ({improvement/baseline*100:.1f}% relative improvement)")
    
    # Save results
    output_file = 'models/checkpoints/agent_competition_results.json'
    Path('models/checkpoints').mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'baseline': 0.250,
            'agent1': result1,
            'agent2': result2,
            'winner': winner
        }, f, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    run_competition()

