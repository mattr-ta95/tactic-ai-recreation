#!/usr/bin/env python3
"""
Investigate why accuracy decreased with expanded dataset
Analyzes data distribution, label quality, and competition differences
"""

import os
import sys
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.processor import CornerKickProcessor, get_data_statistics
from sklearn.model_selection import train_test_split


def investigate_data():
    """Investigate the expanded dataset"""
    print("=" * 70)
    print("Investigating Performance Drop with Expanded Dataset")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading data...")
    training_path = 'data/processed/training_shots.pkl'
    events_path = 'data/raw/events_multi_league.csv'
    
    if not os.path.exists(training_path):
        print(f"❌ Training data not found: {training_path}")
        return
    
    shots = pd.read_pickle(training_path)
    print(f"   ✅ Loaded {len(shots)} labeled shots")
    
    if os.path.exists(events_path):
        events = pd.read_csv(events_path, low_memory=False)
        print(f"   ✅ Loaded {len(events)} events")
    else:
        events = None
    
    # Competition breakdown
    print("\n2. Competition Distribution:")
    if 'competition_name' in shots.columns:
        comp_counts = shots['competition_name'].value_counts()
        print(f"   Total examples: {len(shots)}")
        for comp, count in comp_counts.items():
            pct = count / len(shots) * 100
            print(f"   {comp}: {count} ({pct:.1f}%)")
    else:
        print("   ⚠️  No competition_name column")
    
    # Analyze label quality
    print("\n3. Label Quality Analysis:")
    print(f"   Total examples: {len(shots)}")
    print(f"   Has recipient_id: {shots['corner_pass_recipient_id'].notna().sum()} ({shots['corner_pass_recipient_id'].notna().mean()*100:.1f}%)")
    print(f"   Has freeze_frame: {shots.get('freeze_frame_parsed', pd.Series()).notna().sum() if 'freeze_frame_parsed' in shots.columns else 'N/A'}")
    
    # Receiver label distribution
    print("\n4. Receiver Label Distribution:")
    if 'corner_pass_recipient_id' in shots.columns:
        recipient_counts = shots['corner_pass_recipient_id'].value_counts()
        print(f"   Unique receivers: {len(recipient_counts)}")
        print(f"   Most common receivers:")
        for rec_id, count in recipient_counts.head(10).items():
            print(f"      Receiver {rec_id}: {count} ({count/len(shots)*100:.1f}%)")
        print(f"   Label balance (entropy): {calculate_entropy(recipient_counts)}")
    
    # Create graphs and analyze
    print("\n5. Converting to graphs...")
    processor = CornerKickProcessor(
        distance_threshold=5.0,
        normalize_positions=True
    )
    
    dataset = processor.create_dataset(shots)
    labeled_dataset = [g for g in dataset if hasattr(g, 'y') and g.y is not None]
    
    print(f"   Created {len(labeled_dataset)} labeled graphs")
    
    # Graph statistics by competition
    print("\n6. Graph Characteristics by Competition:")
    if 'competition_name' in shots.columns:
        comp_graphs = {}
        # Create mapping from shot index to competition
        shot_to_comp = {}
        for idx, row in shots.iterrows():
            if idx < len(labeled_dataset):
                shot_to_comp[idx] = row.get('competition_name', 'Unknown')
        
        for i, graph in enumerate(labeled_dataset):
            # Try to map graph back to shot
            comp = shot_to_comp.get(i, 'Unknown')
            if comp not in comp_graphs:
                comp_graphs[comp] = []
            comp_graphs[comp].append(graph)
        
        for comp, graphs in comp_graphs.items():
            stats = get_data_statistics(graphs)
            print(f"\n   {comp} ({len(graphs)} graphs):")
            print(f"      Avg players: {stats.get('avg_players', 0):.2f}")
            print(f"      Avg edges: {stats.get('avg_edges', 0):.2f}")
            print(f"      Avg attackers: {stats.get('avg_attackers', 0):.2f}")
            print(f"      Avg defenders: {stats.get('avg_defenders', 0):.2f}")
    
    # Train/test split analysis
    print("\n7. Train/Test Split Analysis:")
    match_ids = [g.match_id for g in labeled_dataset]
    unique_matches = list(set(match_ids))
    
    train_matches, test_matches = train_test_split(
        unique_matches,
        test_size=0.2,
        random_state=42
    )
    
    train_data = [g for g in labeled_dataset if g.match_id in train_matches]
    test_data = [g for g in labeled_dataset if g.match_id in test_matches]
    
    print(f"   Train: {len(train_data)} graphs from {len(train_matches)} matches")
    print(f"   Test:  {len(test_data)} graphs from {len(test_matches)} matches")
    
    # Competition distribution in train vs test
    if 'competition_name' in shots.columns:
        print("\n   Competition distribution in train/test:")
        
        # Create match_id to competition mapping
        match_to_comp = {}
        for i, graph in enumerate(labeled_dataset):
            if i < len(shots):
                match_to_comp[graph.match_id] = shots.iloc[i].get('competition_name', 'Unknown')
        
        train_comps = [match_to_comp.get(m, 'Unknown') for m in train_matches]
        test_comps = [match_to_comp.get(m, 'Unknown') for m in test_matches]
        
        train_comp_counts = Counter(train_comps)
        test_comp_counts = Counter(test_comps)
        
        print("\n   Train matches by competition:")
        for comp, count in train_comp_counts.items():
            pct = count / len(train_matches) * 100
            print(f"      {comp}: {count} matches ({pct:.1f}%)")
        
        print("\n   Test matches by competition:")
        for comp, count in test_comp_counts.items():
            pct = count / len(test_matches) * 100
            print(f"      {comp}: {count} matches ({pct:.1f}%)")
    
    # Receiver label distribution in train vs test
    print("\n8. Receiver Label Distribution (Train vs Test):")
    train_labels = [g.y.item() if hasattr(g.y, 'item') else int(g.y) for g in train_data]
    test_labels = [g.y.item() if hasattr(g.y, 'item') else int(g.y) for g in test_data]
    
    train_label_counts = Counter(train_labels)
    test_label_counts = Counter(test_labels)
    
    print(f"   Train: {len(train_label_counts)} unique receivers")
    print(f"   Test:  {len(test_label_counts)} unique receivers")
    print(f"   Overlap: {len(set(train_label_counts.keys()) & set(test_label_counts.keys()))} common receivers")
    
    # Class imbalance
    print("\n9. Class Imbalance Analysis:")
    all_labels = train_labels + test_labels
    all_label_counts = Counter(all_labels)
    
    print(f"   Total unique receivers: {len(all_label_counts)}")
    print(f"   Most common receiver appears {max(all_label_counts.values())} times ({max(all_label_counts.values())/len(all_labels)*100:.1f}%)")
    print(f"   Least common receiver appears {min(all_label_counts.values())} times")
    
    # Graph size distribution
    print("\n10. Graph Size Distribution:")
    train_sizes = [len(g.x) for g in train_data]
    test_sizes = [len(g.x) for g in test_data]
    
    print(f"   Train - Mean: {np.mean(train_sizes):.2f}, Std: {np.std(train_sizes):.2f}, Min: {min(train_sizes)}, Max: {max(train_sizes)}")
    print(f"   Test  - Mean: {np.mean(test_sizes):.2f}, Std: {np.std(test_sizes):.2f}, Min: {min(test_sizes)}, Max: {max(test_sizes)}")
    
    # Potential issues summary
    print("\n" + "=" * 70)
    print("POTENTIAL ISSUES IDENTIFIED:")
    print("=" * 70)
    
    issues = []
    
    # Check for competition imbalance
    if 'competition_name' in shots.columns:
        comp_counts = shots['competition_name'].value_counts()
        if len(comp_counts) > 1:
            max_comp_pct = comp_counts.iloc[0] / len(shots) * 100
            if max_comp_pct > 70:
                issues.append(f"⚠️  Competition imbalance: {comp_counts.index[0]} dominates ({max_comp_pct:.1f}%)")
    
    # Check for class imbalance
    if len(all_label_counts) > 0:
        max_label_pct = max(all_label_counts.values()) / len(all_labels) * 100
        if max_label_pct > 30:
            issues.append(f"⚠️  Severe class imbalance: most common receiver is {max_label_pct:.1f}% of data")
    
    # Check for train/test distribution mismatch
    if len(train_label_counts) > 0 and len(test_label_counts) > 0:
        overlap = len(set(train_label_counts.keys()) & set(test_label_counts.keys()))
        overlap_pct = overlap / len(set(all_label_counts.keys())) * 100
        if overlap_pct < 50:
            issues.append(f"⚠️  Low train/test label overlap: only {overlap_pct:.1f}% of receivers appear in both sets")
    
    # Check graph size differences
    if len(train_sizes) > 0 and len(test_sizes) > 0:
        train_mean = np.mean(train_sizes)
        test_mean = np.mean(test_sizes)
        if abs(train_mean - test_mean) > 2:
            issues.append(f"⚠️  Graph size mismatch: train avg {train_mean:.1f} vs test avg {test_mean:.1f} players")
    
    if issues:
        for issue in issues:
            print(f"   {issue}")
    else:
        print("   ✅ No obvious data quality issues detected")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS:")
    print("=" * 70)
    print("   1. Check if test set has different competition distribution")
    print("   2. Verify label quality is consistent across competitions")
    print("   3. Consider stratified splitting by competition")
    print("   4. Check if model capacity is sufficient for larger dataset")
    print("   5. Consider data augmentation or class balancing")
    print("=" * 70)


def calculate_entropy(counts):
    """Calculate entropy of label distribution"""
    if isinstance(counts, pd.Series):
        total = counts.sum()
        probs = counts / total
    else:
        total = sum(counts.values() if hasattr(counts, 'values') else counts)
        probs = [c / total for c in (counts.values() if hasattr(counts, 'values') else counts)]
    
    if total == 0:
        return 0.0
    
    if isinstance(probs, pd.Series):
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    else:
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    
    return entropy


if __name__ == "__main__":
    investigate_data()

