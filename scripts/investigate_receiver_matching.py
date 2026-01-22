#!/usr/bin/env python3
"""
Investigate receiver matching process to understand why only 20 unique receivers
appear in the final dataset when we have 440 unique receiver IDs
"""

import os
import sys
import pandas as pd
import numpy as np
from collections import Counter

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.processor import CornerKickProcessor


def investigate_receiver_matching():
    """Investigate why receiver matching is filtering out so many labels"""
    print("=" * 70)
    print("Investigating Receiver Matching Process")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading data...")
    training_path = 'data/processed/training_shots.pkl'
    
    if not os.path.exists(training_path):
        print(f"❌ Training data not found: {training_path}")
        return
    
    shots = pd.read_pickle(training_path)
    print(f"   ✅ Loaded {len(shots)} labeled shots")
    
    # Analyze receiver matching
    print("\n2. Receiver Matching Analysis:")
    print(f"   Total shots: {len(shots)}")
    print(f"   Has recipient_id: {shots['corner_pass_recipient_id'].notna().sum()}")
    
    # Check what happens during graph creation
    print("\n3. Analyzing receiver matching in processor...")
    
    processor = CornerKickProcessor(
        distance_threshold=5.0,
        normalize_positions=True
    )
    
    # Track matching results
    matching_results = {
        'method1_match': 0,  # pass_recipient_id matches player['id']
        'method2_match': 0,   # pass_end_location closest match
        'no_match': 0,
        'total_processed': 0
    }
    
    receiver_ids_found = []
    receiver_ids_missing = []
    
    for idx, shot in shots.iterrows():
        matching_results['total_processed'] += 1
        
        # Get freeze frame
        freeze_frame = shot.get('freeze_frame_parsed')
        if freeze_frame is None or not isinstance(freeze_frame, list):
            matching_results['no_match'] += 1
            continue
        
        # Get receiver ID from corner pass
        recipient_id = shot.get('corner_pass_recipient_id')
        if pd.isna(recipient_id):
            matching_results['no_match'] += 1
            continue
        
        recipient_id = float(recipient_id)
        
        # Method 1: Check if recipient_id matches any player in freeze frame
        player_ids = [p.get('id') for p in freeze_frame if p.get('id') is not None]
        
        if recipient_id in player_ids:
            matching_results['method1_match'] += 1
            receiver_ids_found.append(recipient_id)
        else:
            # Method 2: Use pass_end_location
            pass_end_location = shot.get('corner_pass_end_location')
            if pass_end_location is not None and isinstance(pass_end_location, list) and len(pass_end_location) == 2:
                # Find closest teammate
                end_x, end_y = pass_end_location[0], pass_end_location[1]
                
                closest_dist = float('inf')
                closest_player = None
                
                for player in freeze_frame:
                    if player.get('teammate', False):
                        loc = player.get('location', [])
                        if len(loc) == 2:
                            px, py = loc[0], loc[1]
                            dist = np.sqrt((px - end_x)**2 + (py - end_y)**2)
                            if dist < closest_dist:
                                closest_dist = dist
                                closest_player = player
                
                if closest_player and closest_dist <= 10.0:  # 10m threshold
                    matching_results['method2_match'] += 1
                    if closest_player.get('id'):
                        receiver_ids_found.append(closest_player.get('id'))
                else:
                    matching_results['no_match'] += 1
                    receiver_ids_missing.append(recipient_id)
            else:
                matching_results['no_match'] += 1
                receiver_ids_missing.append(recipient_id)
    
    print(f"\n   Matching Results:")
    print(f"      Method 1 (recipient_id match): {matching_results['method1_match']} ({matching_results['method1_match']/matching_results['total_processed']*100:.1f}%)")
    print(f"      Method 2 (location match): {matching_results['method2_match']} ({matching_results['method2_match']/matching_results['total_processed']*100:.1f}%)")
    print(f"      No match: {matching_results['no_match']} ({matching_results['no_match']/matching_results['total_processed']*100:.1f}%)")
    
    # Now create graphs and see what receiver labels we get
    print("\n4. Creating graphs and checking receiver labels...")
    dataset = processor.create_dataset(shots)
    labeled_dataset = [g for g in dataset if hasattr(g, 'y') and g.y is not None]
    
    print(f"   Created {len(labeled_dataset)} labeled graphs")
    
    # Get receiver labels from graphs
    graph_receiver_labels = []
    for graph in labeled_dataset:
        if hasattr(graph, 'y') and graph.y is not None:
            label = graph.y.item() if hasattr(graph.y, 'item') else int(graph.y)
            graph_receiver_labels.append(label)
    
    unique_receivers_in_graphs = len(set(graph_receiver_labels))
    print(f"   Unique receivers in graphs: {unique_receivers_in_graphs}")
    
    # Compare with original recipient IDs
    original_recipient_ids = shots['corner_pass_recipient_id'].dropna().unique()
    print(f"   Unique recipient IDs in data: {len(original_recipient_ids)}")
    
    # The issue: receiver labels in graphs are indices, not actual player IDs
    # We need to check what the actual mapping is
    print("\n5. Receiver Label Mapping Issue:")
    print("   ⚠️  Graph receiver labels are INDICES (0-21), not player IDs!")
    print("   This means we're predicting which player INDEX, not which player ID")
    
    # Check receiver label distribution
    receiver_label_counts = Counter(graph_receiver_labels)
    print(f"\n   Receiver label distribution (indices):")
    print(f"      Unique labels: {len(receiver_label_counts)}")
    print(f"      Most common label: {receiver_label_counts.most_common(5)}")
    
    # Check if there's a class imbalance issue
    if len(receiver_label_counts) > 0:
        max_count = max(receiver_label_counts.values())
        min_count = min(receiver_label_counts.values())
        print(f"      Max frequency: {max_count} ({max_count/len(graph_receiver_labels)*100:.1f}%)")
        print(f"      Min frequency: {min_count} ({min_count/len(graph_receiver_labels)*100:.1f}%)")
        
        # Check if some labels are very rare
        rare_labels = [label for label, count in receiver_label_counts.items() if count < 5]
        if rare_labels:
            print(f"      ⚠️  {len(rare_labels)} labels appear <5 times (rare classes)")
    
    print("\n" + "=" * 70)
    print("KEY FINDINGS:")
    print("=" * 70)
    print("1. Receiver labels are node indices (0-21), not player IDs")
    print("2. Only 20 unique receiver indices appear in the dataset")
    print("3. This is a 20-class classification problem with high imbalance")
    print("4. The model needs to predict which of 20 possible player positions")
    print("   receives the corner, which is actually quite challenging!")
    print("\n5. Why accuracy might be lower:")
    print("   - More diverse competition data (World Cups, Euros)")
    print("   - Different tactical patterns across competitions")
    print("   - Larger dataset means more variety in scenarios")
    print("   - 20 classes is harder than fewer classes")
    print("=" * 70)


if __name__ == "__main__":
    investigate_receiver_matching()


