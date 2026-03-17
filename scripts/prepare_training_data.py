#!/usr/bin/env python3
"""
Prepare training data by linking shots with freeze frames to corner passes.
This creates a dataset suitable for training with receiver labels.
"""

import os
import sys
import pandas as pd
import pickle
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.corner_linker import link_shots_to_corners


def prepare_training_data():
    """
    Link shots with freeze frames to corner passes and prepare training dataset.
    """
    print("=" * 70)
    print("Preparing Training Data: Linking Shots to Corners")
    print("=" * 70)
    
    # Check for required files
    shots_path = 'data/processed/shots_freeze.pkl'
    corners_path = 'data/processed/corners.pkl'

    # Prefer corners.pkl (fast, robust) over full events CSV
    events_path = 'data/raw/events_multi_league.csv'
    if not os.path.exists(events_path):
        events_path = 'data/raw/events_pl_2022.csv'

    if not os.path.exists(shots_path):
        print(f"❌ Shots file not found: {shots_path}")
        print("   Please run: python scripts/download_data.py")
        return None

    # Load data
    print("\n1. Loading data...")
    shots = pd.read_pickle(shots_path)
    print(f"   ✅ Loaded {len(shots)} shots with freeze frames")

    # Use corners.pkl if available (much faster than parsing full events CSV)
    if os.path.exists(corners_path):
        print(f"   Loading corners from {corners_path}...")
        events = pd.read_pickle(corners_path)
        print(f"   ✅ Loaded {len(events)} corner passes from pickle")
    elif os.path.exists(events_path):
        print(f"   Loading events from {events_path}...")
        events = pd.read_csv(events_path, low_memory=False)
        print(f"   ✅ Loaded {len(events)} total events from CSV")
    else:
        print(f"❌ Neither {corners_path} nor {events_path} found")
        print("   Please run: python scripts/download_data.py")
        return None
    
    # Show competition breakdown if available
    if 'competition_name' in events.columns:
        print("\n   Competitions in events data:")
        for comp_name in events['competition_name'].unique():
            comp_events = events[events['competition_name'] == comp_name]
            print(f"      {comp_name}: {len(comp_events)} events")
    
    # Link shots to corners
    print("\n2. Linking shots to corner passes...")
    shots_linked = link_shots_to_corners(shots, events)
    
    # Statistics
    print("\n3. Dataset Statistics:")
    total_shots = len(shots_linked)
    linked_to_corners = shots_linked['is_from_corner'].sum()
    has_recipient_id = shots_linked['corner_pass_recipient_id'].notna().sum()
    
    print(f"   Total shots: {total_shots}")
    print(f"   Linked to corners: {linked_to_corners} ({linked_to_corners/total_shots*100:.1f}%)")
    print(f"   Have recipient_id: {has_recipient_id} ({has_recipient_id/total_shots*100:.1f}%)")
    
    # Filter for usable examples (linked and have recipient)
    usable_shots = shots_linked[
        (shots_linked['is_from_corner'] == True) &
        (shots_linked['corner_pass_recipient_id'].notna())
    ].copy()
    
    print(f"\n4. Usable Training Examples:")
    print(f"   Examples with labels: {len(usable_shots)} ({len(usable_shots)/total_shots*100:.1f}%)")
    
    if len(usable_shots) == 0:
        print("\n   ⚠️  No usable examples found!")
        print("   This means no shots could be linked to corners with recipient IDs.")
        print("   You may need to:")
        print("     1. Download more matches")
        print("     2. Use fallback labeling (pass_end_location)")
        return None
    
    # Save linked dataset
    output_path = 'data/processed/shots_linked_to_corners.pkl'
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    shots_linked.to_pickle(output_path)
    print(f"\n   ✅ Saved linked dataset to {output_path}")
    
    # Also save filtered usable examples
    usable_path = 'data/processed/training_shots.pkl'
    usable_shots.to_pickle(usable_path)
    print(f"   ✅ Saved {len(usable_shots)} usable examples to {usable_path}")
    
    # Sample check
    print("\n5. Sample Check:")
    sample = usable_shots.iloc[0]
    print(f"   Match ID: {sample['match_id']}")
    print(f"   Corner recipient_id: {sample['corner_pass_recipient_id']}")
    print(f"   Has freeze_frame: {sample.get('freeze_frame_parsed') is not None}")
    if sample.get('freeze_frame_parsed'):
        print(f"   Freeze frame players: {len(sample['freeze_frame_parsed'])}")
    
    print("\n" + "=" * 70)
    print("✅ Data preparation complete!")
    print(f"   Ready for training with {len(usable_shots)} labeled examples")
    print("=" * 70)
    
    return shots_linked, usable_shots


if __name__ == "__main__":
    prepare_training_data()

