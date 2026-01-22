#!/usr/bin/env python3
"""
Visualize sample corner kicks from the dataset
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def visualize_corner(corner, save_path=None, show=True):
    """
    Visualize a single corner kick
    
    Args:
        corner: Row from corners dataframe
        save_path: Path to save figure (optional)
        show: Whether to display the figure
    """
    freeze_frame = corner['freeze_frame_parsed']
    
    # Create pitch
    pitch = Pitch(
        pitch_type='statsbomb',
        pitch_color='#22312b',
        line_color='#c7d5cc',
        linewidth=2
    )
    fig, ax = pitch.draw(figsize=(12, 8))
    
    # Separate attackers and defenders
    attackers = [p for p in freeze_frame if p['teammate']]
    defenders = [p for p in freeze_frame if not p['teammate']]
    
    # Plot attackers (red)
    if attackers:
        att_x = [p['location'][0] for p in attackers]
        att_y = [p['location'][1] for p in attackers]
        pitch.scatter(
            att_x, att_y, ax=ax,
            c='#FF6B6B', s=400,
            edgecolors='white', linewidths=2,
            label=f'Attackers ({len(attackers)})',
            zorder=3
        )
    
    # Plot defenders (blue)
    if defenders:
        def_x = [p['location'][0] for p in defenders]
        def_y = [p['location'][1] for p in defenders]
        pitch.scatter(
            def_x, def_y, ax=ax,
            c='#4ECDC4', s=400,
            edgecolors='white', linewidths=2,
            label=f'Defenders ({len(defenders)})',
            zorder=3
        )
    
    # Plot ball
    ball_x, ball_y = corner['location']
    pitch.scatter(
        ball_x, ball_y, ax=ax,
        c='white', s=600,
        marker='o',
        edgecolors='black', linewidths=3,
        label='Ball',
        zorder=4
    )
    
    # Add info
    info_text = f"Match: {corner.get('match_id', 'N/A')}\n"
    info_text += f"Team: {corner.get('team', 'N/A')}\n"
    info_text += f"Minute: {corner.get('minute', 'N/A')}\n"
    info_text += f"Players: {len(freeze_frame)}"
    
    ax.text(
        5, 75, info_text,
        fontsize=11,
        color='white',
        bbox=dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.8),
        verticalalignment='top'
    )
    
    plt.legend(
        loc='upper right',
        fontsize=11,
        facecolor='#1a1a1a',
        edgecolor='white',
        labelcolor='white'
    )
    plt.title(
        'Corner Kick Setup',
        fontsize=16,
        color='white',
        pad=20,
        fontweight='bold'
    )
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='#22312b')
        print(f"✅ Saved to {save_path}")

    plt.close()  # Always close, don't show windows
    return fig


def visualize_multiple_corners(corners_df, num_corners=6, save_path=None):
    """Visualize multiple corners in a grid"""
    
    num_corners = min(num_corners, len(corners_df))
    
    # Calculate grid size
    cols = 3
    rows = (num_corners + cols - 1) // cols
    
    fig = plt.figure(figsize=(16, 5 * rows), facecolor='#22312b')
    
    for idx in range(num_corners):
        corner = corners_df.iloc[idx]
        freeze_frame = corner['freeze_frame_parsed']
        
        ax = fig.add_subplot(rows, cols, idx + 1)
        
        # Create pitch
        pitch = Pitch(
            pitch_type='statsbomb',
            pitch_color='#22312b',
            line_color='#c7d5cc',
            linewidth=1.5
        )
        pitch.draw(ax=ax)
        
        # Separate players
        attackers = [p for p in freeze_frame if p['teammate']]
        defenders = [p for p in freeze_frame if not p['teammate']]
        
        # Plot players
        if attackers:
            att_x = [p['location'][0] for p in attackers]
            att_y = [p['location'][1] for p in attackers]
            pitch.scatter(att_x, att_y, ax=ax, c='#FF6B6B', s=200,
                         edgecolors='white', linewidths=1, zorder=3)
        
        if defenders:
            def_x = [p['location'][0] for p in defenders]
            def_y = [p['location'][1] for p in defenders]
            pitch.scatter(def_x, def_y, ax=ax, c='#4ECDC4', s=200,
                         edgecolors='white', linewidths=1, zorder=3)
        
        # Plot ball
        ball_x, ball_y = corner['location']
        pitch.scatter(ball_x, ball_y, ax=ax, c='white', s=300,
                     edgecolors='black', linewidths=2, zorder=4)
        
        # Title
        ax.set_title(
            f"Corner {idx+1} - {corner.get('team', 'Unknown')[:15]}",
            fontsize=12, color='white', pad=10
        )
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='#22312b')
        print(f"✅ Saved to {save_path}")

    plt.close()  # Close instead of showing
    return fig


def main():
    print("=" * 60)
    print("TacticAI Corner Kick Visualizer")
    print("=" * 60)

    # Try to load shots with freeze frames first (they have positional data)
    shots_path = 'data/processed/shots_freeze.pkl'
    corners_path = 'data/processed/corners.pkl'

    data_path = None
    data_type = None

    if os.path.exists(shots_path):
        data_path = shots_path
        data_type = "shots"
        print(f"\nLoading shots with freeze frames from {shots_path}...")
        corners = pd.read_pickle(shots_path)
        print(f"✅ Loaded {len(corners)} shots with freeze frames")
    elif os.path.exists(corners_path):
        print(f"\nLoading corners from {corners_path}...")
        corners = pd.read_pickle(corners_path)
        print(f"✅ Loaded {len(corners)} corners")

        # Check if corners have freeze_frame_parsed
        if 'freeze_frame_parsed' not in corners.columns or corners['freeze_frame_parsed'].isna().all():
            print("\n⚠️  Corners don't have freeze frame data (player positions).")
            print("   Using shots with freeze frames for visualization instead...")
            if os.path.exists(shots_path):
                corners = pd.read_pickle(shots_path)
                print(f"✅ Loaded {len(corners)} shots with freeze frames")
                data_type = "shots"
            else:
                print("\n❌ No data with freeze frames available for visualization.")
                print("   Please run: python scripts/download_data.py")
                return
        else:
            data_type = "corners"
    else:
        print(f"\n❌ Data not found: {corners_path} or {shots_path}")
        print("   Please run: python scripts/download_data.py")
        return
    
    # Visualize first item
    item_name = "shot" if data_type == "shots" else "corner"
    print(f"\n1. Visualizing first {item_name}...")
    fig1 = visualize_corner(
        corners.iloc[0],
        save_path=f'visualizations/{item_name}_sample_1.png',
        show=False
    )

    # Visualize grid
    print(f"\n2. Creating grid visualization of 6 {item_name}s...")
    fig2 = visualize_multiple_corners(
        corners,
        num_corners=6,
        save_path=f'visualizations/{item_name}_grid.png'
    )

    # Statistics
    print("\n" + "=" * 60)
    print(f"{item_name.upper()} STATISTICS")
    print("=" * 60)
    
    player_counts = corners['freeze_frame_parsed'].apply(len)
    print(f"Average players: {player_counts.mean():.1f}")
    print(f"Min players: {player_counts.min()}")
    print(f"Max players: {player_counts.max()}")
    
    # Attacker/defender counts
    def count_attackers_defenders(ff):
        att = sum(1 for p in ff if p['teammate'])
        deff = sum(1 for p in ff if not p['teammate'])
        return att, deff
    
    att_def = corners['freeze_frame_parsed'].apply(count_attackers_defenders)
    corners['num_att'] = att_def.apply(lambda x: x[0])
    corners['num_def'] = att_def.apply(lambda x: x[1])
    
    print(f"Average attackers: {corners['num_att'].mean():.1f}")
    print(f"Average defenders: {corners['num_def'].mean():.1f}")
    
    print("\n✅ Visualizations complete!")
    print("   Saved to visualizations/ directory")
    print("=" * 60)


if __name__ == "__main__":
    # Create output directory
    os.makedirs('visualizations', exist_ok=True)
    main()
