"""Visualization components for Streamlit dashboard."""

import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from typing import Dict, List, Optional
import base64
from io import BytesIO
from PIL import Image


def display_base64_image(b64_string: str, caption: str = ""):
    """Display a base64-encoded image in Streamlit."""
    img_data = base64.b64decode(b64_string)
    img = Image.open(BytesIO(img_data))
    st.image(img, caption=caption, use_container_width=True)


def plot_corner_setup(
    graph_data: Dict,
    target_receiver: Optional[int] = None,
    title: str = "Corner Setup",
) -> plt.Figure:
    """
    Create matplotlib figure for corner setup from graph dict.

    Args:
        graph_data: Dictionary with 'x' containing node features
        target_receiver: Optional index of target receiver to highlight
        title: Plot title

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    pitch = Pitch(
        pitch_type='statsbomb',
        pitch_color='#22312b',
        line_color='#c7d5cc',
        goal_type='box',
    )
    pitch.draw(ax=ax)

    # Extract positions (denormalize from [0,1] to meters)
    x_vals = [p[0] * 120 for p in graph_data['x']]
    y_vals = [p[1] * 80 for p in graph_data['x']]
    teams = [p[2] > 0.5 for p in graph_data['x']]

    for i, (x, y, is_att) in enumerate(zip(x_vals, y_vals, teams)):
        if target_receiver is not None and i == target_receiver:
            color = '#FFD93D'  # Yellow for target
            marker = '*'
            size = 300
        else:
            color = '#FF6B6B' if is_att else '#4ECDC4'
            marker = 'o'
            size = 200

        ax.scatter(x, y, c=color, s=size, marker=marker, zorder=3,
                   edgecolors='white', linewidths=2)
        ax.annotate(str(i), (x, y), ha='center', va='center', fontsize=8,
                    fontweight='bold', color='white' if is_att else 'black', zorder=4)

    ax.set_title(title, fontsize=12, fontweight='bold')
    return fig


def create_probability_bars(predictions: List[Dict], top_n: int = 5):
    """
    Display probability bars for top receivers.

    Args:
        predictions: List of prediction dicts with player_index, probability, is_attacker
        top_n: Number of top receivers to show
    """
    attackers = [p for p in predictions if p.get('is_attacker', True)][:top_n]

    st.markdown("**Top Receivers by Probability**")
    for p in attackers:
        col1, col2 = st.columns([4, 1])
        col1.progress(p['probability'])
        col2.write(f"P{p['player_index']}: {p['probability']:.1%}")


def create_position_change_table(changes: List[Dict]):
    """
    Display position changes as a formatted table.

    Args:
        changes: List of position change dicts
    """
    if not changes:
        st.info("No significant position changes")
        return

    data = []
    for change in changes:
        data.append({
            "Player": change['player_index'],
            "Original": f"({change['original'][0]:.1f}, {change['original'][1]:.1f})",
            "Optimized": f"({change['optimized'][0]:.1f}, {change['optimized'][1]:.1f})",
            "Movement": f"{change['movement_distance']:.1f}m",
        })

    st.dataframe(data, use_container_width=True, hide_index=True)
