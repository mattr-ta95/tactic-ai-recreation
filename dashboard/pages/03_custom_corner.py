"""Custom corner builder page."""

import streamlit as st
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import json

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(page_title="Custom Corner - TacticAI", layout="wide")
st.title("Custom Corner Builder")

st.markdown("""
### What is TacticAI?

TacticAI uses **Graph Neural Networks** to analyze corner kick scenarios:

1. **Receiver Prediction**: Given player positions, predict which attacker is most likely to receive the ball
2. **Position Optimization**: Adjust attacking positions to maximize a target player's receiving probability

---

### Build Your Own Corner

Create a custom corner kick setup by placing players on the pitch. The model will predict
receiver probabilities and suggest optimal positions.
""")

# Check API
try:
    health = requests.get(f"{API_BASE}/health", timeout=2).json()
    if not health.get('model_loaded'):
        st.error("Model not loaded")
        st.stop()
except:
    st.error("API not available. Start with: python3 -m uvicorn src.api.main:app --reload --port 8000")
    st.stop()

# Initialize session state for players
if 'custom_players' not in st.session_state:
    # Default setup: 5 attackers, 6 defenders near goal
    st.session_state.custom_players = [
        # Attackers (red)
        {"x": 108.0, "y": 40.0, "is_teammate": True, "role": "FWD"},
        {"x": 105.0, "y": 35.0, "is_teammate": True, "role": "FWD"},
        {"x": 110.0, "y": 45.0, "is_teammate": True, "role": "MID"},
        {"x": 103.0, "y": 30.0, "is_teammate": True, "role": "MID"},
        {"x": 106.0, "y": 50.0, "is_teammate": True, "role": "DEF"},
        # Defenders (teal)
        {"x": 115.0, "y": 40.0, "is_teammate": False, "role": "GK"},
        {"x": 112.0, "y": 35.0, "is_teammate": False, "role": "DEF"},
        {"x": 112.0, "y": 45.0, "is_teammate": False, "role": "DEF"},
        {"x": 109.0, "y": 38.0, "is_teammate": False, "role": "DEF"},
        {"x": 109.0, "y": 42.0, "is_teammate": False, "role": "DEF"},
        {"x": 107.0, "y": 32.0, "is_teammate": False, "role": "MID"},
    ]

# Sidebar: Player editor
st.sidebar.header("Player Editor")

# Add player
st.sidebar.subheader("Add Player")
new_x = st.sidebar.number_input("X Position (0-120)", 90.0, 120.0, 108.0, 1.0)
new_y = st.sidebar.number_input("Y Position (0-80)", 0.0, 80.0, 40.0, 1.0)
new_team = st.sidebar.selectbox("Team", ["Attacker", "Defender"])
new_role = st.sidebar.selectbox("Role", ["FWD", "MID", "DEF", "GK"])

if st.sidebar.button("Add Player"):
    st.session_state.custom_players.append({
        "x": new_x,
        "y": new_y,
        "is_teammate": new_team == "Attacker",
        "role": new_role,
    })
    st.rerun()

# Remove player
if st.session_state.custom_players:
    st.sidebar.subheader("Remove Player")
    remove_idx = st.sidebar.selectbox(
        "Select Player",
        range(len(st.session_state.custom_players)),
        format_func=lambda i: f"Player {i} ({'A' if st.session_state.custom_players[i]['is_teammate'] else 'D'}) at ({st.session_state.custom_players[i]['x']:.0f}, {st.session_state.custom_players[i]['y']:.0f})"
    )
    if st.sidebar.button("Remove Selected"):
        st.session_state.custom_players.pop(remove_idx)
        st.rerun()

# Reset button
if st.sidebar.button("Reset to Default"):
    del st.session_state.custom_players
    st.rerun()

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Current Setup")

    # Draw pitch with players
    fig, ax = plt.subplots(figsize=(12, 8))
    pitch = Pitch(
        pitch_type='statsbomb',
        pitch_color='#22312b',
        line_color='#c7d5cc',
        goal_type='box',
    )
    pitch.draw(ax=ax)

    # Plot players
    for i, p in enumerate(st.session_state.custom_players):
        color = '#FF6B6B' if p['is_teammate'] else '#4ECDC4'
        ax.scatter(p['x'], p['y'], c=color, s=250, zorder=3,
                   edgecolors='white', linewidths=2)
        ax.annotate(str(i), (p['x'], p['y']), ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    color='white' if p['is_teammate'] else 'black', zorder=4)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', edgecolor='white', label='Attackers'),
        Patch(facecolor='#4ECDC4', edgecolor='white', label='Defenders'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    num_att = sum(1 for p in st.session_state.custom_players if p['is_teammate'])
    num_def = len(st.session_state.custom_players) - num_att
    ax.set_title(f"Custom Corner Setup ({num_att} attackers, {num_def} defenders)", fontsize=12)

    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.subheader("Player List")

    attackers = [(i, p) for i, p in enumerate(st.session_state.custom_players) if p['is_teammate']]
    defenders = [(i, p) for i, p in enumerate(st.session_state.custom_players) if not p['is_teammate']]

    st.markdown("**Attackers**")
    for i, p in attackers:
        st.markdown(f"- Player {i}: ({p['x']:.0f}, {p['y']:.0f}) - {p['role']}")

    st.markdown("**Defenders**")
    for i, p in defenders:
        st.markdown(f"- Player {i}: ({p['x']:.0f}, {p['y']:.0f}) - {p['role']}")

st.markdown("---")

# Analysis buttons
analysis_col1, analysis_col2 = st.columns(2)

with analysis_col1:
    if st.button("Predict Receiver", type="primary", use_container_width=True):
        if len(st.session_state.custom_players) < 5:
            st.error("Need at least 5 players")
        else:
            with st.spinner("Predicting..."):
                payload = {
                    "corner_setup": {
                        "players": [
                            {"x": p['x'], "y": p['y'], "is_teammate": p['is_teammate'], "position_role": p['role']}
                            for p in st.session_state.custom_players
                        ],
                        "corner_location": [120.0, 0.0]
                    }
                }

                resp = requests.post(f"{API_BASE}/predict", json=payload, timeout=30)

                if resp.ok:
                    result = resp.json()
                    st.session_state['custom_predictions'] = result
                    st.success("Prediction complete!")
                else:
                    st.error(f"Failed: {resp.text}")

with analysis_col2:
    target_options = [i for i, p in enumerate(st.session_state.custom_players) if p['is_teammate']]
    if target_options:
        target = st.selectbox("Target Receiver for Optimization", target_options,
                              format_func=lambda i: f"Player {i} (Attacker)")

        if st.button("Optimize Positions", use_container_width=True):
            with st.spinner("Optimizing..."):
                payload = {
                    "corner_setup": {
                        "players": [
                            {"x": p['x'], "y": p['y'], "is_teammate": p['is_teammate'], "position_role": p['role']}
                            for p in st.session_state.custom_players
                        ],
                        "corner_location": [120.0, 0.0]
                    },
                    "target_receiver": target,
                    "num_iterations": 50,
                }

                resp = requests.post(f"{API_BASE}/optimize", json=payload, timeout=60)

                if resp.ok:
                    result = resp.json()
                    st.session_state['custom_optimization'] = result
                    st.success("Optimization complete!")
                else:
                    st.error(f"Failed: {resp.text}")

# Display results
if 'custom_predictions' in st.session_state:
    st.subheader("Receiver Predictions")
    preds = st.session_state['custom_predictions']

    st.markdown(f"**Most Likely Receiver:** Player {preds['top_receiver']} ({preds['top_probability']:.1%})")

    # Show top attackers
    attacker_preds = [p for p in preds['predictions'] if p['is_attacker']][:5]
    for p in attacker_preds:
        st.progress(p['probability'], text=f"Player {p['player_index']}: {p['probability']:.1%}")

if 'custom_optimization' in st.session_state:
    st.subheader("Optimization Results")
    opt = st.session_state['custom_optimization']

    m1, m2, m3 = st.columns(3)
    m1.metric("Original", f"{opt['original_probability']:.1%}")
    m2.metric("Optimized", f"{opt['optimized_probability']:.1%}", f"+{opt['improvement_percentage']:.1f}%")
    m3.metric("Converged", "Yes" if opt['converged'] else "No")

    if opt.get('visualization_base64'):
        import base64
        from io import BytesIO
        from PIL import Image

        img_data = base64.b64decode(opt['visualization_base64'])
        img = Image.open(BytesIO(img_data))
        st.image(img, use_container_width=True)

    if opt['position_changes']:
        st.markdown("**Suggested Position Changes:**")
        for change in opt['position_changes']:
            st.markdown(f"- Player {change['player_index']}: "
                        f"({change['original'][0]:.1f}, {change['original'][1]:.1f}) → "
                        f"({change['optimized'][0]:.1f}, {change['optimized'][1]:.1f}) "
                        f"[{change['movement_distance']:.1f}m]")
