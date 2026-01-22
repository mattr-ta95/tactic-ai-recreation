"""Dataset exploration page."""

import streamlit as st
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mplsoccer import Pitch

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(page_title="Explorer - TacticAI", layout="wide")
st.title("Dataset Explorer")

st.markdown("""
Browse and visualize corner kicks from the training dataset.
""")

# Check API connection
try:
    health = requests.get(f"{API_BASE}/health", timeout=2).json()
    dataset_size = health.get('dataset_size', 0)
except:
    st.error("API not available. Start the API server first.")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")
min_players = st.sidebar.slider("Minimum Players", 10, 25, 15)
max_players = st.sidebar.slider("Maximum Players", 15, 30, 25)
show_labeled_only = st.sidebar.checkbox("Labeled Only", value=False)

# Pagination
st.sidebar.subheader("Pagination")
page_size = st.sidebar.selectbox("Items per page", [10, 20, 50], index=0)
page = st.sidebar.number_input("Page", min_value=1, value=1)

# Fetch corners
@st.cache_data(ttl=60)
def fetch_all_corners(total_size):
    """Fetch all corners from API."""
    try:
        resp = requests.get(f"{API_BASE}/corners?limit={total_size}", timeout=30)
        return resp.json() if resp.ok else []
    except:
        return []

all_corners = fetch_all_corners(dataset_size)

if not all_corners:
    st.error("No corners available in dataset.")
    st.stop()

# Apply filters
filtered = [
    c for c in all_corners
    if min_players <= c['num_players'] <= max_players
]
if show_labeled_only:
    filtered = [c for c in filtered if c['has_label']]

# Dataset statistics
st.subheader("Dataset Statistics")
stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
stat_col1.metric("Total Corners", len(all_corners))
stat_col2.metric("Filtered", len(filtered))
stat_col3.metric("With Labels", sum(1 for c in all_corners if c['has_label']))
stat_col4.metric("Avg Players", f"{sum(c['num_players'] for c in all_corners) / len(all_corners):.1f}")

# Distribution charts
st.subheader("Player Distribution")
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    # Player count distribution
    player_counts = [c['num_players'] for c in all_corners]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(player_counts, bins=range(min(player_counts), max(player_counts) + 2),
            edgecolor='white', color='#4ECDC4')
    ax.set_xlabel("Number of Players")
    ax.set_ylabel("Count")
    ax.set_title("Players per Corner")
    st.pyplot(fig)
    plt.close(fig)

with chart_col2:
    # Attacker/defender ratio
    ratios = [c['num_attackers'] / c['num_players'] for c in all_corners]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(ratios, bins=20, edgecolor='white', color='#FF6B6B')
    ax.set_xlabel("Attacker Ratio")
    ax.set_ylabel("Count")
    ax.set_title("Attacker/Total Ratio Distribution")
    st.pyplot(fig)
    plt.close(fig)

st.markdown("---")

# Corner list with pagination
st.subheader(f"Corner List ({len(filtered)} results)")

start_idx = (page - 1) * page_size
end_idx = min(start_idx + page_size, len(filtered))
page_corners = filtered[start_idx:end_idx]

total_pages = (len(filtered) + page_size - 1) // page_size
st.markdown(f"Showing {start_idx + 1}-{end_idx} of {len(filtered)} (Page {page}/{total_pages})")

for corner in page_corners:
    with st.expander(f"Corner #{corner['corner_id']} - {corner['num_players']} players"):
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown(f"**Match ID:** {corner['match_id']}")
            st.markdown(f"**Players:** {corner['num_players']}")
            st.markdown(f"**Attackers:** {corner['num_attackers']}")
            st.markdown(f"**Defenders:** {corner['num_defenders']}")
            st.markdown(f"**Has Label:** {'Yes' if corner['has_label'] else 'No'}")

        with col2:
            # Visualize button
            if st.button(f"Visualize", key=f"viz_{corner['corner_id']}"):
                with st.spinner("Loading visualization..."):
                    resp = requests.get(
                        f"{API_BASE}/corners/{corner['corner_id']}?include_graph=true",
                        timeout=10
                    )

                    if resp.ok:
                        data = resp.json()
                        graph = data['graph']

                        # Create pitch plot
                        fig, ax = plt.subplots(figsize=(10, 7))
                        pitch = Pitch(
                            pitch_type='statsbomb',
                            pitch_color='#22312b',
                            line_color='#c7d5cc',
                            goal_type='box',
                        )
                        pitch.draw(ax=ax)

                        # Extract and denormalize positions
                        x_vals = [p[0] * 120 for p in graph['x']]
                        y_vals = [p[1] * 80 for p in graph['x']]
                        teams = [p[2] > 0.5 for p in graph['x']]

                        # Plot players
                        for i, (x, y, is_att) in enumerate(zip(x_vals, y_vals, teams)):
                            # Check if this is the receiver (if labeled)
                            if data.get('label') is not None and i == data['label']:
                                color = '#FFD93D'
                                marker = '*'
                                size = 300
                            else:
                                color = '#FF6B6B' if is_att else '#4ECDC4'
                                marker = 'o'
                                size = 200

                            ax.scatter(x, y, c=color, s=size, marker=marker, zorder=3,
                                       edgecolors='white', linewidths=2)
                            ax.annotate(str(i), (x, y), ha='center', va='center',
                                        fontsize=8, fontweight='bold',
                                        color='white' if is_att else 'black', zorder=4)

                        # Title with receiver info
                        title = f"Corner #{corner['corner_id']}"
                        if data.get('label') is not None:
                            title += f" - Receiver: Player {data['label']}"
                        ax.set_title(title, fontsize=12, fontweight='bold')

                        # Legend
                        from matplotlib.patches import Patch
                        legend_elements = [
                            Patch(facecolor='#FF6B6B', edgecolor='white', label='Attackers'),
                            Patch(facecolor='#4ECDC4', edgecolor='white', label='Defenders'),
                        ]
                        if data.get('label') is not None:
                            legend_elements.append(
                                Patch(facecolor='#FFD93D', edgecolor='white', label='Receiver')
                            )
                        ax.legend(handles=legend_elements, loc='upper left')

                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.error("Failed to load corner data")

# Page navigation
nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
with nav_col1:
    if page > 1:
        if st.button("Previous Page"):
            st.session_state['page'] = page - 1
            st.rerun()
with nav_col3:
    if page < total_pages:
        if st.button("Next Page"):
            st.session_state['page'] = page + 1
            st.rerun()
