"""Position optimization page."""

import streamlit as st
import requests
import base64
from io import BytesIO
from PIL import Image

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(page_title="Optimizer - TacticAI", layout="wide")
st.title("Position Optimizer")

st.markdown("""
Optimize attacking player positions to maximize the probability of a target receiver
getting the ball from a corner kick.
""")

# Check API connection
try:
    health = requests.get(f"{API_BASE}/health", timeout=2).json()
    if not health.get('model_loaded'):
        st.error("Model not loaded. Cannot run optimization.")
        st.stop()
except:
    st.error("API not available. Start the API server first.")
    st.stop()

# Sidebar controls
st.sidebar.header("Configuration")

# Fetch corners for selection
@st.cache_data(ttl=60)
def fetch_corners(limit=200):
    """Fetch available corners from API."""
    try:
        resp = requests.get(f"{API_BASE}/corners?limit={limit}", timeout=5)
        return resp.json() if resp.ok else []
    except:
        return []

corners = fetch_corners()

if not corners:
    st.error("No corners available in dataset.")
    st.stop()

# Corner selection
st.sidebar.subheader("Corner Selection")
corner_options = {
    f"#{c['corner_id']} - {c['num_players']} players ({c['num_attackers']}A/{c['num_defenders']}D)": c['corner_id']
    for c in corners
}
selected_label = st.sidebar.selectbox("Select Corner", list(corner_options.keys()))
corner_id = corner_options[selected_label]

# Get corner details
corner_detail = requests.get(f"{API_BASE}/corners/{corner_id}").json()

# Target receiver selection
st.sidebar.subheader("Target Receiver")
receiver_options = [None] + list(range(corner_detail['num_players']))
target_receiver = st.sidebar.selectbox(
    "Target Receiver",
    options=receiver_options,
    format_func=lambda x: "Auto-select best attacker" if x is None else f"Player {x}",
)

# Optimization parameters
st.sidebar.subheader("Optimization Parameters")
num_iterations = st.sidebar.slider("Iterations", 10, 200, 50, step=10)
learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1, step=0.01)
constraint_penalty = st.sidebar.slider("Constraint Penalty", 0.0, 50.0, 10.0, step=1.0)
min_spacing = st.sidebar.slider("Min Player Spacing (m)", 0.5, 3.0, 1.0, step=0.1)

# Main content
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader(f"Corner #{corner_id}")

    # Display corner info
    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.metric("Total Players", corner_detail['num_players'])
    info_col2.metric("Attackers", corner_detail['num_attackers'])
    info_col3.metric("Defenders", corner_detail['num_defenders'])

    if corner_detail.get('has_label'):
        st.info(f"Ground truth receiver: Player {corner_detail['label']}")

    # Run optimization button
    if st.button("Run Optimization", type="primary", use_container_width=True):
        with st.spinner(f"Optimizing positions ({num_iterations} iterations)..."):
            payload = {
                "corner_id": corner_id,
                "target_receiver": target_receiver,
                "num_iterations": num_iterations,
                "learning_rate": learning_rate,
                "constraint_penalty": constraint_penalty,
                "min_spacing": min_spacing,
            }

            try:
                resp = requests.post(f"{API_BASE}/optimize", json=payload, timeout=120)

                if resp.ok:
                    result = resp.json()
                    st.session_state['optimization_result'] = result
                    st.success("Optimization complete!")
                else:
                    st.error(f"Optimization failed: {resp.text}")
            except requests.exceptions.Timeout:
                st.error("Optimization timed out. Try reducing iterations.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

with col2:
    st.subheader("Receiver Predictions")

    if st.button("Get Current Predictions"):
        with st.spinner("Predicting..."):
            resp = requests.post(f"{API_BASE}/predict", json={"corner_id": corner_id}, timeout=10)
            if resp.ok:
                preds = resp.json()
                st.session_state['predictions'] = preds
            else:
                st.error("Prediction failed")

    # Display predictions if available
    if 'predictions' in st.session_state:
        preds = st.session_state['predictions']
        st.markdown(f"**Top Receiver:** Player {preds['top_receiver']} ({preds['top_probability']:.1%})")

        # Show top 5 attackers
        attackers = [p for p in preds['predictions'] if p['is_attacker']][:5]
        for p in attackers:
            st.progress(p['probability'], text=f"Player {p['player_index']}: {p['probability']:.1%}")

# Display optimization results
if 'optimization_result' in st.session_state:
    result = st.session_state['optimization_result']

    st.markdown("---")
    st.subheader("Optimization Results")

    # Metrics
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    metric_col1.metric(
        "Original Probability",
        f"{result['original_probability']:.1%}",
    )
    metric_col2.metric(
        "Optimized Probability",
        f"{result['optimized_probability']:.1%}",
        f"+{result['improvement_percentage']:.1f}%",
    )
    metric_col3.metric(
        "Iterations",
        result['num_iterations'],
    )
    metric_col4.metric(
        "Converged",
        "Yes" if result['converged'] else "No",
    )

    # Visualization
    if result.get('visualization_base64'):
        st.subheader("Position Comparison")
        img_data = base64.b64decode(result['visualization_base64'])
        img = Image.open(BytesIO(img_data))
        st.image(img, use_container_width=True, caption="Original (left) vs Optimized (right)")

    # Position changes table
    if result['position_changes']:
        st.subheader("Position Changes")

        changes_data = []
        for change in result['position_changes']:
            changes_data.append({
                "Player": change['player_index'],
                "Original X": f"{change['original'][0]:.1f}",
                "Original Y": f"{change['original'][1]:.1f}",
                "Optimized X": f"{change['optimized'][0]:.1f}",
                "Optimized Y": f"{change['optimized'][1]:.1f}",
                "Movement (m)": f"{change['movement_distance']:.1f}",
            })

        st.dataframe(changes_data, use_container_width=True, hide_index=True)
    else:
        st.info("No significant position changes - setup may already be optimal")

# Clear results button
if 'optimization_result' in st.session_state or 'predictions' in st.session_state:
    if st.sidebar.button("Clear Results"):
        if 'optimization_result' in st.session_state:
            del st.session_state['optimization_result']
        if 'predictions' in st.session_state:
            del st.session_state['predictions']
        st.rerun()
