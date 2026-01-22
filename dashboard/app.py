"""TacticAI Dashboard - Streamlit entry point."""

import streamlit as st
import requests

# Page configuration
st.set_page_config(
    page_title="TacticAI Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and description
st.title("TacticAI - Corner Kick Tactical Analysis")

st.markdown("""
## What is TacticAI?

TacticAI is a **Graph Neural Network** system that analyzes soccer corner kicks.
It was trained on **4,239 corner kick scenarios** from Premier League matches (StatsBomb data).

### The Problem

In a corner kick, the attacking team wants to deliver the ball to a specific player.
But with 20+ players crowded near the goal, which positions give the best chance of success?

### What TacticAI Does

1. **Receiver Prediction**: Given player positions, predict which attacker is most likely to receive the ball
2. **Position Optimization**: Suggest how attackers should reposition to maximize a target player's receiving probability

### How It Works

- Players are represented as **nodes** in a graph
- Nearby players are connected by **edges** (within 5 meters)
- A Graph Attention Network learns patterns from 4,239 real corner kicks
- Gradient-based optimization adjusts positions while respecting constraints (pitch bounds, player spacing)

---

### Dashboard Pages

| Page | Description |
|------|-------------|
| **Optimizer** | Select a corner from the training data and optimize positions |
| **Explorer** | Browse the training dataset and visualize corner setups |
| **Custom Corner** | Build your own corner setup and get predictions |

---
""")

# API connection status
st.subheader("System Status")

API_BASE = "http://localhost:8000/api/v1"

try:
    resp = requests.get(f"{API_BASE}/health", timeout=3)
    if resp.ok:
        data = resp.json()
        col1, col2, col3, col4 = st.columns(4)

        status_emoji = "" if data['status'] == 'healthy' else ""
        col1.metric("API Status", f"{status_emoji} {data['status'].title()}")
        col2.metric("Model Loaded", "" if data['model_loaded'] else "")
        col3.metric("Dataset Size", f"{data['dataset_size']} corners")
        col4.metric("Device", data['device'].upper())

        if data['status'] == 'healthy':
            st.success("System ready! Navigate to **Optimizer** or **Explorer** in the sidebar.")
        else:
            st.warning("System in degraded state. Some features may not work.")
    else:
        st.error(f"API returned error: {resp.status_code}")
except requests.exceptions.ConnectionError:
    st.error("Cannot connect to API server.")
    st.markdown("""
    **To start the API server, run:**
    ```bash
    cd tacticai-project
    uvicorn src.api.main:app --reload --port 8000
    ```
    """)
except Exception as e:
    st.error(f"Error checking API status: {str(e)}")

# Quick links
st.markdown("---")
st.subheader("Get Started")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    **Optimizer**

    Select corners from training data and optimize positions.
    """)

with col2:
    st.markdown("""
    **Explorer**

    Browse training data and visualize setups.
    """)

with col3:
    st.markdown("""
    **Custom Corner**

    Build your own corner and get predictions.
    """)

with col4:
    st.markdown("""
    **API Docs**

    [localhost:8000/docs](http://localhost:8000/docs)
    """)

# Footer
st.markdown("---")
st.markdown(
    "*Powered by Graph Neural Networks and PyTorch. "
    "Recreation of DeepMind's TacticAI for educational purposes.*"
)
