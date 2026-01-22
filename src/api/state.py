"""Shared application state for TacticAI API."""

# Global state dictionary - populated by lifespan handler in main.py
_state = {
    'model': None,
    'config': None,
    'processor': None,
    'optimizer': None,
    'visualizer': None,
    'dataset': [],
    'device': 'cpu',
}


def get_state():
    """Get the global application state."""
    return _state
