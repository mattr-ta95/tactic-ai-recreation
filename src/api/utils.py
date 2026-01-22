"""Utility functions for TacticAI API."""

import torch
import base64
import io
from typing import Dict, List, Optional, Tuple, Any
from torch_geometric.data import Data
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt


def graph_to_dict(graph: Data) -> Dict[str, Any]:
    """
    Convert PyTorch Geometric Data object to JSON-serializable dict.

    Args:
        graph: PyG Data object

    Returns:
        Dictionary with graph structure
    """
    result = {
        'x': graph.x.cpu().numpy().tolist(),
        'edge_index': graph.edge_index.cpu().numpy().tolist(),
        'num_nodes': graph.num_nodes,
    }

    if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
        result['edge_attr'] = graph.edge_attr.cpu().numpy().tolist()
    else:
        result['edge_attr'] = None

    if hasattr(graph, 'y') and graph.y is not None:
        result['y'] = graph.y.item() if graph.y.numel() == 1 else graph.y.cpu().numpy().tolist()
    else:
        result['y'] = None

    if hasattr(graph, 'match_id'):
        result['match_id'] = graph.match_id
    else:
        result['match_id'] = -1

    return result


def dict_to_graph(data: Dict[str, Any], device: str = 'cpu') -> Data:
    """
    Convert dictionary back to PyTorch Geometric Data object.

    Args:
        data: Dictionary with graph structure
        device: Target device

    Returns:
        PyG Data object
    """
    x = torch.tensor(data['x'], dtype=torch.float, device=device)
    edge_index = torch.tensor(data['edge_index'], dtype=torch.long, device=device)

    edge_attr = None
    if data.get('edge_attr'):
        edge_attr = torch.tensor(data['edge_attr'], dtype=torch.float, device=device)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    if data.get('y') is not None:
        if isinstance(data['y'], list):
            graph.y = torch.tensor(data['y'], dtype=torch.long, device=device)
        else:
            graph.y = torch.tensor([data['y']], dtype=torch.long, device=device)

    return graph


def corner_setup_to_graph(
    players: List[Dict],
    processor,
    corner_location: Optional[Tuple[float, float]] = None,
) -> Data:
    """
    Convert API request player data to PyG graph using processor.

    Args:
        players: List of player dicts with x, y, is_teammate, position_role
        processor: CornerKickProcessor instance
        corner_location: Optional corner kick location

    Returns:
        PyG Data object
    """
    import pandas as pd

    # Build freeze_frame format expected by processor
    freeze_frame = []
    for p in players:
        position_id = _role_to_position_id(p.get('position_role'))
        freeze_frame.append({
            'location': [p['x'], p['y']],
            'teammate': p['is_teammate'],
            'position': {'id': position_id},
        })

    # Set default corner location if not provided
    if corner_location is None:
        corner_location = [120.0, 0.0]  # Default: right corner

    # Create fake row for processor
    row = pd.Series({
        'freeze_frame_parsed': freeze_frame,
        'corner_pass_end_location': corner_location,
        'location': corner_location,
    })

    return processor.corner_to_graph(row)


def _role_to_position_id(role: Optional[str]) -> int:
    """
    Map role string to StatsBomb position ID.

    Args:
        role: Role string (GK, DEF, MID, FWD)

    Returns:
        StatsBomb position ID
    """
    if role is None:
        return 10  # Default: midfielder

    role_upper = role.upper() if isinstance(role, str) else str(role).upper()
    mapping = {
        'GK': 1,
        'DEF': 4,
        'MID': 10,
        'FWD': 23,
    }
    return mapping.get(role_upper, 10)


def figure_to_base64(fig: plt.Figure, dpi: int = 100) -> str:
    """
    Convert matplotlib figure to base64-encoded PNG string.

    Args:
        fig: matplotlib Figure object
        dpi: Resolution for PNG output

    Returns:
        Base64-encoded PNG string
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64


def base64_to_image(b64_string: str) -> bytes:
    """
    Convert base64 string back to image bytes.

    Args:
        b64_string: Base64-encoded image string

    Returns:
        Image bytes
    """
    return base64.b64decode(b64_string)
