"""Corner kick data endpoints."""

from fastapi import APIRouter, HTTPException, Query
from typing import List

from ..schemas import CornerSummary, CornerDetail
from ..utils import graph_to_dict
from ..state import get_state

router = APIRouter()


@router.get("/corners", response_model=List[CornerSummary])
async def list_corners(
    skip: int = Query(0, ge=0, description="Number of corners to skip"),
    limit: int = Query(50, ge=1, le=10000, description="Maximum corners to return"),
):
    """
    List available corner kicks in the dataset.

    Returns summary information for each corner including player counts
    and whether receiver labels are available.
    """
    state = get_state()
    dataset = state['dataset']

    if not dataset:
        return []

    summaries = []
    end_idx = min(skip + limit, len(dataset))

    for i in range(skip, end_idx):
        graph = dataset[i]
        num_attackers = int((graph.x[:, 2] > 0.5).sum().item())

        summaries.append(CornerSummary(
            corner_id=i,
            match_id=getattr(graph, 'match_id', -1),
            num_players=graph.num_nodes,
            num_attackers=num_attackers,
            num_defenders=graph.num_nodes - num_attackers,
            has_label=hasattr(graph, 'y') and graph.y is not None,
        ))

    return summaries


@router.get("/corners/{corner_id}", response_model=CornerDetail)
async def get_corner(
    corner_id: int,
    include_graph: bool = Query(False, description="Include full graph data"),
):
    """
    Get detailed information for a specific corner kick.

    Optionally includes the full graph structure with node features,
    edge indices, and edge attributes.
    """
    state = get_state()
    dataset = state['dataset']

    if corner_id < 0 or corner_id >= len(dataset):
        raise HTTPException(
            status_code=404,
            detail=f"Corner {corner_id} not found. Valid range: 0-{len(dataset)-1}"
        )

    graph = dataset[corner_id]
    num_attackers = int((graph.x[:, 2] > 0.5).sum().item())

    response = CornerDetail(
        corner_id=corner_id,
        match_id=getattr(graph, 'match_id', -1),
        num_players=graph.num_nodes,
        num_attackers=num_attackers,
        num_defenders=graph.num_nodes - num_attackers,
        has_label=hasattr(graph, 'y') and graph.y is not None,
        label=graph.y.item() if hasattr(graph, 'y') and graph.y is not None else None,
    )

    if include_graph:
        response.graph = graph_to_dict(graph)

    return response


@router.get("/corners/count")
async def get_corner_count():
    """Get the total number of corners in the dataset."""
    state = get_state()
    return {"total": len(state['dataset'])}
