"""Receiver prediction endpoints."""

from fastapi import APIRouter, HTTPException
import torch
import torch.nn.functional as F

from ..schemas import PredictRequest, PredictResponse, ReceiverProbability
from ..utils import corner_setup_to_graph
from ..state import get_state

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
async def predict_receiver(request: PredictRequest):
    """
    Predict receiver probabilities for a corner kick setup.

    Can use either a corner_id from the dataset or a custom corner_setup
    with player positions. Returns probabilities for all players,
    sorted by likelihood.
    """
    state = get_state()

    if state['model'] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Get graph from request
    if request.corner_id is not None:
        if request.corner_id < 0 or request.corner_id >= len(state['dataset']):
            raise HTTPException(
                status_code=404,
                detail=f"Corner {request.corner_id} not found"
            )
        graph = state['dataset'][request.corner_id].clone()
    elif request.corner_setup:
        try:
            graph = corner_setup_to_graph(
                [p.model_dump() for p in request.corner_setup.players],
                state['processor'],
                request.corner_setup.corner_location,
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to create graph from corner setup: {str(e)}"
            )
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either corner_id or corner_setup"
        )

    # Move to device and predict
    device = state['device']
    graph = graph.to(device)

    with torch.no_grad():
        batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)
        edge_attr = graph.edge_attr if hasattr(graph, 'edge_attr') else None
        logits = state['model'](graph.x, graph.edge_index, batch, edge_attr=edge_attr)
        if isinstance(logits, dict):
            logits = logits['receiver']
        probs = F.softmax(logits, dim=0)

    # Build response
    predictions = []
    for i in range(graph.num_nodes):
        x_pos = graph.x[i, 0].item() * 120  # Denormalize
        y_pos = graph.x[i, 1].item() * 80
        is_attacker = graph.x[i, 2].item() > 0.5

        predictions.append(ReceiverProbability(
            player_index=i,
            probability=round(probs[i].item(), 4),
            is_attacker=is_attacker,
            position=(round(x_pos, 1), round(y_pos, 1)),
        ))

    # Sort by probability (descending)
    predictions.sort(key=lambda p: p.probability, reverse=True)

    return PredictResponse(
        success=True,
        predictions=predictions,
        top_receiver=predictions[0].player_index,
        top_probability=predictions[0].probability,
    )


@router.post("/predict/top")
async def predict_top_receivers(
    request: PredictRequest,
    top_n: int = 5,
):
    """
    Predict top N most likely receivers.

    Simplified endpoint that returns only the top N attackers
    by receiver probability.
    """
    # Use the full prediction endpoint
    full_response = await predict_receiver(request)

    # Filter to top N attackers
    attackers = [p for p in full_response.predictions if p.is_attacker][:top_n]

    return {
        "success": True,
        "top_receivers": attackers,
        "most_likely": attackers[0] if attackers else None,
    }
