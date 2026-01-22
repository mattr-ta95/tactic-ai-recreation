"""Position optimization endpoints."""

from fastapi import APIRouter, HTTPException

from ..schemas import OptimizationRequest, OptimizationResponse, PositionChange
from ..utils import corner_setup_to_graph, figure_to_base64
from ..state import get_state

router = APIRouter()


@router.post("/optimize", response_model=OptimizationResponse)
async def optimize_positions(request: OptimizationRequest):
    """
    Optimize attacker positions to maximize receiver probability.

    Uses gradient-based optimization to adjust attacking player positions
    while respecting pitch boundaries and player spacing constraints.
    Returns the optimized positions along with a visualization.
    """
    state = get_state()

    if state['optimizer'] is None:
        raise HTTPException(status_code=503, detail="Optimizer not initialized")

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

    # Validate target receiver if specified
    if request.target_receiver is not None:
        if request.target_receiver < 0 or request.target_receiver >= graph.num_nodes:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid target_receiver: {request.target_receiver}. "
                       f"Valid range: 0-{graph.num_nodes - 1}"
            )

    # Run optimization
    try:
        result = state['optimizer'].optimize_positions(
            graph.to(state['device']),
            target_receiver=request.target_receiver,
            num_iterations=request.num_iterations,
            learning_rate=request.learning_rate,
            constraint_penalty=request.constraint_penalty,
            min_spacing=request.min_spacing,
            verbose=False,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Optimization failed: {str(e)}"
        )

    # Build position changes list
    changes = []
    for player_idx, (ox, oy, nx, ny) in result.position_changes.items():
        dist = ((nx - ox) ** 2 + (ny - oy) ** 2) ** 0.5
        changes.append(PositionChange(
            player_index=player_idx,
            original=(round(ox, 1), round(oy, 1)),
            optimized=(round(nx, 1), round(ny, 1)),
            movement_distance=round(dist, 2),
        ))

    # Sort by movement distance (largest first)
    changes.sort(key=lambda c: c.movement_distance, reverse=True)

    # Generate visualization
    viz_b64 = None
    if state['visualizer']:
        try:
            fig = state['visualizer'].plot_optimization_result(result)
            viz_b64 = figure_to_base64(fig)
        except Exception:
            pass  # Visualization is optional

    return OptimizationResponse(
        success=True,
        target_receiver=result.target_receiver,
        original_probability=round(result.original_probability, 4),
        optimized_probability=round(result.optimized_probability, 4),
        improvement_percentage=round(result.improvement_percentage, 1),
        position_changes=changes,
        num_iterations=result.num_iterations,
        converged=result.converged,
        visualization_base64=viz_b64,
    )


@router.post("/optimize/quick")
async def quick_optimize(
    corner_id: int,
    target_receiver: int = None,
):
    """
    Quick optimization with default parameters.

    Simplified endpoint that uses default optimization settings.
    """
    request = OptimizationRequest(
        corner_id=corner_id,
        target_receiver=target_receiver,
        num_iterations=30,  # Fewer iterations for speed
        learning_rate=0.1,
        constraint_penalty=10.0,
        min_spacing=1.0,
    )
    return await optimize_positions(request)


@router.post("/analyze/sensitivity")
async def analyze_sensitivity(
    corner_id: int,
    target_receiver: int,
    num_samples: int = 50,
):
    """
    Analyze which attackers have most impact on target receiver probability.

    Perturbs each attacker's position randomly and measures the average
    change in the target receiver's probability.
    """
    state = get_state()

    if state['optimizer'] is None:
        raise HTTPException(status_code=503, detail="Optimizer not initialized")

    if corner_id < 0 or corner_id >= len(state['dataset']):
        raise HTTPException(status_code=404, detail=f"Corner {corner_id} not found")

    graph = state['dataset'][corner_id].clone().to(state['device'])

    if target_receiver < 0 or target_receiver >= graph.num_nodes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid target_receiver: {target_receiver}"
        )

    try:
        sensitivity = state['optimizer'].analyze_sensitivity(
            graph,
            target_receiver=target_receiver,
            num_samples=num_samples,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Sensitivity analysis failed: {str(e)}"
        )

    # Format response
    results = [
        {"player_index": idx, "sensitivity_score": round(score, 4)}
        for idx, score in sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)
    ]

    return {
        "success": True,
        "corner_id": corner_id,
        "target_receiver": target_receiver,
        "sensitivities": results,
    }
