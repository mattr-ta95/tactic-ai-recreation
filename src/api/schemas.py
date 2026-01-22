"""Pydantic models for TacticAI API request/response validation."""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum


# --- Enums ---

class PlayerRole(str, Enum):
    """Player position roles."""
    GK = "GK"
    DEF = "DEF"
    MID = "MID"
    FWD = "FWD"


# --- Request Models ---

class PlayerPosition(BaseModel):
    """Single player in corner kick setup."""
    x: float = Field(..., ge=0, le=120, description="X position in meters (0-120)")
    y: float = Field(..., ge=0, le=80, description="Y position in meters (0-80)")
    is_teammate: bool = Field(..., description="True if attacking team")
    position_role: Optional[PlayerRole] = Field(None, description="Player role: GK/DEF/MID/FWD")


class CornerSetupRequest(BaseModel):
    """Request format for custom corner kick data."""
    players: List[PlayerPosition] = Field(
        ...,
        min_length=5,
        max_length=30,
        description="List of players with positions"
    )
    corner_location: Optional[Tuple[float, float]] = Field(
        None,
        description="Corner kick location (x, y) in meters"
    )


class OptimizationRequest(BaseModel):
    """Request for position optimization."""
    corner_id: Optional[int] = Field(None, description="Corner ID from dataset")
    corner_setup: Optional[CornerSetupRequest] = Field(
        None,
        description="Custom corner setup (alternative to corner_id)"
    )
    target_receiver: Optional[int] = Field(
        None,
        description="Target receiver player index (auto-select if not specified)"
    )
    num_iterations: int = Field(50, ge=1, le=200, description="Optimization iterations")
    learning_rate: float = Field(0.1, gt=0, le=1.0, description="Learning rate")
    constraint_penalty: float = Field(10.0, ge=0, description="Constraint penalty weight")
    min_spacing: float = Field(1.0, ge=0, description="Minimum player spacing in meters")


class PredictRequest(BaseModel):
    """Request for receiver prediction."""
    corner_id: Optional[int] = Field(None, description="Corner ID from dataset")
    corner_setup: Optional[CornerSetupRequest] = Field(
        None,
        description="Custom corner setup (alternative to corner_id)"
    )


# --- Response Models ---

class ReceiverProbability(BaseModel):
    """Probability prediction for a single player."""
    player_index: int = Field(..., description="Player index in graph")
    probability: float = Field(..., ge=0, le=1, description="Receiver probability")
    is_attacker: bool = Field(..., description="True if attacking team")
    position: Tuple[float, float] = Field(..., description="Position (x, y) in meters")


class PredictResponse(BaseModel):
    """Response for receiver prediction."""
    success: bool = Field(..., description="Whether prediction succeeded")
    predictions: List[ReceiverProbability] = Field(
        ...,
        description="Predictions for all players, sorted by probability"
    )
    top_receiver: int = Field(..., description="Index of most likely receiver")
    top_probability: float = Field(..., description="Probability of top receiver")


class PositionChange(BaseModel):
    """Position change for a single player."""
    player_index: int = Field(..., description="Player index")
    original: Tuple[float, float] = Field(..., description="Original position (x, y)")
    optimized: Tuple[float, float] = Field(..., description="Optimized position (x, y)")
    movement_distance: float = Field(..., description="Distance moved in meters")


class OptimizationResponse(BaseModel):
    """Response for position optimization."""
    success: bool = Field(..., description="Whether optimization succeeded")
    target_receiver: int = Field(..., description="Target receiver index")
    original_probability: float = Field(..., description="Original receiver probability")
    optimized_probability: float = Field(..., description="Optimized receiver probability")
    improvement_percentage: float = Field(..., description="Relative improvement percentage")
    position_changes: List[PositionChange] = Field(
        ...,
        description="List of position changes"
    )
    num_iterations: int = Field(..., description="Number of iterations run")
    converged: bool = Field(..., description="Whether optimization converged")
    visualization_base64: Optional[str] = Field(
        None,
        description="Base64-encoded PNG visualization"
    )


class CornerSummary(BaseModel):
    """Summary info for a corner kick."""
    corner_id: int = Field(..., description="Corner index in dataset")
    match_id: int = Field(..., description="Match ID")
    num_players: int = Field(..., description="Total number of players")
    num_attackers: int = Field(..., description="Number of attacking players")
    num_defenders: int = Field(..., description="Number of defending players")
    has_label: bool = Field(..., description="Whether receiver label is available")


class CornerDetail(BaseModel):
    """Detailed corner kick information."""
    corner_id: int
    match_id: int
    num_players: int
    num_attackers: int
    num_defenders: int
    has_label: bool
    label: Optional[int] = Field(None, description="Receiver index if labeled")
    graph: Optional[Dict[str, Any]] = Field(None, description="Full graph data if requested")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="API status: healthy/degraded/unhealthy")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    dataset_size: int = Field(..., description="Number of corners in dataset")
    device: str = Field(..., description="Compute device: cuda/mps/cpu")
