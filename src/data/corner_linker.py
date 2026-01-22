"""
Utilities for linking shots with freeze frames to corner passes
to extract receiver labels for training.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict


def find_preceding_corner(shot_row: pd.Series, events_df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Find the corner pass that preceded a shot, if any.

    Args:
        shot_row: Row from shots dataframe
        events_df: Full events dataframe for the match

    Returns:
        Corner pass row if found, None otherwise
    """
    match_id = shot_row.get('match_id')
    shot_index = shot_row.get('index')
    shot_team = shot_row.get('team')
    shot_minute = shot_row.get('minute', 0)
    shot_second = shot_row.get('second', 0)

    if pd.isna(match_id) or pd.isna(shot_index):
        return None

    # Calculate shot timestamp in seconds
    shot_timestamp = shot_minute * 60 + shot_second

    # Find corner passes in same match, SAME TEAM, before this shot
    match_corners = events_df[
        (events_df['match_id'] == match_id) &
        (events_df['team'] == shot_team) &  # FIXED: Ensure same team
        (events_df['type'] == 'Pass') &
        (events_df['pass_type'] == 'Corner') &
        (events_df['index'] < shot_index)
    ].copy()

    if len(match_corners) == 0:
        return None

    # Calculate timestamp for each corner using minute + second
    match_corners['timestamp_sec'] = (
        match_corners['minute'] * 60 +
        match_corners['second'].fillna(0)
    )

    # Calculate time difference in seconds
    match_corners['time_diff'] = shot_timestamp - match_corners['timestamp_sec']

    # Filter to corners within 30 seconds before the shot
    # (Corner kicks typically result in shots within 5-30 seconds)
    recent_corners = match_corners[
        (match_corners['time_diff'] >= 0) &
        (match_corners['time_diff'] <= 30)
    ]

    if len(recent_corners) == 0:
        return None

    # Get most recent corner before shot (closest in time)
    corner = recent_corners.sort_values('timestamp_sec', ascending=False).iloc[0]

    return corner


def link_shots_to_corners(shots_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Link shots with freeze frames to preceding corner passes.
    Adds corner metadata to shot rows.
    
    Args:
        shots_df: Dataframe of shots with freeze frames
        events_df: Full events dataframe (must include corner passes)
        
    Returns:
        Shots dataframe with added corner linking columns:
        - linked_corner_id: index of linked corner pass
        - corner_pass_recipient_id: recipient ID from corner
        - corner_pass_end_location: end location from corner
        - is_from_corner: boolean indicating successful link
    """
    shots_df = shots_df.copy()
    
    # Initialize new columns
    shots_df['linked_corner_id'] = None
    shots_df['corner_pass_recipient_id'] = None
    shots_df['corner_pass_end_location'] = None
    shots_df['is_from_corner'] = False
    
    linked_count = 0
    
    for idx, shot in shots_df.iterrows():
        corner = find_preceding_corner(shot, events_df)
        
        if corner is not None:
            shots_df.at[idx, 'linked_corner_id'] = corner.name
            shots_df.at[idx, 'corner_pass_recipient_id'] = corner.get('pass_recipient_id')
            shots_df.at[idx, 'corner_pass_end_location'] = corner.get('pass_end_location')
            shots_df.at[idx, 'is_from_corner'] = True
            linked_count += 1
    
    print(f"Linked {linked_count}/{len(shots_df)} shots to corner passes ({linked_count/len(shots_df)*100:.1f}%)")
    
    return shots_df


