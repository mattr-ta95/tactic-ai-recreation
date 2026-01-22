#!/usr/bin/env python3
"""
Download StatsBomb data for TacticAI project
Run this first to get the data needed for training
Supports downloading from multiple competitions/seasons
"""

import os
import pandas as pd
from statsbombpy import sb
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Predefined competitions with good 360 coverage
DEFAULT_COMPETITIONS = [
    # Format: (competition_name, competition_id, season_id, description)
    ("Premier League", 2, 27, "2015/2016"),  # 380 matches
    ("World Cup", 43, 3, "2018"),  # 64 matches, full 360
    ("World Cup", 43, 106, "2022"),  # 64 matches, full 360
    ("Euro", 55, 43, "2020"),  # 51 matches
    ("Euro", 55, 282, "2024"),  # 51 matches
]


def download_competition_data(
    competition_id: int,
    season_id: int,
    competition_name: str,
    season_desc: str,
    num_matches: Optional[int] = None,
    output_dir: str = 'data/raw'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download matches and events for a single competition/season
    
    Returns:
        Tuple of (matches_df, events_df)
    """
    print(f"\n{'='*70}")
    print(f"Downloading: {competition_name} {season_desc}")
    print(f"{'='*70}")
    
    try:
        # Get matches
        matches = sb.matches(competition_id=competition_id, season_id=season_id)
        print(f"   Found {len(matches)} matches")
        
        # Limit if specified
        if num_matches is not None and num_matches > 0:
            matches = matches.head(num_matches)
            print(f"   Limiting to first {num_matches} matches")
        
        if len(matches) == 0:
            print(f"   ⚠️  No matches found")
            return pd.DataFrame(), pd.DataFrame()
        
        # Download events
        all_events = []
        successful_matches = 0
        
        for idx, (_, match) in enumerate(matches.iterrows()):
            match_id = match['match_id']
            try:
                events = sb.events(match_id=match_id)
                events['match_id'] = match_id
                events['competition_id'] = competition_id
                events['season_id'] = season_id
                events['competition_name'] = competition_name
                events['season_desc'] = season_desc
                all_events.append(events)
                successful_matches += 1
                
                if (idx + 1) % 10 == 0:
                    print(f"   [{idx+1}/{len(matches)}] Downloaded {successful_matches} matches...")
            except Exception as e:
                print(f"   ⚠️  Failed match {match_id}: {e}")
                continue
        
        # Combine events
        events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
        
        print(f"   ✅ Successfully downloaded {len(events_df)} events from {successful_matches}/{len(matches)} matches")
        
        # Add metadata to matches
        matches['competition_id'] = competition_id
        matches['season_id'] = season_id
        matches['competition_name'] = competition_name
        matches['season_desc'] = season_desc
        
        return matches, events_df
        
    except Exception as e:
        print(f"   ❌ Error downloading {competition_name} {season_desc}: {e}")
        return pd.DataFrame(), pd.DataFrame()


def download_data(
    output_dir: str = 'data/raw',
    competitions: Optional[List[Tuple[str, int, int, str]]] = None,
    num_matches_per_comp: Optional[int] = None,
    max_total_matches: Optional[int] = None
):
    """
    Download StatsBomb open data from multiple competitions
    
    Args:
        output_dir: Directory to save raw data
        competitions: List of (name, comp_id, season_id, desc) tuples. 
                     If None, uses DEFAULT_COMPETITIONS
        num_matches_per_comp: Max matches per competition (None = all)
        max_total_matches: Max total matches across all competitions (None = all)
    """
    print("=" * 70)
    print("TacticAI Multi-League Data Download Script")
    print("=" * 70)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    # Use default competitions if none provided
    if competitions is None:
        competitions = DEFAULT_COMPETITIONS
    
    print(f"\n📋 Downloading from {len(competitions)} competitions:")
    for name, comp_id, season_id, desc in competitions:
        print(f"   - {name} {desc} (comp_id={comp_id}, season_id={season_id})")
    
    # Download from each competition
    all_matches = []
    all_events = []
    total_matches_downloaded = 0
    
    for comp_name, comp_id, season_id, season_desc in competitions:
        # Check if we've hit the total limit
        if max_total_matches is not None and total_matches_downloaded >= max_total_matches:
            print(f"\n⚠️  Reached max_total_matches limit ({max_total_matches})")
            break
        
        # Calculate remaining matches for this competition
        remaining = None
        if max_total_matches is not None:
            remaining = max_total_matches - total_matches_downloaded
        
        matches_per_comp = num_matches_per_comp
        if remaining is not None and (matches_per_comp is None or matches_per_comp > remaining):
            matches_per_comp = remaining
        
        matches_df, events_df = download_competition_data(
            comp_id, season_id, comp_name, season_desc,
            num_matches=matches_per_comp,
            output_dir=output_dir
        )
        
        if not matches_df.empty:
            all_matches.append(matches_df)
            total_matches_downloaded += len(matches_df)
        
        if not events_df.empty:
            all_events.append(events_df)
    
    # Combine all data
    if not all_matches:
        print("\n❌ No matches downloaded")
        return
    
    combined_matches = pd.concat(all_matches, ignore_index=True)
    combined_events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    
    # Save combined data
    matches_file = f'{output_dir}/matches_multi_league.csv'
    events_file = f'{output_dir}/events_multi_league.csv'
    
    combined_matches.to_csv(matches_file, index=False)
    print(f"\n✅ Saved {len(combined_matches)} matches to {matches_file}")
    
    if not combined_events.empty:
        combined_events.to_csv(events_file, index=False)
        print(f"✅ Saved {len(combined_events)} events to {events_file}")
    else:
        print("\n❌ No events downloaded")
        return
    
    # Extract corners and shots
    print("\n" + "=" * 70)
    print("EXTRACTING CORNERS AND SHOTS WITH FREEZE FRAMES")
    print("=" * 70)
    
    # First get all corner kick passes
    corners = combined_events[
        (combined_events['type'] == 'Pass') &
        (combined_events['pass_type'] == 'Corner')
    ].copy()
    print(f"\n   Found {len(corners)} corner kick passes")

    # Get shots with freeze frames
    shots_with_freeze = combined_events[
        (combined_events['type'] == 'Shot') &
        (combined_events['shot_freeze_frame'].notna())
    ].copy()
    print(f"   Found {len(shots_with_freeze)} shots with freeze frames")

    if len(corners) == 0 and len(shots_with_freeze) == 0:
        print("   ⚠️  No corner data or shots with freeze frames found.")
        if len(corners) > 0:
            corners.to_pickle('data/processed/corners.pkl')
            print(f"   Saved {len(corners)} corners (without freeze frames) to data/processed/corners.pkl")
        return

    # Parse freeze frames for shots
    if len(shots_with_freeze) > 0:
        def parse_freeze_frame(ff):
            if ff is None:
                return None
            if not isinstance(ff, (list, dict)):
                try:
                    if pd.isna(ff):
                        return None
                except (TypeError, ValueError):
                    pass
            if isinstance(ff, list):
                return ff if len(ff) > 0 else None
            if isinstance(ff, str):
                try:
                    result = eval(ff)
                    return result if result else None
                except:
                    return None
            return ff if ff else None

        shots_with_freeze['freeze_frame_parsed'] = shots_with_freeze['shot_freeze_frame'].apply(parse_freeze_frame)
        shots_with_freeze = shots_with_freeze[shots_with_freeze['freeze_frame_parsed'].notna()]
    
    # Save corners
    if len(corners) > 0:
        corners.to_pickle('data/processed/corners.pkl')
        print(f"   ✅ Saved {len(corners)} corners to data/processed/corners.pkl")

    # Save shots with freeze frames
    if len(shots_with_freeze) > 0:
        shots_with_freeze.to_pickle('data/processed/shots_freeze.pkl')
        print(f"   ✅ Saved {len(shots_with_freeze)} shots with freeze frames to data/processed/shots_freeze.pkl")

    # Print detailed statistics
    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"Total matches: {len(combined_matches)}")
    print(f"Total events: {len(combined_events)}")
    print(f"Corner kicks: {len(corners)}")
    
    if len(corners) > 0:
        print(f"Unique teams: {corners['team'].nunique()}")
        print("\nCorners by competition:")
        for comp_name in combined_events['competition_name'].unique():
            comp_corners = corners[corners['competition_name'] == comp_name]
            print(f"   {comp_name}: {len(comp_corners)} corners")
    
    if len(shots_with_freeze) > 0:
        print(f"\nShots with freeze frames: {len(shots_with_freeze)}")
        print("\nShots by competition:")
        for comp_name in combined_events['competition_name'].unique():
            comp_shots = shots_with_freeze[shots_with_freeze['competition_name'] == comp_name]
            print(f"   {comp_name}: {len(comp_shots)} shots")
    
    print("\n" + "=" * 70)
    print("✅ Data download complete!")
    print("\nNext steps:")
    print("  1. Run: python scripts/prepare_training_data.py  # Link shots to corners")
    print("  2. Run: python scripts/train_baseline.py          # Train model")
    print("=" * 70)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download StatsBomb data from multiple competitions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all matches from default competitions (Premier League, World Cups, Euros)
  python scripts/download_data.py
  
  # Download up to 50 matches per competition
  python scripts/download_data.py --matches-per-comp 50
  
  # Download up to 200 total matches across all competitions
  python scripts/download_data.py --max-total-matches 200
  
  # Download only Premier League and World Cup 2022
  python scripts/download_data.py --competitions "Premier League:2:27:2015/2016" "World Cup:43:106:2022"
        """
    )
    parser.add_argument("--matches-per-comp", type=int, default=None,
                       help="Max matches per competition (None = all matches)")
    parser.add_argument("--max-total-matches", type=int, default=None,
                       help="Max total matches across all competitions (None = all)")
    parser.add_argument("--output-dir", type=str, default="data/raw",
                       help="Output directory for raw data")
    parser.add_argument("--competitions", nargs="+", default=None,
                       help="Custom competitions as 'name:comp_id:season_id:desc' (uses defaults if not provided)")
    
    args = parser.parse_args()
    
    # Parse custom competitions if provided
    competitions = None
    if args.competitions:
        competitions = []
        for comp_str in args.competitions:
            try:
                parts = comp_str.split(':')
                if len(parts) != 4:
                    raise ValueError(f"Invalid format: {comp_str}")
                name, comp_id, season_id, desc = parts
                competitions.append((name, int(comp_id), int(season_id), desc))
            except Exception as e:
                print(f"⚠️  Error parsing competition '{comp_str}': {e}")
                print("   Format should be: 'name:comp_id:season_id:desc'")
                continue
    
    download_data(
        output_dir=args.output_dir,
        competitions=competitions,
        num_matches_per_comp=args.matches_per_comp,
        max_total_matches=args.max_total_matches
    )
