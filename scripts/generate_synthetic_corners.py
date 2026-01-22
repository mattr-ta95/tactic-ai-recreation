#!/usr/bin/env python3
"""
Generate synthetic corner kick scenarios for TacticAI training.

Produces statistically realistic corner kick data compatible with CornerKickProcessor.
Includes 5 tactical formations: zonal defense, man-marking, mixed, near-post attack,
and far-post overload.

Usage:
    python scripts/generate_synthetic_corners.py --num-samples 3500
    python scripts/generate_synthetic_corners.py --num-samples 3500 --combine
    python scripts/generate_synthetic_corners.py --num-samples 3500 --combine --validate
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Player position mapping (StatsBomb position IDs)
POSITION_NAMES = {
    1: 'Goalkeeper',
    2: 'Right Back',
    3: 'Right Center Back',
    4: 'Center Back',
    5: 'Left Center Back',
    6: 'Left Back',
    7: 'Right Wing Back',
    8: 'Left Wing Back',
    9: 'Right Defensive Midfield',
    10: 'Center Defensive Midfield',
    11: 'Left Defensive Midfield',
    12: 'Right Midfield',
    13: 'Right Center Midfield',
    14: 'Center Midfield',
    15: 'Left Center Midfield',
    16: 'Left Midfield',
    17: 'Right Wing',
    18: 'Right Attacking Midfield',
    19: 'Center Attacking Midfield',
    20: 'Left Attacking Midfield',
    21: 'Left Wing',
    22: 'Right Center Forward',
    23: 'Striker',
    24: 'Left Center Forward',
    25: 'Secondary Striker',
}

# Position categories
DEFENDER_POSITIONS = [2, 3, 4, 5, 6, 7, 8]
MIDFIELDER_POSITIONS = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
FORWARD_POSITIONS = [22, 23, 24, 25]


class StatisticalDistributions:
    """Extract and sample from real data distributions."""

    def __init__(self, real_data_path: str = 'data/processed/training_shots.pkl'):
        """
        Initialize by fitting distributions from real data.

        Args:
            real_data_path: Path to real training data pickle file
        """
        self.real_data_path = real_data_path
        self._fit_distributions()

    def _fit_distributions(self):
        """Extract statistical distributions from real data."""
        if not os.path.exists(self.real_data_path):
            print(f"Warning: Real data not found at {self.real_data_path}")
            print("Using default distributions")
            self._use_default_distributions()
            return

        df = pd.read_pickle(self.real_data_path)

        # Extract player counts
        self.attacker_counts = []
        self.defender_counts = []
        self.corner_end_xs = []
        self.corner_end_ys = []
        self.attacker_positions = []
        self.defender_positions = []

        for _, row in df.iterrows():
            freeze_frame = row.get('freeze_frame_parsed', [])
            if not isinstance(freeze_frame, list):
                continue

            attackers = [p for p in freeze_frame if p.get('teammate', False)]
            defenders = [p for p in freeze_frame if not p.get('teammate', True)]

            self.attacker_counts.append(len(attackers))
            self.defender_counts.append(len(defenders))

            # Collect positions
            for p in attackers:
                loc = p.get('location', [0, 0])
                self.attacker_positions.append(loc)
                pos_info = p.get('position', {})
                if isinstance(pos_info, dict) and 'id' in pos_info:
                    pass  # Could collect position IDs here

            for p in defenders:
                loc = p.get('location', [0, 0])
                self.defender_positions.append(loc)

            # Corner end location
            corner_end = row.get('corner_pass_end_location')
            if corner_end and isinstance(corner_end, (list, tuple)) and len(corner_end) >= 2:
                self.corner_end_xs.append(corner_end[0])
                self.corner_end_ys.append(corner_end[1])

        # Compute statistics
        self.attacker_mean = np.mean(self.attacker_counts) if self.attacker_counts else 6.4
        self.attacker_std = np.std(self.attacker_counts) if self.attacker_counts else 1.3
        self.defender_mean = np.mean(self.defender_counts) if self.defender_counts else 10.4
        self.defender_std = np.std(self.defender_counts) if self.defender_counts else 0.9

        # Position statistics
        if self.attacker_positions:
            att_pos = np.array(self.attacker_positions)
            self.att_x_mean, self.att_y_mean = att_pos.mean(axis=0)
            self.att_x_std, self.att_y_std = att_pos.std(axis=0)
        else:
            self.att_x_mean, self.att_y_mean = 108.3, 40.3
            self.att_x_std, self.att_y_std = 7.5, 12.8

        if self.defender_positions:
            def_pos = np.array(self.defender_positions)
            self.def_x_mean, self.def_y_mean = def_pos.mean(axis=0)
            self.def_x_std, self.def_y_std = def_pos.std(axis=0)
        else:
            self.def_x_mean, self.def_y_mean = 111.0, 40.3
            self.def_x_std, self.def_y_std = 5.9, 7.5

        # Corner end location statistics
        if self.corner_end_xs:
            self.corner_x_mean = np.mean(self.corner_end_xs)
            self.corner_x_std = np.std(self.corner_end_xs)
            self.corner_y_mean = np.mean(self.corner_end_ys)
            self.corner_y_std = np.std(self.corner_end_ys)
        else:
            self.corner_x_mean, self.corner_x_std = 111.4, 4.7
            self.corner_y_mean, self.corner_y_std = 39.9, 15.3

        print(f"   Fitted distributions from {len(df)} real examples")
        print(f"   Attackers: {self.attacker_mean:.1f} +/- {self.attacker_std:.1f}")
        print(f"   Defenders: {self.defender_mean:.1f} +/- {self.defender_std:.1f}")

    def _use_default_distributions(self):
        """Use default distributions when real data is unavailable."""
        self.attacker_mean, self.attacker_std = 6.4, 1.3
        self.defender_mean, self.defender_std = 10.4, 0.9
        self.att_x_mean, self.att_y_mean = 108.3, 40.3
        self.att_x_std, self.att_y_std = 7.5, 12.8
        self.def_x_mean, self.def_y_mean = 111.0, 40.3
        self.def_x_std, self.def_y_std = 5.9, 7.5
        self.corner_x_mean, self.corner_x_std = 111.4, 4.7
        self.corner_y_mean, self.corner_y_std = 39.9, 15.3

    def sample_player_counts(self) -> Tuple[int, int]:
        """Sample attacker and defender counts from fitted distributions."""
        attackers = int(np.clip(np.random.normal(self.attacker_mean, self.attacker_std), 3, 10))
        defenders = int(np.clip(np.random.normal(self.defender_mean, self.defender_std), 6, 11))
        return attackers, defenders

    def sample_corner_end_location(self) -> List[float]:
        """Sample corner pass end location."""
        x = np.clip(np.random.normal(self.corner_x_mean, self.corner_x_std), 95, 118)
        y = np.clip(np.random.normal(self.corner_y_mean, self.corner_y_std), 10, 70)
        return [float(x), float(y)]

    def sample_attacker_position(self) -> List[float]:
        """Sample an attacker position."""
        x = np.clip(np.random.normal(self.att_x_mean, self.att_x_std), 90, 119)
        y = np.clip(np.random.normal(self.att_y_mean, self.att_y_std), 5, 75)
        return [float(x), float(y)]

    def sample_defender_position(self) -> List[float]:
        """Sample a defender position."""
        x = np.clip(np.random.normal(self.def_x_mean, self.def_x_std), 95, 119)
        y = np.clip(np.random.normal(self.def_y_mean, self.def_y_std), 10, 70)
        return [float(x), float(y)]


class FormationGenerator:
    """Generate tactical formations for corner kicks."""

    def __init__(self, distributions: StatisticalDistributions):
        """
        Initialize formation generator.

        Args:
            distributions: StatisticalDistributions instance for sampling
        """
        self.distributions = distributions
        self._player_id_counter = 100000  # Start IDs for synthetic players

    def _next_player_id(self) -> int:
        """Generate unique player ID."""
        self._player_id_counter += 1
        return self._player_id_counter

    def _create_player(self, x: float, y: float, teammate: bool,
                       position_id: Optional[int] = None,
                       name_prefix: str = "Player") -> Dict:
        """Create a player dictionary."""
        if position_id is None:
            if teammate:
                position_id = np.random.choice(FORWARD_POSITIONS + MIDFIELDER_POSITIONS)
            else:
                position_id = np.random.choice(DEFENDER_POSITIONS + [1])  # Include GK

        player_id = self._next_player_id()
        return {
            'location': [float(x), float(y)],
            'player': {
                'id': player_id,
                'name': f"{name_prefix}_{player_id}"
            },
            'position': {
                'id': position_id,
                'name': POSITION_NAMES.get(position_id, 'Unknown')
            },
            'teammate': teammate
        }

    def _generate_goalkeeper(self) -> Dict:
        """Generate goalkeeper positioned on goal line."""
        x = np.clip(np.random.normal(118.6, 1.0), 117, 120)
        y = np.clip(np.random.normal(40.0, 1.5), 35, 45)
        return self._create_player(x, y, teammate=False, position_id=1, name_prefix="GK")

    def generate_standard_attackers(self, num_attackers: int) -> List[Dict]:
        """Generate attackers with standard positioning."""
        attackers = []
        for i in range(num_attackers):
            pos = self.distributions.sample_attacker_position()
            # Add some noise for variety
            pos[0] += np.random.normal(0, 1.5)
            pos[1] += np.random.normal(0, 2.0)
            pos[0] = np.clip(pos[0], 90, 118)
            pos[1] = np.clip(pos[1], 5, 75)

            position_id = np.random.choice(FORWARD_POSITIONS + MIDFIELDER_POSITIONS)
            attackers.append(self._create_player(
                pos[0], pos[1], teammate=True,
                position_id=position_id, name_prefix="ATT"
            ))
        return attackers

    def generate_near_post_attack(self, num_attackers: int) -> List[Dict]:
        """
        Generate near-post stacking formation.
        Attackers cluster at near post (y: 30-40) with some at far post.
        """
        attackers = []

        # Near post cluster (60% of attackers)
        near_post_count = max(2, int(num_attackers * 0.6))
        for _ in range(near_post_count):
            x = np.clip(np.random.normal(112, 3), 105, 117)
            y = np.clip(np.random.normal(35, 3), 28, 42)
            attackers.append(self._create_player(
                x, y, teammate=True,
                position_id=np.random.choice([22, 23, 24]),
                name_prefix="ATT_NP"
            ))

        # Far post (25% of attackers)
        far_post_count = max(1, int(num_attackers * 0.25))
        for _ in range(far_post_count):
            x = np.clip(np.random.normal(110, 3), 103, 116)
            y = np.clip(np.random.normal(50, 4), 45, 58)
            attackers.append(self._create_player(
                x, y, teammate=True,
                position_id=np.random.choice([22, 24]),
                name_prefix="ATT_FP"
            ))

        # Edge runners (remaining)
        remaining = num_attackers - len(attackers)
        for _ in range(remaining):
            x = np.clip(np.random.normal(102, 4), 95, 108)
            y = np.clip(np.random.normal(40, 10), 20, 60)
            attackers.append(self._create_player(
                x, y, teammate=True,
                position_id=np.random.choice(MIDFIELDER_POSITIONS),
                name_prefix="ATT_EDGE"
            ))

        return attackers

    def generate_far_post_overload(self, num_attackers: int) -> List[Dict]:
        """
        Generate far-post overload formation.
        Attackers overload far post area (y: 45-60) with decoys at near post.
        """
        attackers = []

        # Far post overload (70% of attackers)
        far_count = max(3, int(num_attackers * 0.7))
        for _ in range(far_count):
            x = np.clip(np.random.normal(113, 4), 105, 118)
            y = np.clip(np.random.normal(52, 5), 45, 65)
            attackers.append(self._create_player(
                x, y, teammate=True,
                position_id=np.random.choice([22, 23, 24]),
                name_prefix="ATT_FP"
            ))

        # Near post decoy (remaining)
        near_count = num_attackers - far_count
        for _ in range(near_count):
            x = np.clip(np.random.normal(114, 2), 108, 117)
            y = np.clip(np.random.normal(32, 3), 25, 38)
            attackers.append(self._create_player(
                x, y, teammate=True,
                position_id=np.random.choice(MIDFIELDER_POSITIONS),
                name_prefix="ATT_DECOY"
            ))

        return attackers

    def generate_zonal_defense(self, num_defenders: int) -> List[Dict]:
        """
        Generate zonal defense formation.
        Defenders position in fixed zones regardless of attacker locations.
        """
        defenders = [self._generate_goalkeeper()]

        # Define defensive zones
        zones = [
            {'x': 117, 'y': 36, 'x_std': 1.0, 'y_std': 1.5, 'name': 'near_post'},
            {'x': 117, 'y': 44, 'x_std': 1.0, 'y_std': 1.5, 'name': 'far_post'},
            {'x': 112, 'y': 40, 'x_std': 2.0, 'y_std': 2.0, 'name': 'center'},
            {'x': 107, 'y': 40, 'x_std': 2.0, 'y_std': 3.0, 'name': 'penalty_spot'},
            {'x': 103, 'y': 35, 'x_std': 2.0, 'y_std': 3.0, 'name': 'edge_near'},
            {'x': 103, 'y': 45, 'x_std': 2.0, 'y_std': 3.0, 'name': 'edge_far'},
            {'x': 98, 'y': 40, 'x_std': 3.0, 'y_std': 5.0, 'name': 'top_box'},
        ]

        # Fill zones with defenders
        remaining = num_defenders - 1  # Minus GK
        for i, zone in enumerate(zones[:remaining]):
            x = np.clip(np.random.normal(zone['x'], zone['x_std']), 95, 118)
            y = np.clip(np.random.normal(zone['y'], zone['y_std']), 15, 65)
            defenders.append(self._create_player(
                x, y, teammate=False,
                position_id=np.random.choice(DEFENDER_POSITIONS),
                name_prefix=f"DEF_{zone['name']}"
            ))

        # Add any remaining defenders randomly
        while len(defenders) < num_defenders:
            pos = self.distributions.sample_defender_position()
            defenders.append(self._create_player(
                pos[0], pos[1], teammate=False,
                position_id=np.random.choice(DEFENDER_POSITIONS),
                name_prefix="DEF_EXTRA"
            ))

        return defenders

    def generate_man_marking(self, attackers: List[Dict], num_defenders: int) -> List[Dict]:
        """
        Generate man-marking defense.
        Each defender shadows a specific attacker within 2-4 meters.
        """
        defenders = [self._generate_goalkeeper()]

        # Sort attackers by danger (closer to goal = more dangerous)
        sorted_attackers = sorted(
            attackers,
            key=lambda a: -a['location'][0]  # Higher x = more dangerous
        )

        # Mark most dangerous attackers
        markers_needed = min(len(sorted_attackers), num_defenders - 1)
        for i in range(markers_needed):
            att = sorted_attackers[i]
            att_x, att_y = att['location']

            # Position defender between attacker and goal
            offset = np.random.uniform(1.5, 3.5)
            def_x = min(att_x + offset, 119)
            def_y = att_y + np.random.uniform(-1.5, 1.5)
            def_y = np.clip(def_y, 10, 70)

            defenders.append(self._create_player(
                def_x, def_y, teammate=False,
                position_id=np.random.choice(DEFENDER_POSITIONS),
                name_prefix="DEF_MAN"
            ))

        # Add remaining defenders to cover space
        while len(defenders) < num_defenders:
            pos = self.distributions.sample_defender_position()
            defenders.append(self._create_player(
                pos[0], pos[1], teammate=False,
                position_id=np.random.choice(DEFENDER_POSITIONS),
                name_prefix="DEF_COVER"
            ))

        return defenders

    def generate_mixed_defense(self, attackers: List[Dict], num_defenders: int) -> List[Dict]:
        """
        Generate mixed defense (zonal + man-marking).
        Some defenders in zones, others man-mark dangerous attackers.
        """
        defenders = [self._generate_goalkeeper()]

        # Zonal defenders (3-4 in fixed positions)
        zonal_positions = [
            (117, 38),  # Near post
            (117, 42),  # Far post
            (105, 40),  # Edge of box
        ]

        for x, y in zonal_positions:
            x += np.random.normal(0, 1)
            y += np.random.normal(0, 1.5)
            x = np.clip(x, 100, 118)
            y = np.clip(y, 25, 55)
            defenders.append(self._create_player(
                x, y, teammate=False,
                position_id=np.random.choice(DEFENDER_POSITIONS),
                name_prefix="DEF_ZONAL"
            ))

        # Man-mark remaining attackers
        sorted_attackers = sorted(
            attackers,
            key=lambda a: -a['location'][0]
        )

        man_markers_needed = num_defenders - len(defenders)
        for i in range(min(man_markers_needed, len(sorted_attackers))):
            att = sorted_attackers[i]
            att_x, att_y = att['location']

            offset = np.random.uniform(1.5, 3.0)
            def_x = min(att_x + offset, 119)
            def_y = att_y + np.random.uniform(-1, 1)
            def_y = np.clip(def_y, 10, 70)

            defenders.append(self._create_player(
                def_x, def_y, teammate=False,
                position_id=np.random.choice(DEFENDER_POSITIONS),
                name_prefix="DEF_MAN"
            ))

        # Fill any remaining slots
        while len(defenders) < num_defenders:
            pos = self.distributions.sample_defender_position()
            defenders.append(self._create_player(
                pos[0], pos[1], teammate=False,
                position_id=np.random.choice(DEFENDER_POSITIONS),
                name_prefix="DEF_EXTRA"
            ))

        return defenders


class ReceiverSelector:
    """Select realistic receivers for corner kicks."""

    def __init__(self):
        """Initialize receiver selector."""
        pass

    def select_receiver(self, attackers: List[Dict],
                        corner_end_location: List[float]) -> Tuple[int, Dict]:
        """
        Select receiver based on proximity to pass end and danger zone.

        Args:
            attackers: List of attacker player dicts
            corner_end_location: [x, y] of corner pass target

        Returns:
            Tuple of (receiver index, receiver player dict)
        """
        if not attackers:
            raise ValueError("No attackers to select receiver from")

        end_x, end_y = corner_end_location
        scores = []

        for i, att in enumerate(attackers):
            x, y = att['location']

            # Distance score (closer to pass end = higher)
            dist = np.sqrt((x - end_x) ** 2 + (y - end_y) ** 2)
            dist_score = 1.0 / (1.0 + dist * 0.1)

            # Zone score (penalty box is valuable)
            zone_score = 1.0
            if x > 102 and 18 < y < 62:  # In penalty box
                zone_score = 1.5
            if x > 108 and 30 < y < 50:  # Prime scoring zone
                zone_score = 2.0

            # Height advantage (forwards more likely)
            position_id = att.get('position', {}).get('id', 14)
            position_score = 1.2 if position_id in FORWARD_POSITIONS else 1.0

            # Random factor for variety
            random_factor = np.random.uniform(0.8, 1.2)

            final_score = dist_score * zone_score * position_score * random_factor
            scores.append((i, final_score, att))

        # Select highest scoring attacker
        scores.sort(key=lambda x: -x[1])
        receiver_idx, _, receiver = scores[0]

        return receiver_idx, receiver


class CornerKickSimulator:
    """Main orchestrator for generating synthetic corner kick data."""

    def __init__(self, real_data_path: str = 'data/processed/training_shots.pkl'):
        """
        Initialize simulator.

        Args:
            real_data_path: Path to real training data for distribution fitting
        """
        print("\n1. Fitting statistical distributions from real data...")
        self.distributions = StatisticalDistributions(real_data_path)
        self.formation_gen = FormationGenerator(self.distributions)
        self.receiver_selector = ReceiverSelector()

        # Formation probabilities
        self.formation_weights = {
            'zonal': 0.30,
            'man_marking': 0.25,
            'mixed': 0.25,
            'near_post_attack': 0.10,
            'far_post_overload': 0.10
        }

    def _sample_formation(self) -> str:
        """Sample a formation type based on weights."""
        formations = list(self.formation_weights.keys())
        weights = list(self.formation_weights.values())
        return np.random.choice(formations, p=weights)

    def generate_scenario(self, match_id: int) -> Dict:
        """
        Generate a single corner kick scenario.

        Args:
            match_id: Match identifier

        Returns:
            Dict with freeze_frame_parsed, corner_pass_recipient_id, etc.
        """
        # 1. Sample player counts
        num_attackers, num_defenders = self.distributions.sample_player_counts()

        # 2. Select formation type
        formation = self._sample_formation()

        # 3. Generate attackers based on formation
        if formation == 'near_post_attack':
            attackers = self.formation_gen.generate_near_post_attack(num_attackers)
        elif formation == 'far_post_overload':
            attackers = self.formation_gen.generate_far_post_overload(num_attackers)
        else:
            attackers = self.formation_gen.generate_standard_attackers(num_attackers)

        # 4. Generate defenders based on formation
        if formation == 'zonal':
            defenders = self.formation_gen.generate_zonal_defense(num_defenders)
        elif formation == 'man_marking':
            defenders = self.formation_gen.generate_man_marking(attackers, num_defenders)
        else:  # mixed
            defenders = self.formation_gen.generate_mixed_defense(attackers, num_defenders)

        # 5. Generate corner pass end location
        corner_end = self.distributions.sample_corner_end_location()

        # 6. Select receiver
        receiver_local_idx, receiver = self.receiver_selector.select_receiver(
            attackers, corner_end
        )

        # 7. Build freeze frame (combine and shuffle)
        freeze_frame = attackers + defenders
        np.random.shuffle(freeze_frame)

        # Find receiver index in shuffled frame
        receiver_id = receiver['player']['id']

        # 8. Generate ball/shot location (near corner end with noise)
        shot_x = corner_end[0] + np.random.normal(0, 2)
        shot_y = corner_end[1] + np.random.normal(0, 2)
        shot_location = [
            float(np.clip(shot_x, 90, 120)),
            float(np.clip(shot_y, 0, 80))
        ]

        return {
            'freeze_frame_parsed': freeze_frame,
            'corner_pass_recipient_id': receiver_id,
            'corner_pass_end_location': corner_end,
            'location': shot_location,
            'match_id': match_id,
            'formation_type': formation,
            'is_synthetic': True,
            'is_from_corner': True,  # Required by processor
        }

    def generate_dataset(self, num_samples: int = 3500,
                         num_synthetic_matches: int = 500) -> pd.DataFrame:
        """
        Generate full synthetic dataset.

        Args:
            num_samples: Total number of samples to generate
            num_synthetic_matches: Number of unique match IDs (for splitting)

        Returns:
            DataFrame with synthetic corner kick data
        """
        scenarios = []
        samples_per_match = max(1, num_samples // num_synthetic_matches)

        print(f"\n2. Generating {num_samples} synthetic scenarios...")
        print(f"   Samples per match: ~{samples_per_match}")

        for match_idx in range(num_synthetic_matches):
            # Synthetic match IDs start at 900000 to avoid collision
            match_id = 900000 + match_idx

            # Vary samples per match slightly
            match_samples = samples_per_match + np.random.randint(-2, 3)
            match_samples = max(1, match_samples)

            for _ in range(match_samples):
                if len(scenarios) >= num_samples:
                    break
                scenario = self.generate_scenario(match_id)
                scenarios.append(scenario)

            if len(scenarios) >= num_samples:
                break

        # Ensure we hit exactly num_samples
        while len(scenarios) < num_samples:
            match_id = 900000 + np.random.randint(0, num_synthetic_matches)
            scenarios.append(self.generate_scenario(match_id))

        df = pd.DataFrame(scenarios[:num_samples])
        return df


def validate_compatibility(synthetic_df: pd.DataFrame) -> bool:
    """
    Verify synthetic data works with existing CornerKickProcessor.

    Args:
        synthetic_df: DataFrame with synthetic corner kick data

    Returns:
        True if validation passes, False otherwise
    """
    try:
        from data.processor import CornerKickProcessor

        processor = CornerKickProcessor(
            distance_threshold=5.0,
            normalize_positions=True,
            use_enhanced_features=True
        )

        # Test with small sample
        sample = synthetic_df.head(20)
        graphs = processor.create_dataset(sample)

        # Verify graphs have expected attributes
        valid_count = 0
        for g in graphs:
            if not hasattr(g, 'x'):
                print(f"   Warning: Graph missing node features")
                continue
            if not hasattr(g, 'edge_index'):
                print(f"   Warning: Graph missing edges")
                continue
            if not hasattr(g, 'y'):
                print(f"   Warning: Graph missing receiver label")
                continue
            valid_count += 1

        print(f"   Validation: {valid_count}/{len(graphs)} graphs valid")
        return valid_count == len(graphs)

    except Exception as e:
        print(f"   Validation failed: {e}")
        return False


def compare_distributions(real_df: pd.DataFrame, synthetic_df: pd.DataFrame):
    """Compare key statistics between real and synthetic data."""
    print("\n   Distribution comparison:")

    # Player counts
    for df, name in [(real_df, 'Real'), (synthetic_df, 'Synthetic')]:
        counts = df['freeze_frame_parsed'].apply(len)
        att_counts = df['freeze_frame_parsed'].apply(
            lambda x: sum(1 for p in x if p.get('teammate', False))
        )
        def_counts = df['freeze_frame_parsed'].apply(
            lambda x: sum(1 for p in x if not p.get('teammate', True))
        )

        print(f"   {name}:")
        print(f"     Total players: {counts.mean():.1f} +/- {counts.std():.1f}")
        print(f"     Attackers: {att_counts.mean():.1f} +/- {att_counts.std():.1f}")
        print(f"     Defenders: {def_counts.mean():.1f} +/- {def_counts.std():.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic corner kick data for TacticAI"
    )
    parser.add_argument(
        '--num-samples', type=int, default=3500,
        help='Number of synthetic samples to generate (default: 3500)'
    )
    parser.add_argument(
        '--num-matches', type=int, default=500,
        help='Number of synthetic matches for train/test splitting (default: 500)'
    )
    parser.add_argument(
        '--output', type=str, default='data/processed/synthetic_corners.pkl',
        help='Output path for synthetic data'
    )
    parser.add_argument(
        '--combine', action='store_true',
        help='Combine with real data into training_shots_combined.pkl'
    )
    parser.add_argument(
        '--validate', action='store_true',
        help='Validate compatibility with CornerKickProcessor'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    np.random.seed(args.seed)

    print("=" * 70)
    print("TacticAI Synthetic Corner Kick Generator")
    print("=" * 70)

    # Check real data exists
    real_data_path = 'data/processed/training_shots.pkl'
    if not os.path.exists(real_data_path):
        print(f"\nWarning: Real data not found at {real_data_path}")
        print("Generating with default distributions.")
        print("For better results, run: python scripts/prepare_training_data.py")

    # Generate synthetic data
    simulator = CornerKickSimulator(real_data_path)
    synthetic_df = simulator.generate_dataset(
        num_samples=args.num_samples,
        num_synthetic_matches=args.num_matches
    )

    print(f"   Generated {len(synthetic_df)} samples")
    print(f"   Unique match IDs: {synthetic_df['match_id'].nunique()}")

    # Print formation distribution
    print("\n3. Formation distribution:")
    formation_counts = synthetic_df['formation_type'].value_counts()
    for formation, count in formation_counts.items():
        pct = count / len(synthetic_df) * 100
        print(f"   {formation}: {count} ({pct:.1f}%)")

    # Validate compatibility
    if args.validate:
        print("\n4. Validating compatibility with CornerKickProcessor...")
        if not validate_compatibility(synthetic_df):
            print("   WARNING: Validation issues detected!")

    # Save synthetic data
    print(f"\n5. Saving synthetic data...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    synthetic_df.to_pickle(args.output)
    print(f"   Saved to {args.output}")

    # Optionally combine with real data
    if args.combine:
        print("\n6. Combining with real data...")
        if os.path.exists(real_data_path):
            real_df = pd.read_pickle(real_data_path)
            real_df['is_synthetic'] = False
            real_df['formation_type'] = 'real'

            combined_df = pd.concat([real_df, synthetic_df], ignore_index=True)
            combined_path = 'data/processed/training_shots_combined.pkl'
            combined_df.to_pickle(combined_path)

            print(f"   Combined dataset: {len(combined_df)} samples")
            print(f"   - Real: {len(real_df)} ({len(real_df)/len(combined_df)*100:.1f}%)")
            print(f"   - Synthetic: {len(synthetic_df)} ({len(synthetic_df)/len(combined_df)*100:.1f}%)")
            print(f"   Saved to {combined_path}")

            # Compare distributions
            compare_distributions(real_df, synthetic_df)
        else:
            print(f"   Skipping combine: Real data not found at {real_data_path}")

    print("\n" + "=" * 70)
    print("Generation complete!")
    print("=" * 70)

    print("\nNext steps:")
    if args.combine:
        print("  1. Train with combined data:")
        print("     python scripts/train_baseline.py")
        print("     (The script will automatically detect training_shots_combined.pkl)")
    else:
        print("  1. Combine with real data:")
        print(f"     python scripts/generate_synthetic_corners.py --num-samples {args.num_samples} --combine")
        print("  2. Then train:")
        print("     python scripts/train_baseline.py")


if __name__ == "__main__":
    main()
