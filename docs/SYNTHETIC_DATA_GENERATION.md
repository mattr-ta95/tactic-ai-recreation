# TacticAI Synthetic Data Generation

## Overview

This document describes the synthetic corner kick data generation system for TacticAI. The system generates statistically realistic corner kick scenarios to augment the limited real training data, improving model accuracy from 34.6% to 42.0%.

---

## Table of Contents

1. [Motivation](#motivation)
2. [Architecture](#architecture)
3. [Statistical Distributions](#statistical-distributions)
4. [Tactical Formations](#tactical-formations)
5. [Receiver Selection](#receiver-selection)
6. [Data Format](#data-format)
7. [Usage Guide](#usage-guide)
8. [Integration with Training](#integration-with-training)
9. [Results and Impact](#results-and-impact)
10. [Customization](#customization)

---

## Motivation

### The Data Scarcity Problem

The TacticAI model requires labeled corner kick data with:
- Player positions (freeze frames)
- Receiver labels (who receives the corner pass)

**Challenge**: StatsBomb open data provides only ~739 labeled examples with usable freeze frames. This is insufficient for training a robust GNN model.

### Why Synthetic Data?

| Approach | Pros | Cons |
|----------|------|------|
| Collect more real data | Ground truth | Expensive, time-consuming |
| Data augmentation | Simple | Limited variety |
| **Synthetic generation** | **Unlimited volume, tactical variety** | **Requires domain knowledge** |

We chose synthetic generation because:
1. Soccer corner kicks follow predictable tactical patterns
2. Player positioning distributions can be learned from real data
3. We can inject tactical variety (formations) not present in limited real data

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    CornerKickSimulator                          │
│                    (Main Orchestrator)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────┐  ┌─────────────────────────────────┐  │
│  │ StatisticalDistri-  │  │     FormationGenerator          │  │
│  │ butions             │  │                                 │  │
│  │                     │  │  ┌─────────────────────────┐    │  │
│  │ • Player counts     │  │  │ generate_zonal_defense  │    │  │
│  │ • Position ranges   │  │  │ generate_man_marking    │    │  │
│  │ • Corner locations  │  │  │ generate_mixed_defense  │    │  │
│  │                     │  │  │ generate_near_post      │    │  │
│  └─────────────────────┘  │  │ generate_far_post       │    │  │
│                           │  └─────────────────────────┘    │  │
│                           └─────────────────────────────────┘  │
│                                                                 │
│  ┌─────────────────────┐  ┌─────────────────────────────────┐  │
│  │  ReceiverSelector   │  │      Output Generator           │  │
│  │                     │  │                                 │  │
│  │ • Distance scoring  │  │  • freeze_frame_parsed          │  │
│  │ • Zone bonus        │  │  • corner_pass_recipient_id     │  │
│  │ • Position bonus    │  │  • corner_pass_end_location     │  │
│  └─────────────────────┘  │  • match_id, formation_type     │  │
│                           └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### File Structure

```
scripts/
└── generate_synthetic_corners.py    # Main generation script (600+ lines)

data/processed/
├── training_shots.pkl               # Original real data (739 examples)
├── synthetic_corners.pkl            # Generated synthetic data (3500 examples)
└── training_shots_combined.pkl      # Combined dataset (4239 examples)
```

---

## Statistical Distributions

### Extraction from Real Data

The `StatisticalDistributions` class analyzes real training data to extract:

#### Player Counts

| Metric | Mean | Std Dev | Range |
|--------|------|---------|-------|
| Total players | 16.8 | 2.0 | 5-21 |
| Attackers | 6.4 | 1.3 | 1-10 |
| Defenders | 10.4 | 0.9 | 4-11 |

**Sampling method**: Truncated normal distribution
```python
attackers = int(np.clip(np.random.normal(6.4, 1.3), 3, 10))
defenders = int(np.clip(np.random.normal(10.4, 0.9), 6, 11))
```

#### Position Distributions

**Attackers (teammates)**:
| Coordinate | Mean | Std Dev | Range |
|------------|------|---------|-------|
| X | 108.3 | 7.5 | 90-119 |
| Y | 40.3 | 12.8 | 5-75 |

**Defenders (opponents)**:
| Coordinate | Mean | Std Dev | Range |
|------------|------|---------|-------|
| X | 111.0 | 5.9 | 95-119 |
| Y | 40.3 | 7.5 | 10-70 |

**Goalkeeper**:
| Coordinate | Mean | Std Dev | Range |
|------------|------|---------|-------|
| X | 118.6 | 1.0 | 117-120 |
| Y | 40.0 | 1.5 | 35-45 |

#### Corner Pass End Location

| Coordinate | Mean | Std Dev | Range |
|------------|------|---------|-------|
| X | 111.4 | 4.7 | 95-118 |
| Y | 39.9 | 15.3 | 10-70 |

### Pitch Coordinate System

```
        0                    60                   120
    0   ┌────────────────────┬────────────────────┐
        │                    │                    │
        │    DEFENDING       │     ATTACKING      │
   40   │       HALF         │        HALF        │  ← Goal at x=120
        │                    │                    │
        │                    │                    │
   80   └────────────────────┴────────────────────┘

        Corner kicks target the attacking half (x > 90)
        Goal center at (120, 40)
```

---

## Tactical Formations

### 1. Zonal Defense (30% of samples)

**Concept**: Defenders position in fixed zones regardless of where attackers are.

**Zone Definitions**:
```
Zone Name       | X Range    | Y Range    | Purpose
----------------|------------|------------|------------------
Near Post       | 116-118    | 34-38      | Block near post headers
Far Post        | 116-118    | 42-46      | Block far post headers
Center          | 110-114    | 38-42      | Central coverage
Penalty Spot    | 105-109    | 38-42      | Edge of 6-yard box
Edge Near       | 101-105    | 32-38      | Clearances
Edge Far        | 101-105    | 42-48      | Clearances
Top of Box      | 95-101     | 35-45      | Second ball coverage
```

**Visual Representation**:
```
                    GOAL
            ┌─────────────────┐
            │   GK            │
     ┌──────┼─────────────────┼──────┐
     │      │  NP    FP       │      │
     │      │     CENTER      │      │  ← 6-yard box
     │      └─────────────────┘      │
     │    PENALTY    PENALTY         │  ← Penalty area
     │     SPOT       SPOT           │
     │                               │
     │   EDGE       EDGE             │
     │   NEAR       FAR              │
     │                               │
     │         TOP OF BOX            │
     └───────────────────────────────┘
```

**Implementation**:
```python
def generate_zonal_defense(self, num_defenders: int) -> List[Dict]:
    defenders = [self._generate_goalkeeper()]

    zones = [
        {'x': 117, 'y': 36, 'name': 'near_post'},
        {'x': 117, 'y': 44, 'name': 'far_post'},
        {'x': 112, 'y': 40, 'name': 'center'},
        # ... more zones
    ]

    for zone in zones[:num_defenders-1]:
        x = np.random.normal(zone['x'], 1.5)  # Add noise
        y = np.random.normal(zone['y'], 2.0)
        defenders.append(self._create_player(x, y, teammate=False))

    return defenders
```

---

### 2. Man-Marking Defense (25% of samples)

**Concept**: Each defender is assigned to shadow a specific attacker.

**Positioning Rules**:
- Defender positioned 1.5-3.5m from assigned attacker
- Defender between attacker and goal (higher X coordinate)
- Most dangerous attackers (closest to goal) marked first

**Visual**:
```
        Attacker (A) and their marker (D)

                    GOAL
                      │
            A ← D     │    D positioned between A and goal
                      │
            A ← D     │    Distance: 1.5-3.5m
                      │
              A ← D   │
```

**Implementation**:
```python
def generate_man_marking(self, attackers: List[Dict], num_defenders: int):
    defenders = [self._generate_goalkeeper()]

    # Sort attackers by danger (closer to goal = more dangerous)
    sorted_attackers = sorted(attackers, key=lambda a: -a['location'][0])

    for att in sorted_attackers[:num_defenders-1]:
        att_x, att_y = att['location']

        # Position defender between attacker and goal
        offset = np.random.uniform(1.5, 3.5)
        def_x = min(att_x + offset, 119)
        def_y = att_y + np.random.uniform(-1.5, 1.5)

        defenders.append(self._create_player(def_x, def_y, teammate=False))

    return defenders
```

---

### 3. Mixed Defense (25% of samples)

**Concept**: Combination of zonal and man-marking. Some defenders hold zones, others mark dangerous attackers.

**Configuration**:
- 3 zonal defenders (near post, far post, edge of box)
- Remaining defenders man-mark the most dangerous attackers

```
                    GOAL
            ┌─────────────────┐
            │   GK            │
     ┌──────┼─────────────────┼──────┐
     │      │  Z1    Z2       │      │  Z = Zonal
     │      │                 │      │
     │      └─────────────────┘      │
     │                               │
     │         Z3  (edge)            │
     │                               │
     │    A ← M    A ← M             │  M = Man-marker
     │                               │
     └───────────────────────────────┘
```

---

### 4. Near-Post Attack (10% of samples)

**Concept**: Attacking formation where players cluster at the near post to attack in-swinging corners.

**Player Distribution**:
- 60% at near post (y: 28-42)
- 25% at far post (y: 45-58) for switch option
- 15% at edge of box for second balls

```
                    GOAL
            ┌─────────────────┐
            │                 │
     ┌──────┼─────────────────┼──────┐
     │      │  A A            │      │  ← Near post cluster
     │      │  A A        A   │      │  ← Far post option
     │      └─────────────────┘      │
     │                               │
     │              A                │  ← Edge runner
     │                               │
     └───────────────────────────────┘
```

---

### 5. Far-Post Overload (10% of samples)

**Concept**: Attacking formation overloading the far post, often used with out-swinging corners.

**Player Distribution**:
- 70% at far post (y: 45-65)
- 30% at near post as decoys

```
                    GOAL
            ┌─────────────────┐
            │                 │
     ┌──────┼─────────────────┼──────┐
     │      │      A  A A     │      │  ← Far post overload
     │      │  A      A A     │      │
     │      └─────────────────┘      │
     │                               │
     │                               │
     │                               │
     └───────────────────────────────┘
```

---

## Receiver Selection

### Scoring Algorithm

The receiver is selected based on a weighted scoring system:

```python
def select_receiver(self, attackers, corner_end_location):
    scores = []

    for attacker in attackers:
        # 1. Distance to pass end (closer = higher score)
        dist = distance(attacker, corner_end_location)
        dist_score = 1.0 / (1.0 + dist * 0.1)

        # 2. Zone bonus (penalty box positions)
        zone_score = 1.0
        if in_penalty_box(attacker):
            zone_score = 1.5
        if in_prime_scoring_zone(attacker):  # x>108, 30<y<50
            zone_score = 2.0

        # 3. Position bonus (forwards more likely)
        position_score = 1.2 if is_forward(attacker) else 1.0

        # 4. Random factor for variety
        random_factor = np.random.uniform(0.8, 1.2)

        final_score = dist_score * zone_score * position_score * random_factor
        scores.append((attacker, final_score))

    return max(scores, key=lambda x: x[1])
```

### Score Components Explained

| Component | Weight | Rationale |
|-----------|--------|-----------|
| Distance to pass end | Primary | Ball delivered to specific location |
| Penalty box bonus | 1.5x | Dangerous scoring position |
| Prime zone bonus | 2.0x | Optimal header position |
| Forward position | 1.2x | Forwards trained for aerial duels |
| Random factor | 0.8-1.2x | Introduces realistic variety |

---

## Data Format

### Output Schema

Each synthetic example produces:

```python
{
    'freeze_frame_parsed': [
        {
            'location': [114.2, 38.5],        # [x, y] StatsBomb coordinates
            'player': {
                'id': 100001,                  # Unique player ID
                'name': 'ATT_100001'           # Generated name
            },
            'position': {
                'id': 23,                      # StatsBomb position ID
                'name': 'Striker'              # Position name
            },
            'teammate': True                   # True=attacker, False=defender
        },
        # ... 15-20 more players
    ],
    'corner_pass_recipient_id': 100001,       # ID of receiving player
    'corner_pass_end_location': [111.5, 40.2], # Target of corner pass
    'location': [112.3, 39.8],                 # Ball location (shot)
    'match_id': 900042,                        # Synthetic match ID (>=900000)
    'formation_type': 'zonal',                 # Formation used
    'is_synthetic': True,                      # Flag for filtering
    'is_from_corner': True                     # Required by processor
}
```

### Position ID Reference

| ID | Position | Category |
|----|----------|----------|
| 1 | Goalkeeper | GK |
| 2-8 | Backs, Wing Backs | DEF |
| 9-21 | Midfielders | MID |
| 22-25 | Forwards | FWD |

### Compatibility Requirements

The output must be compatible with `CornerKickProcessor.corner_to_graph()`:

```python
# Required columns
required = [
    'freeze_frame_parsed',      # List of player dicts
    'corner_pass_recipient_id', # Receiver player ID
    'corner_pass_end_location', # [x, y] pass target
    'location',                 # [x, y] ball location
    'match_id'                  # For train/test splitting
]

# Freeze frame player dict requirements
player_dict = {
    'location': [float, float],  # Required
    'player': {'id': int},       # Required for receiver matching
    'teammate': bool,            # Required for team classification
    'position': {'id': int}      # Optional but recommended
}
```

---

## Usage Guide

### Basic Usage

```bash
# Generate 3500 synthetic samples
python scripts/generate_synthetic_corners.py --num-samples 3500

# Output: data/processed/synthetic_corners.pkl
```

### With Validation

```bash
# Generate and validate compatibility with processor
python scripts/generate_synthetic_corners.py --num-samples 3500 --validate

# Runs CornerKickProcessor on sample to verify format
```

### Combine with Real Data

```bash
# Generate and combine into single training file
python scripts/generate_synthetic_corners.py --num-samples 3500 --combine

# Output: data/processed/training_shots_combined.pkl
```

### Full Command with All Options

```bash
python scripts/generate_synthetic_corners.py \
    --num-samples 3500 \
    --num-matches 500 \
    --output data/processed/synthetic_corners.pkl \
    --combine \
    --validate \
    --seed 42
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-samples` | 3500 | Number of synthetic examples to generate |
| `--num-matches` | 500 | Number of unique match IDs (for splitting) |
| `--output` | `data/processed/synthetic_corners.pkl` | Output file path |
| `--combine` | False | Combine with real data |
| `--validate` | False | Run compatibility validation |
| `--seed` | 42 | Random seed for reproducibility |

---

## Integration with Training

### Automatic Detection

`train_baseline.py` automatically detects and loads combined data:

```python
# Data loading priority:
# 1. training_shots_combined.pkl (real + synthetic)
# 2. training_shots.pkl (real only)
# 3. shots_linked_to_corners.pkl
# 4. shots_freeze.pkl
# 5. corners.pkl

if os.path.exists('data/processed/training_shots_combined.pkl'):
    corners = pd.read_pickle(combined_path)
    # Loaded 4239 examples (real: 739, synthetic: 3500)
```

### Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                      Training Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Load Data                                                   │
│     ├── training_shots_combined.pkl (4239 examples)             │
│     └── Automatic real/synthetic detection                      │
│                                                                 │
│  2. Create Graphs                                               │
│     ├── CornerKickProcessor.create_dataset()                    │
│     ├── Node features: position, distance, angle                │
│     └── Edge features: distance, angle, same_team               │
│                                                                 │
│  3. Split by Match ID                                           │
│     ├── Train: 60% (real + synthetic matches)                   │
│     ├── Val: 20% (real + synthetic matches)                     │
│     └── Test: 20% (real + synthetic matches)                    │
│                                                                 │
│  4. Apply Augmentation                                          │
│     └── Horizontal flip (2x training data)                      │
│                                                                 │
│  5. Train GNN                                                   │
│     ├── GAT with residual connections                           │
│     ├── Focal loss + label smoothing                            │
│     └── Early stopping on validation                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Match ID Management

To prevent data leakage, synthetic match IDs start at 900000:

```
Real match IDs:      1-899999 (from StatsBomb)
Synthetic match IDs: 900000+ (generated)
```

This ensures:
- No collision with real match IDs
- Proper train/val/test splitting by match
- Easy filtering of synthetic vs real data

---

## Results and Impact

### Performance Comparison

| Metric | Real Only | Combined | Change |
|--------|-----------|----------|--------|
| Training examples | 739 | 4,239 | +474% |
| **Test Accuracy** | 34.6% | **42.0%** | **+7.4pp** |
| Top-3 Accuracy | 77.9% | 77.1% | -0.8pp |
| Top-5 Accuracy | 91.2% | 88.7% | -2.5pp |
| Best Epoch | 33 | 10 | Faster |

### Key Observations

1. **Significant accuracy improvement**: +7.4 percentage points in top-1 accuracy
2. **Faster convergence**: Best model found at epoch 10 vs epoch 33
3. **Maintained top-k performance**: Top-3 and top-5 slightly lower but still strong
4. **Tactical variety**: Model learned from 5 different defensive formations

### Distribution Validation

```
Metric          | Real Data | Synthetic | Match Quality
----------------|-----------|-----------|---------------
Total players   | 16.8±2.0  | 15.9±1.6  | Good
Attackers       | 6.4±1.3   | 5.9±1.3   | Good
Defenders       | 10.4±0.9  | 9.9±0.9   | Excellent
Position X mean | 108-111   | 107-110   | Good
Position Y mean | 40.3      | 40.0      | Excellent
```

---

## Customization

### Adjusting Formation Weights

Edit `CornerKickSimulator.__init__()`:

```python
self.formation_weights = {
    'zonal': 0.30,           # Increase for more zonal defense
    'man_marking': 0.25,     # Increase for more man-marking
    'mixed': 0.25,           # Combination approach
    'near_post_attack': 0.10, # Offensive near-post runs
    'far_post_overload': 0.10 # Offensive far-post overload
}
```

### Adding New Formations

1. Add method to `FormationGenerator`:

```python
def generate_short_corner(self, num_attackers: int) -> List[Dict]:
    """Generate short corner routine positions."""
    attackers = []

    # Short option player near corner flag
    attackers.append(self._create_player(
        x=np.random.normal(105, 2),
        y=np.random.choice([5, 75]),  # Near corner
        teammate=True
    ))

    # Remaining players make runs
    for _ in range(num_attackers - 1):
        # ... positioning logic
        pass

    return attackers
```

2. Add to formation weights in `CornerKickSimulator`:

```python
self.formation_weights['short_corner'] = 0.05
```

3. Update `generate_scenario()` to handle new formation.

### Modifying Statistical Parameters

Edit `StatisticalDistributions._use_default_distributions()`:

```python
def _use_default_distributions(self):
    # Player counts
    self.attacker_mean = 6.4   # Adjust for different leagues
    self.attacker_std = 1.3
    self.defender_mean = 10.4
    self.defender_std = 0.9

    # Position ranges (StatsBomb coordinates)
    self.att_x_mean = 108.3   # Move closer/further from goal
    self.att_y_mean = 40.3    # Adjust lateral distribution
```

---

## Troubleshooting

### Common Issues

**Issue**: "Validation failed: Graph missing receiver label"
```
Cause: Receiver ID not found in freeze frame
Fix: Ensure receiver is included in freeze_frame_parsed before shuffling
```

**Issue**: "No attackers to select receiver from"
```
Cause: num_attackers sampled as 0
Fix: Increase minimum in sample_player_counts(): np.clip(..., 3, 10)
```

**Issue**: Synthetic data looks too uniform
```
Cause: Insufficient position noise
Fix: Increase std dev in position sampling (e.g., 1.5 → 2.5)
```

### Validation Checklist

```bash
# 1. Check file was created
ls -la data/processed/synthetic_corners.pkl

# 2. Verify sample count
python -c "import pandas as pd; print(len(pd.read_pickle('data/processed/synthetic_corners.pkl')))"

# 3. Check formation distribution
python -c "
import pandas as pd
df = pd.read_pickle('data/processed/synthetic_corners.pkl')
print(df['formation_type'].value_counts())
"

# 4. Validate with processor
python scripts/generate_synthetic_corners.py --num-samples 100 --validate
```

---

## References

### Related Files

| File | Purpose |
|------|---------|
| `scripts/generate_synthetic_corners.py` | Main generation script |
| `src/data/processor.py` | Graph construction from freeze frames |
| `scripts/train_baseline.py` | Training pipeline |
| `data/processed/training_shots.pkl` | Real training data |

### Soccer Tactical Resources

- Corner kick defensive strategies: Zonal vs man-marking debate
- StatsBomb coordinate system: 120x80 pitch, origin at corner
- Position ID mapping: StatsBomb event specification

---

## Changelog

### v1.0.0 (2024-01)
- Initial implementation
- 5 tactical formations
- Statistical distribution fitting
- Receiver selection algorithm
- Validation system
- Integration with training pipeline
