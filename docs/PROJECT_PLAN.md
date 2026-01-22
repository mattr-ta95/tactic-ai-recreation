# TacticAI Recreation Project Plan & Roadmap

## 🎯 Project Overview

**Goal:** Recreate TacticAI, Google DeepMind's geometric deep learning system that analyzes and optimizes soccer set pieces (specifically corner kicks), which was successfully deployed with Liverpool FC and achieved 90% coach preference over existing tactics.

**Core Value Proposition:**
- Predict corner kick outcomes (receiver, shot probability, goal probability)
- Generate optimized player positioning to improve tactical outcomes
- Provide interpretable, coach-friendly tactical recommendations
- Extend beyond corners to other set pieces and eventually open play

**Key Innovation:** Uses Graph Neural Networks (GNNs) to model player relationships and tactical patterns, moving beyond traditional coordinate-based approaches.

---

## 📊 Project Impact & Motivation

### Why This Matters
- **30% of goals** come from set pieces in professional soccer
- DeepMind proved this approach works: Liverpool FC coaches favored TacticAI suggestions 90% of the time
- Current methods are manual, time-consuming, and limited in scale
- Real-world deployment path exists (clubs actively seeking such solutions)

### Open Problems You Can Tackle
1. **Extend to other set pieces**: Free kicks, throw-ins, penalty box positioning
2. **Open play analysis**: Transition phases, counter-attacks, build-up patterns
3. **Real-time tactical adjustments**: Live game recommendations
4. **Opponent modeling**: Predict and counter opponent set-piece routines

---

## 🏗️ System Architecture

### High-Level Pipeline

```
INPUT: Corner Kick Scenario
    ↓
┌─────────────────────────────────┐
│  1. GRAPH CONSTRUCTION          │
│  - 22 nodes (players)           │
│  - Node features: position,     │
│    velocity, height, role       │
│  - Edges: distances, marking    │
│    relationships                 │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  2. GRAPH NEURAL NETWORK        │
│  - Message passing (3-5 layers) │
│  - Learn tactical patterns      │
│  - Build holistic understanding │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  3. PREDICTION HEADS            │
│  ├─→ Receiver prediction        │
│  ├─→ Shot prediction            │
│  └─→ Goal probability           │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  4. TACTICAL GENERATOR          │
│  - Suggest position adjustments │
│  - Optimize for desired outcome │
│  - Provide explanations         │
└─────────────────────────────────┘
    ↓
OUTPUT: Predictions + Tactical Recommendations
```

### Core Components

#### 1. Graph Representation

**Nodes (Players):**
```python
node_features = {
    'position': (x, y),              # Location on pitch
    'velocity': (vx, vy),            # Movement vector
    'height': float,                 # Physical attribute
    'role': categorical,             # Attacker/Defender
    'distance_to_ball': float,
    'distance_to_goal': float,
    'team': binary                   # 0 or 1
}
```

**Edges (Relationships):**
```python
edge_features = {
    'type': categorical,             # marking, passing, screening
    'distance': float,               # Euclidean distance
    'angle': float,                  # Relative angle
    'relative_velocity': float,      # Speed difference
    'marking_tightness': float       # How close is marking
}
```

**Graph Construction Logic:**
```python
# Create edge if players are within interaction distance
for i, j in player_pairs:
    if euclidean_distance(player_i, player_j) < 5m:
        create_edge(i, j, compute_features(i, j))
    
    # Special edges for tactical relationships
    if is_marking(player_i, player_j):
        create_edge(i, j, type='marking')
    
    if is_potential_pass(ball_carrier, player_j):
        create_edge(ball_carrier, j, type='pass_option')
```

#### 2. Graph Neural Network Architecture

**Message Passing Mechanism:**
```python
class TacticAI_GNN(nn.Module):
    def __init__(self, node_features, edge_features, hidden_dim):
        super().__init__()
        
        # Graph convolution layers
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Prediction heads
        self.receiver_head = nn.Linear(hidden_dim, 22)  # 22 players
        self.shot_head = nn.Linear(hidden_dim, 1)       # Binary
        self.goal_head = nn.Linear(hidden_dim, 1)       # Binary
        
    def forward(self, x, edge_index, edge_attr):
        # Layer 1: Local neighborhood
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        
        # Layer 2: Extended neighborhood
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        
        # Layer 3: Global tactical understanding
        x = self.conv3(x, edge_index, edge_attr)
        
        # Generate predictions
        receiver_probs = F.softmax(self.receiver_head(x), dim=0)
        shot_prob = torch.sigmoid(self.shot_head(x.mean(dim=0)))
        goal_prob = torch.sigmoid(self.goal_head(x.mean(dim=0)))
        
        return receiver_probs, shot_prob, goal_prob
```

**Why Message Passing Works:**

Round 1: Each player considers immediate neighbors
```
Van Dijk: "Dias is marking me tightly (0.7m away)"
```

Round 2: Players consider neighbors-of-neighbors  
```
Van Dijk: "Dias marks me, but if Salah pulls Walker away..."
```

Round 3: Full team tactical understanding
```
Van Dijk: "If Salah runs near post AND Mané delays,
           I'll have 1.5m of space on my late run!"
```

#### 3. Symmetry Handling

**Challenge:** Left-side and right-side corners are tactically identical but appear different in coordinates.

**Solution:** Apply data augmentation with reflections
```python
def apply_symmetries(corner_data):
    """Generate all symmetric versions of corner kick"""
    augmented_data = []
    
    # Original
    augmented_data.append(corner_data)
    
    # Horizontal flip (left ↔ right)
    flipped = corner_data.copy()
    flipped.x = -flipped.x
    augmented_data.append(flipped)
    
    # Vertical flip (top ↔ bottom)  
    flipped_v = corner_data.copy()
    flipped_v.y = -flipped_v.y
    augmented_data.append(flipped_v)
    
    # Both flips
    flipped_both = corner_data.copy()
    flipped_both.x = -flipped_both.x
    flipped_both.y = -flipped_both.y
    augmented_data.append(flipped_both)
    
    return augmented_data
```

**Benefit:** 4x data augmentation + model learns tactical invariance

#### 4. Tactical Generator (Optimization)

**Goal:** Find player position adjustments that improve outcomes

```python
def generate_tactical_improvements(model, initial_graph, target='shot'):
    """
    Optimize player positions to maximize desired outcome
    """
    # Start with current positions
    positions = initial_graph.x[:, :2].clone().requires_grad_(True)
    optimizer = torch.optim.Adam([positions], lr=0.01)
    
    for iteration in range(100):
        # Build graph with new positions
        graph = update_graph_positions(initial_graph, positions)
        
        # Forward pass
        _, shot_prob, _ = model(graph.x, graph.edge_index, graph.edge_attr)
        
        # Maximize shot probability
        loss = -shot_prob
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Constraints: players can't move too far
        with torch.no_grad():
            delta = positions - initial_graph.x[:, :2]
            delta = torch.clamp(delta, -2.0, 2.0)  # Max 2m adjustment
            positions.data = initial_graph.x[:, :2] + delta
    
    return positions, shot_prob

# Usage
optimized_positions, new_shot_prob = generate_tactical_improvements(
    model, corner_graph, target='shot'
)

print(f"Original shot probability: 12%")
print(f"Optimized shot probability: {new_shot_prob*100:.1f}%")
print(f"Suggested adjustments:")
for player_id, (old_pos, new_pos) in enumerate(zip(original, optimized)):
    if distance(old_pos, new_pos) > 0.5:  # Only show significant moves
        print(f"  Player {player_id}: Move from {old_pos} → {new_pos}")
```

---

## 🛠️ Technology Stack

### Core ML Frameworks

```yaml
Graph Neural Networks:
  - PyTorch Geometric (torch_geometric): Primary GNN library
  - DGL (Deep Graph Library): Alternative for experimentation
  - NetworkX: Graph visualization and analysis

Deep Learning:
  - PyTorch: 2.0+ for core neural network operations
  - PyTorch Lightning: Training loop management
  - Weights & Biases: Experiment tracking

Data Processing:
  - Pandas: Event data manipulation
  - NumPy: Numerical operations
  - Scikit-learn: Preprocessing, metrics
```

### Soccer-Specific Tools

```yaml
Data Access:
  - statsbombpy: StatsBomb open data API
  - kloppy: Standardized soccer data processing
  - socceraction: SPADL format and xT calculations

Visualization:
  - mplsoccer: Pitch plotting and player positions
  - matplotlib: Custom visualizations
  - plotly: Interactive dashboards

Data Formats:
  - SPADL: Standardized event representation
  - StatsBomb 360: Freeze-frame player positions
  - Wyscout: Alternative event data format
```

### Development Environment

```yaml
Version Control:
  - Git: Code versioning
  - DVC: Data version control for large datasets

Environment Management:
  - Conda/venv: Python environment isolation
  - Docker: Reproducible deployment (optional)

IDE Recommendations:
  - VS Code: Python + Jupyter integration
  - PyCharm: Advanced debugging
  - Jupyter: Exploratory analysis
```

### Installation Commands

```bash
# Create environment
conda create -n tacticai python=3.10
conda activate tacticai

# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv

# Soccer data tools
pip install statsbombpy kloppy socceraction mplsoccer

# ML utilities
pip install pytorch-lightning wandb scikit-learn pandas numpy matplotlib seaborn

# Development tools
pip install jupyter ipython black flake8 pytest
```

---

## 📈 Phased Development Roadmap

### Phase 0: Setup & Data Exploration (Week 1-2)

**Goals:**
- Set up development environment
- Download and explore data
- Understand data structure and quality
- Create basic visualizations

**Tasks:**
```python
# 1. Install dependencies (see above)

# 2. Download StatsBomb data
from statsbombpy import sb
competitions = sb.competitions()
matches = sb.matches(competition_id=2, season_id=44)  # Premier League
events = sb.events(match_id=matches.iloc[0].match_id)

# 3. Filter for corner kicks
corners = events[events.type == 'Corner']
print(f"Total corners: {len(corners)}")

# 4. Visualize corner kick
from mplsoccer import Pitch
pitch = Pitch()
fig, ax = pitch.draw()
# Plot players, ball, etc.

# 5. Understand StatsBomb 360 freeze frames
freeze_frame = corners.iloc[0]['freeze_frame']
print(f"Players in frame: {len(freeze_frame)}")
```

**Deliverables:**
- ✅ Working development environment
- ✅ Data downloaded and validated
- ✅ Jupyter notebook with data exploration
- ✅ 5-10 corner kick visualizations
- ✅ Understanding of data quality and limitations

**Success Metrics:**
- Can programmatically access all corner kicks from dataset
- Can visualize any corner kick on a pitch
- Understand player tracking data structure

---

### Phase 1: Simple Baseline Model (Week 3-5)

**Goals:**
- Build simplest possible model for receiver prediction
- Establish baseline performance
- Learn PyTorch Geometric basics

**Tasks:**

#### 1.1 Data Preprocessing
```python
def extract_corner_features(corner_event):
    """Extract features from a corner kick event"""
    freeze_frame = corner_event['freeze_frame']
    
    # Node features for each player
    players = []
    for player in freeze_frame:
        features = [
            player['x'],              # Position
            player['y'],
            player['teammate'],       # Boolean
        ]
        players.append(features)
    
    # Target: who received the ball?
    receiver_id = corner_event.get('recipient', None)
    
    return {
        'node_features': torch.tensor(players, dtype=torch.float),
        'receiver': receiver_id
    }

# Process all corners
dataset = [extract_corner_features(c) for c in corners]
```

#### 1.2 Simple Graph Construction
```python
def build_simple_graph(corner_data):
    """Create graph with distance-based edges"""
    x = corner_data['node_features']  # [num_players, num_features]
    
    # Create edges for players within 5m of each other
    edge_index = []
    for i in range(len(x)):
        for j in range(len(x)):
            if i != j:
                dist = torch.norm(x[i, :2] - x[j, :2])
                if dist < 5.0:  # 5 meter threshold
                    edge_index.append([i, j])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    
    return Data(x=x, edge_index=edge_index)
```

#### 1.3 Baseline GNN Model
```python
from torch_geometric.nn import GCNConv, global_mean_pool

class SimpleCornerGNN(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=64, num_players=22):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, num_players)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # GNN layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        # Predict receiver probabilities
        out = self.classifier(x)
        return F.log_softmax(out, dim=1)
```

#### 1.4 Training Loop
```python
def train_model(model, train_loader, val_loader, epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch}: Loss={total_loss:.4f}, Val Acc={val_acc:.3f}")
```

**Deliverables:**
- ✅ Data preprocessing pipeline
- ✅ Simple GNN implementation
- ✅ Training and evaluation scripts
- ✅ Baseline performance metrics

**Success Metrics:**
- **Target:** 50-60% receiver prediction accuracy
- **Baseline comparison:** Random guess = 4.5% (1/22 players)
- Successfully trains without errors
- Loss decreases over epochs

---

### Phase 2: Enhanced Feature Engineering (Week 6-8)

**Goals:**
- Improve model with richer features
- Add edge features
- Implement player attributes (height, role)
- Target 70%+ accuracy

**Tasks:**

#### 2.1 Advanced Node Features
```python
def extract_advanced_features(player, ball_position, goal_position):
    """Extract comprehensive player features"""
    return {
        # Position & Movement
        'x': player['x'],
        'y': player['y'],
        'velocity_x': player.get('velocity_x', 0),
        'velocity_y': player.get('velocity_y', 0),
        'speed': player.get('speed', 0),
        
        # Physical attributes
        'height': player.get('height', 1.80),  # Default average height
        
        # Tactical context
        'is_attacker': player['teammate'],
        'distance_to_ball': np.linalg.norm([player['x'] - ball_position[0],
                                            player['y'] - ball_position[1]]),
        'distance_to_goal': np.linalg.norm([player['x'] - goal_position[0],
                                            player['y'] - goal_position[1]]),
        'angle_to_goal': np.arctan2(goal_position[1] - player['y'],
                                     goal_position[0] - player['x']),
        
        # Zonal information
        'in_box': is_in_penalty_box(player['x'], player['y']),
        'near_post': abs(player['y'] - goal_position[1]) < 3.0,
    }
```

#### 2.2 Edge Feature Engineering
```python
def compute_edge_features(player_i, player_j):
    """Compute features for edges between players"""
    
    # Distance and angle
    dx = player_j['x'] - player_i['x']
    dy = player_j['y'] - player_i['y']
    distance = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    
    # Relative velocity
    dvx = player_j.get('vx', 0) - player_i.get('vx', 0)
    dvy = player_j.get('vy', 0) - player_i.get('vy', 0)
    rel_velocity = np.sqrt(dvx**2 + dvy**2)
    
    # Tactical relationship
    same_team = player_i['teammate'] == player_j['teammate']
    height_diff = player_j.get('height', 1.80) - player_i.get('height', 1.80)
    
    # Marking intensity (heuristic)
    marking_intensity = 1.0 / (1.0 + distance) if not same_team else 0.0
    
    return {
        'distance': distance,
        'angle': angle,
        'rel_velocity': rel_velocity,
        'same_team': float(same_team),
        'height_diff': height_diff,
        'marking_intensity': marking_intensity
    }
```

#### 2.3 Enhanced GNN with Edge Features
```python
from torch_geometric.nn import GATConv, SAGEConv

class EnhancedCornerGNN(nn.Module):
    def __init__(self, node_dim=12, edge_dim=6, hidden_dim=128):
        super().__init__()
        
        # Use GAT for attention mechanism
        self.conv1 = GATConv(node_dim, hidden_dim, heads=4, edge_dim=edge_dim)
        self.conv2 = GATConv(hidden_dim*4, hidden_dim, heads=4, edge_dim=edge_dim)
        self.conv3 = GATConv(hidden_dim*4, hidden_dim, heads=1, edge_dim=edge_dim)
        
        # Multiple prediction heads
        self.receiver_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 22)
        )
        
        self.shot_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x, edge_index, edge_attr):
        # Message passing with attention
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = self.conv3(x, edge_index, edge_attr)
        
        # Predictions
        receiver_logits = self.receiver_head(x)
        shot_prob = torch.sigmoid(self.shot_head(x.mean(dim=0)))
        
        return receiver_logits, shot_prob
```

**Deliverables:**
- ✅ Enhanced feature extraction pipeline
- ✅ Edge feature computation
- ✅ Advanced GNN architecture (GAT/GraphSAGE)
- ✅ Multi-task learning (receiver + shot prediction)

**Success Metrics:**
- **Target:** 70-75% receiver prediction accuracy
- Shot prediction AUC > 0.70
- Model generalizes to held-out matches

---

### Phase 3: Training Infrastructure & Evaluation (Week 9-11)

**Goals:**
- Implement robust training pipeline
- Add proper train/val/test splits
- Implement data augmentation
- Create comprehensive evaluation metrics

**Tasks:**

#### 3.1 Data Splitting Strategy
```python
def split_dataset(corners_df, test_ratio=0.15, val_ratio=0.15):
    """
    Split by MATCHES not by corners (prevent data leakage)
    """
    unique_matches = corners_df['match_id'].unique()
    
    # Shuffle matches
    np.random.shuffle(unique_matches)
    
    n_test = int(len(unique_matches) * test_ratio)
    n_val = int(len(unique_matches) * val_ratio)
    
    test_matches = unique_matches[:n_test]
    val_matches = unique_matches[n_test:n_test+n_val]
    train_matches = unique_matches[n_test+n_val:]
    
    train_corners = corners_df[corners_df['match_id'].isin(train_matches)]
    val_corners = corners_df[corners_df['match_id'].isin(val_matches)]
    test_corners = corners_df[corners_df['match_id'].isin(test_matches)]
    
    return train_corners, val_corners, test_corners
```

#### 3.2 Data Augmentation
```python
def augment_corner(corner_graph):
    """Apply symmetry-based data augmentation"""
    augmented = []
    
    # Original
    augmented.append(corner_graph)
    
    # Horizontal flip
    flipped = corner_graph.clone()
    flipped.x[:, 0] = 120 - flipped.x[:, 0]  # Flip x-coordinate
    augmented.append(flipped)
    
    # Vertical flip
    flipped_v = corner_graph.clone()
    flipped_v.x[:, 1] = 80 - flipped_v.x[:, 1]  # Flip y-coordinate
    augmented.append(flipped_v)
    
    # Both flips
    flipped_both = corner_graph.clone()
    flipped_both.x[:, 0] = 120 - flipped_both.x[:, 0]
    flipped_both.x[:, 1] = 80 - flipped_both.x[:, 1]
    augmented.append(flipped_both)
    
    return augmented
```

#### 3.3 Training with PyTorch Lightning
```python
import pytorch_lightning as pl

class TacticAIModule(pl.LightningModule):
    def __init__(self, model, learning_rate=0.001):
        super().__init__()
        self.model = model
        self.lr = learning_rate
        self.save_hyperparameters()
        
    def forward(self, batch):
        return self.model(batch.x, batch.edge_index, batch.edge_attr)
    
    def training_step(self, batch, batch_idx):
        receiver_logits, shot_prob = self(batch)
        
        # Multi-task loss
        receiver_loss = F.cross_entropy(receiver_logits, batch.receiver_target)
        shot_loss = F.binary_cross_entropy(shot_prob, batch.shot_target)
        
        loss = receiver_loss + 0.5 * shot_loss  # Weighted combination
        
        self.log('train_loss', loss)
        self.log('train_receiver_loss', receiver_loss)
        self.log('train_shot_loss', shot_loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        receiver_logits, shot_prob = self(batch)
        
        # Compute metrics
        receiver_acc = (receiver_logits.argmax(dim=1) == batch.receiver_target).float().mean()
        shot_acc = ((shot_prob > 0.5) == batch.shot_target).float().mean()
        
        self.log('val_receiver_acc', receiver_acc)
        self.log('val_shot_acc', shot_acc)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_receiver_acc'
        }

# Training
trainer = pl.Trainer(
    max_epochs=100,
    accelerator='gpu',
    devices=1,
    callbacks=[
        pl.callbacks.ModelCheckpoint(monitor='val_receiver_acc', mode='max'),
        pl.callbacks.EarlyStopping(monitor='val_receiver_acc', patience=15, mode='max')
    ]
)
trainer.fit(model, train_dataloader, val_dataloader)
```

#### 3.4 Comprehensive Evaluation
```python
def evaluate_model(model, test_loader, save_predictions=True):
    """Comprehensive model evaluation"""
    model.eval()
    
    metrics = {
        'receiver': {'correct': 0, 'total': 0, 'top3': 0, 'top5': 0},
        'shot': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
        'predictions': []
    }
    
    with torch.no_grad():
        for batch in test_loader:
            receiver_logits, shot_prob = model(batch)
            
            # Receiver prediction
            pred_receiver = receiver_logits.argmax(dim=1)
            top3 = receiver_logits.topk(3, dim=1).indices
            top5 = receiver_logits.topk(5, dim=1).indices
            
            metrics['receiver']['correct'] += (pred_receiver == batch.receiver_target).sum()
            metrics['receiver']['top3'] += sum([t in top3[i] for i, t in enumerate(batch.receiver_target)])
            metrics['receiver']['top5'] += sum([t in top5[i] for i, t in enumerate(batch.receiver_target)])
            metrics['receiver']['total'] += len(batch.receiver_target)
            
            # Shot prediction
            pred_shot = (shot_prob > 0.5).int()
            metrics['shot']['tp'] += ((pred_shot == 1) & (batch.shot_target == 1)).sum()
            metrics['shot']['fp'] += ((pred_shot == 1) & (batch.shot_target == 0)).sum()
            metrics['shot']['tn'] += ((pred_shot == 0) & (batch.shot_target == 0)).sum()
            metrics['shot']['fn'] += ((pred_shot == 0) & (batch.shot_target == 1)).sum()
            
            # Save predictions
            if save_predictions:
                for i in range(len(batch)):
                    metrics['predictions'].append({
                        'true_receiver': batch.receiver_target[i].item(),
                        'pred_receiver': pred_receiver[i].item(),
                        'receiver_probs': receiver_logits[i].cpu().numpy(),
                        'true_shot': batch.shot_target[i].item(),
                        'pred_shot': shot_prob[i].item()
                    })
    
    # Calculate final metrics
    results = {
        'receiver_accuracy': metrics['receiver']['correct'] / metrics['receiver']['total'],
        'receiver_top3_accuracy': metrics['receiver']['top3'] / metrics['receiver']['total'],
        'receiver_top5_accuracy': metrics['receiver']['top5'] / metrics['receiver']['total'],
        'shot_accuracy': (metrics['shot']['tp'] + metrics['shot']['tn']) / 
                        (metrics['shot']['tp'] + metrics['shot']['fp'] + 
                         metrics['shot']['tn'] + metrics['shot']['fn']),
        'shot_precision': metrics['shot']['tp'] / (metrics['shot']['tp'] + metrics['shot']['fp']),
        'shot_recall': metrics['shot']['tp'] / (metrics['shot']['tp'] + metrics['shot']['fn']),
    }
    
    return results, metrics['predictions']
```

#### 3.5 Visualization & Analysis
```python
def visualize_corner_prediction(corner_graph, prediction, true_receiver):
    """Visualize a corner kick with predictions"""
    from mplsoccer import Pitch
    
    fig, ax = plt.subplots(figsize=(12, 8))
    pitch = Pitch(pitch_type='statsbomb', pitch_color='grass', line_color='white')
    pitch.draw(ax=ax)
    
    # Plot players
    attackers = corner_graph.x[corner_graph.x[:, -1] == 1]  # Assume last feature is team
    defenders = corner_graph.x[corner_graph.x[:, -1] == 0]
    
    pitch.scatter(attackers[:, 0], attackers[:, 1], ax=ax, c='red', s=200, label='Attackers')
    pitch.scatter(defenders[:, 0], defenders[:, 1], ax=ax, c='blue', s=200, label='Defenders')
    
    # Highlight predicted receiver
    pred_receiver_pos = corner_graph.x[prediction['pred_receiver'], :2]
    pitch.scatter(pred_receiver_pos[0], pred_receiver_pos[1], ax=ax, 
                 c='yellow', s=500, marker='*', label='Predicted Receiver')
    
    # Highlight true receiver
    true_receiver_pos = corner_graph.x[true_receiver, :2]
    pitch.scatter(true_receiver_pos[0], true_receiver_pos[1], ax=ax,
                 c='green', s=500, marker='D', label='True Receiver')
    
    # Add probability text
    ax.text(5, 75, f"Top 3 Predicted:\n" + 
            "\n".join([f"Player {i}: {p:.2%}" for i, p in 
                      enumerate(prediction['receiver_probs'].argsort()[-3:][::-1])]),
            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.legend()
    plt.title(f"Corner Kick Prediction | Shot Prob: {prediction['pred_shot']:.2%}")
    plt.tight_layout()
    return fig
```

**Deliverables:**
- ✅ Robust training pipeline with PyTorch Lightning
- ✅ Proper data splitting and augmentation
- ✅ Comprehensive evaluation metrics
- ✅ Visualization tools for predictions
- ✅ Experiment tracking with Weights & Biases

**Success Metrics:**
- Receiver accuracy: 72-78%
- Top-3 receiver accuracy: 90%+
- Shot prediction AUC: 0.75+
- Training pipeline is reproducible and well-documented

---

### Phase 4: Tactical Generation System (Week 12-15)

**Goals:**
- Implement tactical optimizer
- Generate improved player positions
- Validate improvements empirically
- Create interpretable explanations

**Tasks:**

#### 4.1 Position Optimization
```python
class TacticalOptimizer:
    def __init__(self, model, constraints):
        self.model = model
        self.constraints = constraints
    
    def optimize_positions(self, initial_graph, objective='shot', iterations=200):
        """
        Find optimal player positions to maximize objective
        
        Args:
            initial_graph: Current corner setup
            objective: 'shot', 'receiver_X', or 'goal'
            iterations: Number of optimization steps
        """
        # Extract initial positions
        positions = initial_graph.x[:, :2].clone().requires_grad_(True)
        fixed_features = initial_graph.x[:, 2:].clone()
        
        # Optimizer
        optimizer = torch.optim.Adam([positions], lr=0.01)
        
        history = {'loss': [], 'positions': []}
        
        for i in range(iterations):
            optimizer.zero_grad()
            
            # Rebuild graph with new positions
            x_new = torch.cat([positions, fixed_features], dim=1)
            edge_index, edge_attr = self._recompute_edges(positions)
            
            # Forward pass
            receiver_logits, shot_prob = self.model(x_new, edge_index, edge_attr)
            
            # Compute loss based on objective
            if objective == 'shot':
                loss = -shot_prob  # Maximize shot probability
            elif objective.startswith('receiver_'):
                receiver_id = int(objective.split('_')[1])
                loss = -receiver_logits[receiver_id]  # Maximize specific player
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Apply constraints
            with torch.no_grad():
                positions.data = self._apply_constraints(
                    positions.data, 
                    initial_graph.x[:, :2]
                )
            
            # Track progress
            history['loss'].append(loss.item())
            history['positions'].append(positions.detach().clone())
        
        # Final predictions
        with torch.no_grad():
            x_final = torch.cat([positions, fixed_features], dim=1)
            edge_index, edge_attr = self._recompute_edges(positions)
            final_receiver_logits, final_shot_prob = self.model(
                x_final, edge_index, edge_attr
            )
        
        return {
            'optimized_positions': positions.detach(),
            'history': history,
            'final_predictions': {
                'shot_prob': final_shot_prob.item(),
                'receiver_probs': F.softmax(final_receiver_logits, dim=0).detach()
            }
        }
    
    def _apply_constraints(self, new_positions, original_positions):
        """Apply realistic constraints to position changes"""
        
        # Max movement distance (e.g., 2 meters)
        delta = new_positions - original_positions
        delta_norm = torch.norm(delta, dim=1, keepdim=True)
        max_movement = 2.0
        delta = delta * torch.clamp(max_movement / (delta_norm + 1e-8), max=1.0)
        constrained_positions = original_positions + delta
        
        # Keep players on pitch
        constrained_positions[:, 0] = torch.clamp(constrained_positions[:, 0], 0, 120)
        constrained_positions[:, 1] = torch.clamp(constrained_positions[:, 1], 0, 80)
        
        # Minimum distance between players (avoid collisions)
        min_dist = 0.5  # meters
        for i in range(len(constrained_positions)):
            for j in range(i+1, len(constrained_positions)):
                dist = torch.norm(constrained_positions[i] - constrained_positions[j])
                if dist < min_dist:
                    # Push apart
                    direction = (constrained_positions[j] - constrained_positions[i]) / (dist + 1e-8)
                    constrained_positions[j] += direction * (min_dist - dist) / 2
                    constrained_positions[i] -= direction * (min_dist - dist) / 2
        
        return constrained_positions
    
    def _recompute_edges(self, positions):
        """Recompute edge connections and features based on new positions"""
        edge_index = []
        edge_features = []
        
        for i in range(len(positions)):
            for j in range(len(positions)):
                if i != j:
                    dist = torch.norm(positions[i] - positions[j])
                    if dist < 5.0:  # Connection threshold
                        edge_index.append([i, j])
                        # Compute edge features
                        edge_features.append([
                            dist.item(),
                            (positions[j, 0] - positions[i, 0]).item(),
                            (positions[j, 1] - positions[i, 1]).item()
                        ])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        return edge_index, edge_attr
```

#### 4.2 Generating Tactical Recommendations
```python
def generate_tactical_report(optimizer, corner_graph, model):
    """Generate comprehensive tactical recommendations"""
    
    # Original predictions
    with torch.no_grad():
        orig_receiver_logits, orig_shot_prob = model(
            corner_graph.x, corner_graph.edge_index, corner_graph.edge_attr
        )
        orig_receiver_probs = F.softmax(orig_receiver_logits, dim=0)
    
    # Optimize for shot probability
    result = optimizer.optimize_positions(corner_graph, objective='shot')
    
    # Compare
    improvements = []
    for player_id in range(len(corner_graph.x)):
        orig_pos = corner_graph.x[player_id, :2]
        new_pos = result['optimized_positions'][player_id]
        movement = torch.norm(new_pos - orig_pos).item()
        
        if movement > 0.3:  # Only report significant movements
            improvements.append({
                'player_id': player_id,
                'original_position': orig_pos.tolist(),
                'suggested_position': new_pos.tolist(),
                'movement_distance': movement,
                'direction': (new_pos - orig_pos).tolist()
            })
    
    report = {
        'original_setup': {
            'shot_probability': orig_shot_prob.item(),
            'most_likely_receiver': orig_receiver_probs.argmax().item(),
            'receiver_confidence': orig_receiver_probs.max().item()
        },
        'optimized_setup': {
            'shot_probability': result['final_predictions']['shot_prob'],
            'most_likely_receiver': result['final_predictions']['receiver_probs'].argmax().item(),
            'receiver_confidence': result['final_predictions']['receiver_probs'].max().item()
        },
        'improvements': {
            'shot_probability_increase': (result['final_predictions']['shot_prob'] - 
                                         orig_shot_prob.item()),
            'relative_improvement': ((result['final_predictions']['shot_prob'] - 
                                     orig_shot_prob.item()) / orig_shot_prob.item() * 100)
        },
        'suggested_adjustments': improvements
    }
    
    return report

# Usage
optimizer = TacticalOptimizer(model, constraints={'max_movement': 2.0})
report = generate_tactical_report(optimizer, test_corner, model)

print(f"Original shot probability: {report['original_setup']['shot_probability']:.2%}")
print(f"Optimized shot probability: {report['optimized_setup']['shot_probability']:.2%}")
print(f"Improvement: +{report['improvements']['relative_improvement']:.1f}%")
print(f"\nSuggested adjustments:")
for adj in report['suggested_adjustments']:
    print(f"  Player {adj['player_id']}: Move {adj['movement_distance']:.2f}m")
```

#### 4.3 Explanation Generation
```python
class TacticalExplainer:
    """Generate human-readable explanations for tactical suggestions"""
    
    def __init__(self, model):
        self.model = model
    
    def explain_suggestion(self, original_graph, optimized_positions):
        """
        Explain WHY the suggested changes improve the corner
        """
        explanations = []
        
        # Analyze each significant position change
        for player_id in range(len(original_graph.x)):
            orig_pos = original_graph.x[player_id, :2]
            new_pos = optimized_positions[player_id]
            
            if torch.norm(new_pos - orig_pos) > 0.3:
                # Analyze the impact of this move
                explanation = self._analyze_player_move(
                    player_id, orig_pos, new_pos, original_graph
                )
                explanations.append(explanation)
        
        return explanations
    
    def _analyze_player_move(self, player_id, orig_pos, new_pos, graph):
        """Analyze why moving a player improves the setup"""
        
        # Get player features
        player_features = graph.x[player_id]
        is_attacker = player_features[-1] == 1
        
        # Analyze nearby players
        nearby_orig = self._get_nearby_players(orig_pos, graph)
        nearby_new = self._get_nearby_players(new_pos, graph)
        
        # Generate explanation based on tactical context
        if is_attacker:
            if new_pos[0] > orig_pos[0]:  # Moved forward
                return {
                    'player_id': player_id,
                    'type': 'ATTACKING',
                    'action': 'Advanced forward',
                    'reason': f"Creates {torch.norm(new_pos - orig_pos).item():.1f}m separation from nearest defender",
                    'impact': 'Increases likelihood of winning header'
                }
            elif len(nearby_new) < len(nearby_orig):
                return {
                    'player_id': player_id,
                    'type': 'ATTACKING',
                    'action': 'Moved to open space',
                    'reason': f"Reduced defensive coverage (from {len(nearby_orig)} to {len(nearby_new)} nearby defenders)",
                    'impact': 'Better receiving position'
                }
        else:  # Defender
            if len(nearby_new) > len(nearby_orig):
                return {
                    'player_id': player_id,
                    'type': 'DEFENSIVE',
                    'action': 'Improved zonal coverage',
                    'reason': f"Now covering {len(nearby_new)} attackers vs {len(nearby_orig)}",
                    'impact': 'Reduces opponent shot probability'
                }
        
        return {
            'player_id': player_id,
            'type': 'ADJUSTMENT',
            'action': 'Tactical repositioning',
            'reason': 'Improves team shape',
            'impact': 'Contributes to overall optimization'
        }
    
    def _get_nearby_players(self, position, graph, radius=3.0):
        """Find players within radius of position"""
        distances = torch.norm(graph.x[:, :2] - position, dim=1)
        return torch.where(distances < radius)[0]

# Usage
explainer = TacticalExplainer(model)
explanations = explainer.explain_suggestion(corner_graph, optimized_positions)

for exp in explanations:
    print(f"\nPlayer {exp['player_id']} ({exp['type']}):")
    print(f"  Action: {exp['action']}")
    print(f"  Reason: {exp['reason']}")
    print(f"  Impact: {exp['impact']}")
```

**Deliverables:**
- ✅ Position optimization algorithm
- ✅ Constraint system for realistic suggestions
- ✅ Tactical report generation
- ✅ Explanation system for recommendations
- ✅ Validation framework

**Success Metrics:**
- Generate 10-15% average improvement in shot probability
- Suggestions respect physical constraints
- 80%+ of optimizations produce interpretable explanations
- Visual comparison of before/after setups

---

### Phase 5: Production System & Extensions (Week 16-20)

**Goals:**
- Build production-ready inference system
- Create user-friendly interface
- Extend to other set pieces
- Prepare for deployment

**Tasks:**

#### 5.1 Inference API
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="TacticAI API")

class CornerKickInput(BaseModel):
    """Input schema for corner kick analysis"""
    players: List[Dict[str, float]]  # List of player positions and features
    ball_position: Tuple[float, float]
    attacking_team: str
    
class TacticalRecommendation(BaseModel):
    """Output schema for tactical recommendations"""
    predictions: Dict[str, float]
    suggested_adjustments: List[Dict]
    explanations: List[str]
    confidence: float

@app.post("/analyze-corner", response_model=TacticalRecommendation)
async def analyze_corner(corner: CornerKickInput):
    """Analyze corner kick and provide tactical recommendations"""
    try:
        # Convert input to graph
        graph = convert_input_to_graph(corner)
        
        # Run model
        predictions = model.predict(graph)
        
        # Optimize positions
        optimizer = TacticalOptimizer(model)
        optimization_result = optimizer.optimize_positions(graph)
        
        # Generate explanations
        explainer = TacticalExplainer(model)
        explanations = explainer.explain_suggestion(graph, optimization_result)
        
        return TacticalRecommendation(
            predictions={
                'shot_probability': predictions['shot_prob'],
                'receiver_probabilities': predictions['receiver_probs']
            },
            suggested_adjustments=optimization_result['suggested_adjustments'],
            explanations=explanations,
            confidence=calculate_confidence(predictions)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}
```

#### 5.2 Dashboard & Visualization
```python
import streamlit as st
import plotly.graph_objects as go

def create_dashboard():
    st.title("TacticAI - Corner Kick Optimizer")
    
    # Upload corner kick data
    uploaded_file = st.file_uploader("Upload corner kick data (JSON/CSV)")
    
    if uploaded_file:
        corner_data = load_corner_data(uploaded_file)
        
        # Display current setup
        st.header("Current Setup")
        fig = plot_corner_kick(corner_data)
        st.plotly_chart(fig)
        
        # Run analysis
        if st.button("Analyze & Optimize"):
            with st.spinner("Analyzing..."):
                # Model inference
                predictions = model.predict(corner_data)
                optimizations = optimizer.optimize(corner_data)
                
                # Display predictions
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Shot Probability", 
                             f"{predictions['shot_prob']:.1%}")
                with col2:
                    st.metric("Optimized", 
                             f"{optimizations['shot_prob']:.1%}",
                             delta=f"+{optimizations['improvement']:.1%}")
                with col3:
                    st.metric("Confidence", 
                             f"{predictions['confidence']:.1%}")
                
                # Display optimized setup
                st.header("Optimized Setup")
                fig_opt = plot_corner_kick(optimizations['new_positions'])
                st.plotly_chart(fig_opt)
                
                # Display recommendations
                st.header("Tactical Recommendations")
                for i, adj in enumerate(optimizations['adjustments']):
                    with st.expander(f"Player {adj['player_id']} Adjustment"):
                        st.write(f"**Movement:** {adj['movement']:.2f}m")
                        st.write(f"**Direction:** {adj['direction']}")
                        st.write(f"**Rationale:** {adj['explanation']}")

def plot_corner_kick(corner_data):
    """Create interactive plotly visualization"""
    fig = go.Figure()
    
    # Draw pitch
    fig.add_shape(type="rect", x0=0, y0=0, x1=120, y1=80,
                 line=dict(color="white", width=2))
    
    # Plot players
    attackers = corner_data[corner_data['team'] == 'attacking']
    defenders = corner_data[corner_data['team'] == 'defending']
    
    fig.add_trace(go.Scatter(
        x=attackers['x'], y=attackers['y'],
        mode='markers', name='Attackers',
        marker=dict(size=15, color='red')
    ))
    
    fig.add_trace(go.Scatter(
        x=defenders['x'], y=defenders['y'],
        mode='markers', name='Defenders',
        marker=dict(size=15, color='blue')
    ))
    
    return fig

if __name__ == "__main__":
    create_dashboard()
```

#### 5.3 Extension to Free Kicks
```python
class FreeKickAnalyzer(TacticAIModule):
    """Extend TacticAI to free kicks"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Additional components for free kicks
        self.wall_analyzer = WallAnalyzer()
        self.shot_path_predictor = ShotPathPredictor()
    
    def analyze_free_kick(self, free_kick_data):
        """Analyze free kick setup including wall"""
        
        # Build graph including wall
        graph = self.build_free_kick_graph(free_kick_data)
        
        # Predict outcomes
        predictions = self.model(graph)
        
        # Analyze wall effectiveness
        wall_analysis = self.wall_analyzer.analyze(
            wall_positions=free_kick_data['wall'],
            ball_position=free_kick_data['ball'],
            goal_position=free_kick_data['goal']
        )
        
        # Predict shot paths
        shot_paths = self.shot_path_predictor.predict_paths(
            free_kick_data, wall_analysis
        )
        
        return {
            'predictions': predictions,
            'wall_analysis': wall_analysis,
            'shot_paths': shot_paths,
            'recommendations': self.generate_free_kick_recommendations(
                free_kick_data, predictions, wall_analysis
            )
        }
    
    def build_free_kick_graph(self, free_kick_data):
        """Build graph for free kick with wall"""
        
        # Standard player nodes
        player_nodes = self.create_player_nodes(free_kick_data['players'])
        
        # Add wall as special node type
        wall_node = self.create_wall_node(free_kick_data['wall'])
        
        # Add shot path nodes (potential trajectories)
        path_nodes = self.create_shot_path_nodes(
            free_kick_data['ball'],
            free_kick_data['goal'],
            free_kick_data['wall']
        )
        
        # Combine
        all_nodes = player_nodes + [wall_node] + path_nodes
        
        # Create edges
        edges = self.create_free_kick_edges(all_nodes)
        
        return Data(x=all_nodes, edge_index=edges)
```

**Deliverables:**
- ✅ Production-ready REST API
- ✅ Interactive web dashboard
- ✅ Extension to free kicks (prototype)
- ✅ Documentation and deployment guide
- ✅ Performance optimization (inference <100ms)

**Success Metrics:**
- API handles 100+ requests/second
- Dashboard is intuitive for non-technical users
- Free kick extension shows promising preliminary results
- System is deployed and accessible

---

## 📚 Data Sources & Requirements

### Primary Dataset: StatsBomb Open Data

**Access:**
```python
from statsbombpy import sb

# Get all available competitions
competitions = sb.competitions()

# Recommended competitions for corner analysis:
# - Premier League 2003-2024
# - UEFA Champions League
# - FIFA World Cup 2018, 2022
# - UEFA Euro 2020, 2024

# Example: Get Premier League 2022/23
matches = sb.matches(competition_id=2, season_id=44)
events_df = sb.events(match_id=3788741)

# Filter for corners with 360 data
corners = events_df[
    (events_df['type'] == 'Pass') & 
    (events_df['pass_type'] == 'Corner') &
    (events_df['freeze_frame'].notna())
]
```

**StatsBomb 360 Coverage:**
- **Player positions:** Freeze-frame snapshots at moment of corner
- **Attributes:** Position (x, y), teammate flag
- **Limited:** No velocity, height in open data
- **Matches:** Varies by competition (full coverage for World Cups, partial for leagues)

### Supplementary Data Sources

#### Wyscout Open Dataset
```python
# Download from: https://figshare.com/collections/Soccer_match_event_dataset/4415000

# Includes:
# - 2017/18 season: Premier League, La Liga, Serie A, Bundesliga, Ligue 1
# - World Cup 2018
# - Events with (x, y) coordinates
# - Player IDs for tracking across matches
```

#### Metrica Sports Tracking Data
```python
# Download from: https://github.com/metrica-sports/sample-data

# Limited to 3 matches but includes:
# - Full tracking data (25 Hz)
# - Synchronized with events
# - Velocity calculations possible
# - Great for validation and testing
```

#### SkillCorner
```python
# Broadcast tracking: 9 Champions League matches
# Access: https://github.com/SkillCorner/opendata

# Features:
# - Tracking from broadcast video
# - Includes velocity estimates
# - Good supplement for validation
```

### Data Requirements by Phase

| Phase | Minimum Data | Recommended Data | Notes |
|-------|--------------|------------------|-------|
| **Phase 1** | 200 corners | 500+ corners | Can use single competition |
| **Phase 2** | 500 corners | 1,000+ corners | Multiple competitions better |
| **Phase 3** | 1,000 corners | 2,000+ corners | Need variety for validation |
| **Phase 4** | Same as Phase 3 | 3,000+ corners | More data = better optimization |
| **Phase 5** | Production ready | 5,000+ corners | Continuous data pipeline |

### Handling Missing Data

```python
def handle_missing_features(player_data):
    """Impute or estimate missing features"""
    
    # Height: Use position-based averages
    if 'height' not in player_data:
        position = player_data.get('position', 'unknown')
        player_data['height'] = DEFAULT_HEIGHTS.get(position, 1.80)
    
    # Velocity: Estimate from position history
    if 'velocity' not in player_data:
        if 'position_history' in player_data:
            player_data['velocity'] = estimate_velocity(
                player_data['position_history']
            )
        else:
            player_data['velocity'] = [0.0, 0.0]
    
    # Role: Infer from position
    if 'role' not in player_data:
        player_data['role'] = infer_role(
            player_data['position'], 
            player_data['team']
        )
    
    return player_data
```

---

## 📖 Learning Resources

### Essential Papers

#### TacticAI & Graph Neural Networks
1. **TacticAI: AI assistant for football tactics** (2024)
   - Zhe Wang, Petar Veličković, et al.
   - Nature Communications
   - [Paper](https://www.nature.com/articles/s41467-024-45965-x)
   - **Why read:** The original TacticAI paper - your blueprint

2. **Geometric Deep Learning** (2021)
   - Bronstein, Bruna, Cohen, Veličković
   - [arXiv:2104.13478](https://arxiv.org/abs/2104.13478)
   - **Why read:** Foundational concepts for GNNs on irregular domains

3. **Graph Neural Networks: A Review of Methods and Applications** (2020)
   - Zhou et al.
   - [arXiv:1812.08434](https://arxiv.org/abs/1812.08434)
   - **Why read:** Comprehensive GNN overview

#### Soccer Analytics
4. **Actions Speak Louder than Goals: Valuing Player Actions in Soccer** (2019)
   - Tom Decroos, et al.
   - [arXiv:1802.07127](https://arxiv.org/abs/1802.07127)
   - **Why read:** VAEP framework for action valuation

5. **Data-Driven Ghosting using Deep Imitation Learning** (2020)
   - Le et al.
   - MIT SSAC
   - **Why read:** Player movement prediction

### Online Courses

#### Graph Neural Networks
- **Stanford CS224W: Machine Learning with Graphs**
  - [cs224w.stanford.edu](http://web.stanford.edu/class/cs224w/)
  - Free lectures, assignments
  - Covers: GCN, GraphSAGE, GAT, applications

- **Geometric Deep Learning Course**
  - [geometricdeeplearning.com](https://geometricdeeplearning.com/)
  - By the authors of the GDL textbook
  - Interactive notebooks

#### PyTorch Geometric
- **Official PyG Tutorial**
  - [pytorch-geometric.readthedocs.io](https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html)
  - Start here for implementation
  
- **PyG Examples Repository**
  - [github.com/pyg-team/pytorch_geometric/tree/master/examples](https://github.com/pyg-team/pytorch_geometric/tree/master/examples)
  - Real code for various GNN architectures

#### Soccer Analytics
- **Friends of Tracking**
  - [YouTube Channel](https://www.youtube.com/channel/UCUBFJYcag8j2rm_9HkrrA7w)
  - Free courses on tracking data, event data, ML
  - Includes Python tutorials

- **Soccermatics**
  - [soccermatics.readthedocs.io](https://soccermatics.readthedocs.io/)
  - Python library + educational content
  - By David Sumpter

### Books

1. **Soccermatics: Mathematical Adventures in the Beautiful Game**
   - David Sumpter
   - Entry-level, intuition-building

2. **The Expected Goals Philosophy**
   - James Tippett
   - Deep dive into xG and soccer analytics

3. **Graph Representation Learning** (2020)
   - William L. Hamilton
   - [Free online](https://www.cs.mcgill.ca/~wlh/grl_book/)

### Tutorials & Blogs

#### Implementing TacticAI
- **"Building TacticAI: A Step-by-Step Guide"**
  - Recommended: Write this yourself as you go!
  - Document your journey

#### GNN Tutorials
- **"A Gentle Introduction to GNNs"**
  - [distill.pub/2021/gnn-intro](https://distill.pub/2021/gnn-intro/)
  - Interactive, visual explanations

- **"Understanding Convolutions on Graphs"**
  - [tkipf.github.io/graph-convolutional-networks](https://tkipf.github.io/graph-convolutional-networks/)
  - By the GCN author

#### Soccer Analytics Blogs
- **StatsBomb Blog**
  - [statsbomb.com/articles](https://statsbomb.com/articles/)
  - Industry best practices

- **Jan Van Haaren's Blog**
  - [janvanhaaren.be](https://www.janvanhaaren.be/)
  - Research highlights, tutorials

### Communities

- **Reddit: r/SoccerAnalytics**
  - Active community, paper discussions
  
- **Twitter/X: #FootballAnalytics**
  - Follow: @deepfinisher, @janvanhaaren, @StatsBomb

- **Kaggle: Soccer Analytics Competitions**
  - Practice on real problems

### Video Lectures

- **TacticAI Authors' Talks**
  - Search YouTube for "Petar Veličković TacticAI"
  - DeepMind presentations

- **Football Analytics YouTube Channels**
  - Friends of Tracking (mentioned above)
  - McKay Johns
  - Analytics FC

---

## 🎯 Evaluation Metrics

### Prediction Tasks

#### 1. Receiver Prediction
```python
metrics = {
    'accuracy': 'Top-1 correct / total',                    # Target: 70-78%
    'top3_accuracy': 'True receiver in top 3 / total',      # Target: 90%+
    'top5_accuracy': 'True receiver in top 5 / total',      # Target: 95%+
    'per_player_precision': 'TP / (TP + FP) for each',     # Varies by position
    'per_player_recall': 'TP / (TP + FN) for each'         # Varies by position
}
```

#### 2. Shot Prediction
```python
metrics = {
    'accuracy': '(TP + TN) / total',                       # Target: 75%+
    'precision': 'TP / (TP + FP)',                         # Target: 0.65+
    'recall': 'TP / (TP + FN)',                            # Target: 0.70+
    'f1_score': '2 * (prec * rec) / (prec + rec)',        # Target: 0.67+
    'auc_roc': 'Area under ROC curve',                     # Target: 0.75+
    'auc_pr': 'Area under Precision-Recall curve'          # Target: 0.40+
}
```

#### 3. Goal Prediction
```python
metrics = {
    'log_loss': 'Cross-entropy loss',                      # Target: <0.15
    'brier_score': 'Mean squared error of probabilities',  # Target: <0.08
    'calibration': 'Expected vs observed frequencies'       # Visual assessment
}
```

### Tactical Generation

#### 4. Optimization Quality
```python
metrics = {
    'avg_shot_improvement': 'Mean increase in shot prob',           # Target: 10-15%
    'success_rate': '% of corners improved',                        # Target: 80%+
    'realistic_suggestions': '% respecting constraints',            # Target: 98%+
    'diversity': 'Unique tactical suggestions generated'            # Target: High
}
```

#### 5. Interpretability
```python
metrics = {
    'explanation_coverage': '% moves with explanations',            # Target: 90%+
    'coach_agreement': 'Human validation of suggestions',           # Target: 70%+ (aspirational)
    'consistency': 'Similar inputs → similar suggestions'           # Qualitative
}
```

### System Performance

#### 6. Computational Efficiency
```python
metrics = {
    'inference_time': 'Prediction latency',                        # Target: <50ms
    'optimization_time': 'Tactical generation latency',            # Target: <5s
    'memory_usage': 'Peak GPU memory',                            # Target: <4GB
    'throughput': 'Corners processed per second'                   # Target: 20+
}
```

### Comparison Baselines

```python
BASELINES = {
    'random': {
        'receiver_accuracy': 1/22,  # 4.5%
        'shot_accuracy': 0.5
    },
    'position_only_mlp': {
        'receiver_accuracy': 0.45,  # Without graph structure
        'shot_auc': 0.60
    },
    'xg_model': {
        'shot_auc': 0.72  # Traditional xG
    },
    'human_expert': {
        'receiver_accuracy': 0.65,  # Estimated
        'shot_auc': 0.70
    }
}
```

### Validation Strategy

```python
def validate_model(model, test_set, baselines):
    """Comprehensive model validation"""
    
    results = {}
    
    # 1. Quantitative metrics
    results['receiver'] = evaluate_receiver_prediction(model, test_set)
    results['shot'] = evaluate_shot_prediction(model, test_set)
    
    # 2. Statistical significance
    results['vs_baseline'] = {
        baseline_name: statistical_test(
            model_performance=results['receiver']['accuracy'],
            baseline_performance=baseline_perf,
            test_set_size=len(test_set)
        )
        for baseline_name, baseline_perf in baselines.items()
    }
    
    # 3. Generalization tests
    results['cross_competition'] = evaluate_cross_competition(model, test_set)
    results['cross_season'] = evaluate_cross_season(model, test_set)
    
    # 4. Robustness tests
    results['noise_robustness'] = evaluate_with_noise(model, test_set)
    results['missing_data'] = evaluate_with_missing_features(model, test_set)
    
    # 5. Optimization validation
    results['optimization'] = evaluate_tactical_generation(model, test_set)
    
    return results
```

---

## 🔬 Advanced Extensions & Open Problems

### 1. Free Kick Optimization

**Challenges:**
- Defensive wall positioning
- Shot trajectory prediction through/around wall
- Goalkeeper positioning and reaction

**Approach:**
```python
# Additional graph components
- Wall node: Aggregate representation of defensive wall
- Shot path edges: Possible ball trajectories
- Obstruction features: Which paths are blocked

# New prediction head
- Wall effectiveness: How well does wall block shots?
- Optimal wall position: Where should wall be?
- Shot selection: Direct vs. cross?
```

**Data Requirements:**
- Free kick events with player positions
- Wall positions (may need to infer from tracking data)
- Shot outcomes and trajectories

### 2. Open Play Analysis

**Challenges:**
- Continuous movement (vs. static set pieces)
- Much larger action space
- Longer temporal dependencies

**Approach:**
```python
# Temporal Graph Networks
- Snapshot approach: Graph at each second
- Temporal edges: Connect same player across time
- Predict next N seconds of play

# Use case: Counter-attack optimization
- Detect transition moment
- Build graph of current state
- Predict optimal passing sequence
```

**Data Requirements:**
- Full tracking data (25 Hz)
- Event data for supervision
- Large dataset for diverse scenarios

### 3. Real-Time Tactical Suggestions

**Challenges:**
- Low latency requirement (<100ms)
- Handle streaming data
- Detect tactical patterns in real-time

**Approach:**
```python
# System architecture
- Edge device deployment (iPad/laptop at pitch-side)
- Optimized model (quantization, pruning)
- Incremental graph updates (not rebuild each frame)

# Pattern detection
- Sliding window over recent play
- Alert when opponent pattern detected
- Suggest counter-tactics
```

**Implementation:**
```python
class RealtimeTacticalAssistant:
    def __init__(self, model, pattern_detector):
        self.model = model
        self.pattern_detector = pattern_detector
        self.alert_threshold = 0.85
    
    def process_frame(self, tracking_data):
        # Build graph from current frame
        graph = self.build_graph(tracking_data)
        
        # Detect tactical patterns
        patterns = self.pattern_detector.detect(graph)
        
        # Generate alerts
        alerts = []
        for pattern in patterns:
            if pattern.confidence > self.alert_threshold:
                suggestion = self.generate_suggestion(pattern, graph)
                alerts.append(suggestion)
        
        return alerts
```

### 4. Opponent Modeling

**Goal:** Predict and counter opponent set-piece routines

**Approach:**
```python
# Pre-match preparation
1. Scrape opponent's last 10 matches
2. Extract all corner kick routines
3. Cluster into routine "types"
4. Learn probability of each type given context

# In-game prediction
1. Observe current game context
2. Predict likely routine type
3. Generate optimal defensive counter-tactics

# Implementation
class OpponentModeler:
    def learn_routines(self, opponent_corners):
        # Cluster corners into routine types
        features = self.extract_routine_features(opponent_corners)
        clusters = KMeans(n_clusters=5).fit(features)
        
        return clusters
    
    def predict_routine(self, game_context):
        # Given score, time, players on pitch
        # Predict which routine they'll use
        features = self.extract_context_features(game_context)
        routine_probs = self.classifier.predict_proba(features)
        
        return routine_probs
    
    def generate_counter_tactics(self, predicted_routine):
        # Use TacticAI optimizer
        # But optimize DEFENSIVE positioning
        # Against predicted routine
        pass
```

### 5. Multi-Task Learning Enhancements

**Additional Prediction Targets:**
```python
# Extend model to predict:
- Goal probability (not just shot)
- Defensive clearance likelihood
- Foul probability
- Injury risk (player collisions)
- Time to next event

# Multi-task architecture
class MultiTaskTacticAI(nn.Module):
    def __init__(self):
        self.shared_encoder = GNN(...)
        
        # Multiple heads
        self.receiver_head = ...
        self.shot_head = ...
        self.goal_head = ...
        self.clearance_head = ...
        self.foul_head = ...
    
    def forward(self, graph):
        embeddings = self.shared_encoder(graph)
        
        return {
            'receiver': self.receiver_head(embeddings),
            'shot': self.shot_head(embeddings),
            'goal': self.goal_head(embeddings),
            'clearance': self.clearance_head(embeddings),
            'foul': self.foul_head(embeddings)
        }
```

### 6. Causal Analysis

**Question:** Do the suggested tactics actually CAUSE better outcomes?

**Approach:**
```python
# Causal inference framework
1. Find corners where team FOLLOWED similar suggestion
2. Find corners where team DID NOT follow suggestion
3. Compare outcomes (controlling for confounders)
4. Estimate causal effect

# Techniques
- Propensity score matching
- Difference-in-differences
- Regression discontinuity
```

### 7. Player-Specific Modeling

**Personalization:**
```python
# Instead of generic "Player X"
# Model specific players: "Van Dijk", "Salah", etc.

# Approach
- Add player embeddings
- Learn player-specific strengths
- Generate tactics tailored to available players

class PlayerAwareTacticAI:
    def __init__(self):
        self.gnn = GNN(...)
        self.player_embeddings = nn.Embedding(num_players=500, embedding_dim=32)
    
    def forward(self, graph, player_ids):
        # Augment node features with player embeddings
        player_embeds = self.player_embeddings(player_ids)
        x_augmented = torch.cat([graph.x, player_embeds], dim=1)
        
        # Rest of GNN
        ...
```

### 8. Transfer Learning

**Leverage data from multiple domains:**
```python
# Pre-training strategy
1. Pre-train on large event dataset (all matches)
2. Fine-tune on corner kicks specifically
3. Transfer to other set pieces (throw-ins, etc.)

# Benefits
- Learn general soccer patterns
- Reduce data requirements for specialized tasks
- Improve generalization
```

---

## 🚀 Deployment Considerations

### Model Serving

```python
# Docker container
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY models/ models/
COPY src/ src/

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Monitoring & Logging

```python
import logging
from prometheus_client import Counter, Histogram

# Metrics
inference_counter = Counter('tacticai_inferences_total', 'Total inferences')
inference_time = Histogram('tacticai_inference_seconds', 'Inference time')

@inference_time.time()
def predict(corner_data):
    inference_counter.inc()
    
    try:
        result = model.predict(corner_data)
        logging.info(f"Prediction successful: shot_prob={result['shot_prob']}")
        return result
    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        raise
```

### CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.10
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest tests/ --cov=src/ --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
  
  deploy:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: |
          # Deploy to cloud provider
          echo "Deploying..."
```

---

## 📋 Checklist & Timeline Summary

### Week 1-2: Setup ✅
- [ ] Install all dependencies
- [ ] Download StatsBomb data
- [ ] Create data exploration notebook
- [ ] Visualize 10+ corner kicks
- [ ] Understand data structure

### Week 3-5: Baseline Model ✅
- [ ] Implement data preprocessing
- [ ] Build simple GNN
- [ ] Train baseline model
- [ ] Achieve >50% receiver accuracy
- [ ] Document results

### Week 6-8: Enhanced Features ✅
- [ ] Add advanced node features
- [ ] Implement edge features
- [ ] Use GAT/GraphSAGE architecture
- [ ] Multi-task learning (receiver + shot)
- [ ] Achieve 70%+ receiver accuracy

### Week 9-11: Training Infrastructure ✅
- [ ] Proper train/val/test splits
- [ ] Data augmentation
- [ ] PyTorch Lightning training
- [ ] Comprehensive evaluation
- [ ] Weights & Biases tracking

### Week 12-15: Tactical Generation ✅
- [ ] Position optimization algorithm
- [ ] Constraint system
- [ ] Tactical report generation
- [ ] Explanation system
- [ ] Validation framework

### Week 16-20: Production & Extensions ✅
- [ ] REST API (FastAPI)
- [ ] Interactive dashboard (Streamlit)
- [ ] Free kick extension (prototype)
- [ ] Documentation
- [ ] Deployment

---

## 💡 Tips for Success

### Development Best Practices

1. **Start Simple, Iterate**
   - Don't try to implement everything at once
   - Get baseline working first
   - Add complexity gradually

2. **Version Everything**
   ```bash
   git init
   dvc init
   
   # Track data
   dvc add data/corners.csv
   git add data/corners.csv.dvc
   
   # Track models
   dvc add models/tacticai_v1.pth
   ```

3. **Experiment Tracking**
   ```python
   import wandb
   
   wandb.init(project="tacticai", config={
       "model": "GAT",
       "hidden_dim": 128,
       "learning_rate": 0.001
   })
   
   # Log metrics
   wandb.log({"train_loss": loss, "val_acc": acc})
   ```

4. **Write Tests**
   ```python
   # tests/test_graph.py
   def test_graph_construction():
       corner = load_sample_corner()
       graph = build_graph(corner)
       
       assert len(graph.x) == 22  # 22 players
       assert graph.edge_index.shape[0] == 2
       assert not torch.isnan(graph.x).any()
   ```

### Common Pitfalls to Avoid

1. **Data Leakage**
   - ❌ Split by corners → same match in train/test
   - ✅ Split by matches

2. **Overfitting**
   - Use dropout, data augmentation
   - Monitor train vs. val metrics
   - Early stopping

3. **Unrealistic Optimizations**
   - Always enforce constraints
   - Validate with domain experts if possible
   - Check physical plausibility

4. **Ignoring Symmetries**
   - Left/right corners are equivalent
   - Use data augmentation

### When You Get Stuck

1. **Model not training:**
   - Check data normalization
   - Verify loss is computed correctly
   - Try simpler model first
   - Check for NaN/Inf values

2. **Poor accuracy:**
   - Are you using enough data?
   - Is data quality good?
   - Try different architectures
   - Visualize predictions to understand failures

3. **Slow training:**
   - Use smaller batch sizes
   - Optimize data loading
   - Profile code to find bottlenecks
   - Use mixed precision training

4. **Questions:**
   - Check PyTorch Geometric documentation
   - Ask on Reddit r/MachineLearning
   - Review TacticAI paper again
   - Look at similar projects on GitHub

---

## 🎓 Additional Context for Claude Code

### File Structure
```
tacticai-project/
├── data/
│   ├── raw/                    # Downloaded data
│   ├── processed/              # Preprocessed graphs
│   └── splits/                 # Train/val/test splits
├── models/
│   ├── checkpoints/           # Saved model weights
│   └── configs/               # Model configurations
├── src/
│   ├── data/
│   │   ├── loader.py         # Data loading
│   │   ├── processor.py      # Feature extraction
│   │   └── augmentation.py   # Data augmentation
│   ├── models/
│   │   ├── gnn.py           # GNN architectures
│   │   ├── optimizer.py     # Tactical optimizer
│   │   └── explainer.py     # Explanation generation
│   ├── training/
│   │   ├── train.py         # Training loop
│   │   └── evaluate.py      # Evaluation
│   └── utils/
│       ├── visualization.py # Plotting functions
│       └── metrics.py       # Custom metrics
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_model.ipynb
│   └── 03_results_analysis.ipynb
├── api/
│   ├── app.py              # FastAPI application
│   └── schemas.py          # API schemas
├── dashboard/
│   └── streamlit_app.py    # Interactive dashboard
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_optimization.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
├── setup.py
└── README.md
```

### Key Variables & Conventions

```python
# Standard variable names to use consistently
corner_graph: Data         # PyG Data object for a corner kick
x: Tensor                 # Node features [num_players, num_features]
edge_index: Tensor        # Edge connections [2, num_edges]
edge_attr: Tensor         # Edge features [num_edges, num_edge_features]

# Feature dimensions
NODE_FEATURES = 12        # Position, velocity, height, role, etc.
EDGE_FEATURES = 6         # Distance, angle, relative velocity, etc.
HIDDEN_DIM = 128         # GNN hidden dimension
NUM_PLAYERS = 22         # Players on pitch

# Coordinate system (StatsBomb)
X_MIN, X_MAX = 0, 120    # Pitch length (meters)
Y_MIN, Y_MAX = 0, 80     # Pitch width (meters)
```

### Debug Mode

```python
# Add to training scripts for debugging
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

if DEBUG:
    # Reduce dataset size
    train_data = train_data[:100]
    val_data = val_data[:20]
    
    # More frequent logging
    log_every = 1
    
    # Save intermediate outputs
    save_predictions = True
```

### Performance Optimization Tips

```python
# 1. Use DataLoader efficiently
from torch_geometric.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,           # Parallel data loading
    pin_memory=True          # Faster GPU transfer
)

# 2. Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(batch)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# 3. Profile your code
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumtime')
stats.print_stats(20)  # Top 20 time consumers
```

---

## 🎯 Success Criteria

By the end of this project, you should have:

✅ **Working System:**
- GNN model achieving 70%+ receiver prediction accuracy
- Tactical optimizer generating 10-15% average improvements
- Production-ready API and dashboard

✅ **Technical Skills:**
- Deep understanding of Graph Neural Networks
- Proficiency with PyTorch Geometric
- Experience with soccer analytics data

✅ **Deliverables:**
- Clean, documented codebase
- Comprehensive evaluation results
- Visualizations and demonstrations
- Technical documentation

✅ **Extensions (Optional):**
- Free kick analysis prototype
- Real-time inference system
- Opponent modeling capabilities

✅ **Publication-Ready (Aspirational):**
- Novel contributions beyond TacticAI
- Rigorous experimental validation
- Potential for academic paper or blog post

---

## 📞 Support & Resources

### Documentation Links
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io
- StatsBomb: https://statsbombpy.readthedocs.io
- FastAPI: https://fastapi.tiangolo.com
- Streamlit: https://docs.streamlit.io

### Example Repositories
- PyG Examples: https://github.com/pyg-team/pytorch_geometric/tree/master/examples
- Soccer Analytics: https://github.com/Friends-of-Tracking-Data-FoTD
- StatsBomb Tutorials: https://github.com/statsbomb/open-data

### Community
- PyTorch Geometric Discussions: https://github.com/pyg-team/pytorch_geometric/discussions
- Reddit r/SoccerAnalytics
- Twitter #FootballAnalytics

---

**Good luck with your TacticAI recreation! Remember: start simple, iterate quickly, and don't hesitate to ask questions. The community is very supportive of newcomers to soccer analytics and graph ML.**

**Most importantly: Have fun and enjoy the intersection of football and AI! ⚽🤖**
