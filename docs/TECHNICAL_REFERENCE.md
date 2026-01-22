# TacticAI Technical Reference

## 📚 Essential Papers (Ordered by Priority)

### Must Read (Week 1)

#### 1. TacticAI: AI assistant for football tactics
**Authors:** Zhe Wang, Petar Veličković, Daniel Hennes, et al. (Google DeepMind)  
**Published:** Nature Communications, 2024  
**Link:** https://www.nature.com/articles/s41467-024-45965-x  
**Code:** Not publicly available

**Key Takeaways:**
- Graph representation with 22 player nodes
- Message passing GNN for tactical understanding
- Multi-task learning: receiver, shot, outcome prediction
- Symmetry-aware design (4-fold augmentation)
- Liverpool FC deployment: 90% coach preference
- Dataset: 7,176 Premier League corners

**Architecture Details:**
```
Input Graph:
- Nodes: 22 players
- Node features: position (x,y), velocity, height, role
- Edges: distance-based + tactical relationships
- Edge features: distance, angle, relative velocity

GNN:
- 3-5 message passing layers
- Aggregation: mean/sum
- Activation: ReLU
- Symmetry group: D4 (4-fold rotation/reflection)

Output Heads:
1. Receiver prediction: 22-way classification
2. Shot prediction: binary classification  
3. Position optimization: gradient-based
```

**Citations to Use:**
> "TacticAI demonstrates the potential of geometric deep learning for tactical analysis in football, with suggestions favored by Liverpool FC coaches 90% of the time over existing tactics."

---

### Foundational GNN Papers

#### 2. Geometric Deep Learning
**Authors:** Michael M. Bronstein, Joan Bruna, Taco Cohen, Petar Veličković  
**Published:** arXiv, 2021  
**Link:** https://arxiv.org/abs/2104.13478  
**Website:** https://geometricdeeplearning.com

**Why Read:** 
- Unified framework for understanding GNNs, CNNs, Transformers
- Explains symmetries and equivariance (crucial for TacticAI)
- Mathematical foundations

**Key Concepts:**
- Geometric priors: symmetry, scale separation, locality
- Group theory and equivariance
- Message passing as generalized convolution

---

#### 3. Graph Neural Networks: A Review of Methods and Applications
**Authors:** Jie Zhou, Ganqu Cui, Shengding Hu, et al.  
**Published:** AI Open, 2020  
**Link:** https://arxiv.org/abs/1812.08434

**Why Read:**
- Comprehensive survey of GNN architectures
- Comparison: GCN vs GAT vs GraphSAGE vs others
- Applications across domains

**Architecture Comparison:**
```
GCN (Graph Convolutional Network):
- Simple, fast
- Mean aggregation
- Good baseline

GAT (Graph Attention Network):
- Attention-weighted aggregation
- Learns importance of neighbors
- Better performance, more parameters

GraphSAGE:
- Scalable to large graphs
- Sampling-based aggregation
- Multiple aggregator functions (mean, LSTM, pool)
```

---

#### 4. Semi-Supervised Classification with Graph Convolutional Networks
**Authors:** Thomas N. Kipf, Max Welling  
**Published:** ICLR 2017  
**Link:** https://arxiv.org/abs/1609.02907  
**Code:** https://github.com/tkipf/gcn

**Why Read:**
- Original GCN paper
- Simple yet effective
- Good starting point for implementation

**GCN Layer Math:**
```
H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l))

Where:
- Ã = A + I (adjacency + self-loops)
- D̃ = degree matrix of Ã
- H^(l) = node features at layer l
- W^(l) = learnable weights
- σ = activation function
```

---

### Soccer Analytics Papers

#### 5. Actions Speak Louder than Goals: Valuing Player Actions in Soccer
**Authors:** Tom Decroos, Lotte Bransen, Jan Van Haaren, Jesse Davis  
**Published:** KDD 2019  
**Link:** https://arxiv.org/abs/1802.07127  
**Code:** https://github.com/ML-KULeuven/socceraction

**Why Read:**
- VAEP framework for action valuation
- SPADL data format (standardized)
- Baseline for comparing tactical suggestions

**Key Ideas:**
- Value = P(score) - P(concede) within next 10 actions
- Gradient boosting on action sequences
- Applications: player recruitment, tactical analysis

---

#### 6. Data-Driven Ghosting using Deep Imitation Learning
**Authors:** Hoang M. Le, Peter Carr, Yisong Yue, Patrick Lucey  
**Published:** MIT SSAC 2017  
**Link:** https://www.sloansportsconference.com/research-papers/data-driven-ghosting-using-deep-imitation-learning

**Why Read:**
- Player movement prediction
- Imitation learning framework
- Relevant for open play extension

---

#### 7. A Framework for Tactical Analysis and Player Evaluation in Soccer
**Authors:** Javier Fernández, Luke Bornn  
**Published:** MIT SSAC 2018  
**Link:** https://www.sloansportsconference.com/research-papers/a-framework-for-the-fine-grained-evaluation-of-the-instantaneous-expected-value-of-soccer-possessions

**Why Read:**
- Pitch control model
- Expected possession value
- Spatial analysis techniques

---

### Advanced Topics

#### 8. Graph Neural Networks for Temporal Graphs
**Authors:** Various (survey paper)  
**Published:** 2023  
**Link:** https://arxiv.org/abs/2307.03729

**Why Read:**
- Relevant for open play analysis
- Temporal dependencies in soccer
- Future extension of TacticAI

---

#### 9. Causal Inference in Sports Analytics
**Authors:** Various  
**Published:** Multiple papers  

**Key Papers:**
- "Causal inference in sports analytics: A review" (2021)
- "Estimating causal effects in football" (2020)

**Why Read:**
- Validate tactical suggestions causally
- Not just correlation
- Phase 5+ extension

---

## 🗂️ Datasets

### StatsBomb Open Data ⭐ PRIMARY

**Access:**
```python
from statsbombpy import sb

# Get competitions
competitions = sb.competitions()

# Recommended for TacticAI:
# Premier League: competition_id=2, multiple seasons
# Champions League: competition_id=16
# World Cup 2018: competition_id=43, season_id=3
# World Cup 2022: competition_id=43, season_id=106
# Euro 2020: competition_id=55, season_id=43
```

**Coverage:**
- **Events:** All matches with detailed event data
- **StatsBomb 360:** Selected matches with freeze-frame positions
- **Free:** Open competitions
- **Commercial:** Full coverage requires license

**Data Structure:**
```python
# Event columns
- id, index, period, timestamp
- minute, second
- type (Pass, Shot, etc.)
- team, player
- location [x, y]
- pass_type, shot_type, etc.
- freeze_frame (for 360 data)

# Freeze frame structure
[
  {
    'location': [x, y],
    'teammate': bool,
    'actor': bool,
    'keeper': bool
  },
  ...
]
```

**Pros:**
- High quality, professionally tagged
- 360 data includes player positions
- Well documented
- Free access to major tournaments

**Cons:**
- Limited 360 coverage
- No velocity in open data
- No player heights in freeze frames

---

### Wyscout Open Dataset

**Access:** https://figshare.com/collections/Soccer_match_event_dataset/4415000

**Coverage:**
- 2017/18 season: Premier League, La Liga, Serie A, Bundesliga, Ligue 1
- World Cup 2018
- Euro 2016
- ~1,000,000 events
- 3,000+ players

**Data Structure:**
```json
{
  "eventId": 8,
  "eventName": "Pass",
  "teamId": 1609,
  "playerId": 25508,
  "positions": [
    {"x": 50, "y": 35},
    {"x": 60, "y": 45}
  ],
  "tags": [...],
  "matchId": 2499841
}
```

**Pros:**
- Large volume
- Multiple leagues
- Consistent format

**Cons:**
- No freeze frames
- Event data only
- Older (2017/18)

---

### Metrica Sports Tracking Data

**Access:** https://github.com/metrica-sports/sample-data

**Coverage:**
- 3 complete matches
- Full tracking (25 Hz)
- Synchronized events
- Anonymized teams

**Data Structure:**
```csv
# Tracking
Time, Period, HomePId, X, Y, VX, VY, ...

# Events  
Period, Time, Type, From, To, Start X, Start Y, End X, End Y
```

**Pros:**
- Full tracking data
- Includes velocity
- Event synchronization
- Great for validation

**Cons:**
- Only 3 matches
- Anonymized (can't link to real teams)
- Not enough for training

---

### SkillCorner Broadcast Tracking

**Access:** https://github.com/SkillCorner/opendata

**Coverage:**
- 9 Champions League matches
- Tracking from broadcast video
- Includes velocity estimates

**Pros:**
- Broadcast tracking (accessible data source)
- More matches than Metrica

**Cons:**
- Lower accuracy than optical tracking
- Limited coverage

---

### Recommended Data Strategy

**Phase 1-3 (Baseline Model):**
- Use StatsBomb open data
- Focus on competitions with 360 coverage
- Target: 500-1,000 corners

**Phase 4+ (Advanced):**
- Supplement with Wyscout for volume
- Use Metrica for velocity validation
- Consider commercial StatsBomb for full 360 coverage

---

## 🏗️ Architecture Deep Dive

### Graph Construction

#### Node Features (Minimal)
```python
node_features = [
    x,              # Position x (0-120m)
    y,              # Position y (0-80m)
    teammate,       # Boolean: attacking team?
]
```

#### Node Features (Enhanced)
```python
node_features = [
    # Position
    x, y,           # Raw coordinates
    x_norm, y_norm, # Normalized (0-1)
    
    # Movement
    vx, vy,         # Velocity components
    speed,          # Total speed
    
    # Physical
    height,         # Player height (if available)
    
    # Tactical
    is_attacker,    # Team assignment
    role,           # Position (one-hot: GK, DEF, MID, FWD)
    
    # Context
    distance_to_ball,
    distance_to_goal,
    angle_to_goal,
    in_penalty_box,
    near_post,
]
```

#### Edge Construction Methods

**Method 1: Distance-based (Simple)**
```python
def build_edges_distance(positions, threshold=5.0):
    edges = []
    for i in range(len(positions)):
        for j in range(len(positions)):
            if i != j:
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < threshold:
                    edges.append([i, j])
    return edges
```

**Method 2: K-Nearest Neighbors**
```python
from sklearn.neighbors import NearestNeighbors

def build_edges_knn(positions, k=5):
    nn = NearestNeighbors(n_neighbors=k+1)
    nn.fit(positions)
    distances, indices = nn.kneighbors(positions)
    
    edges = []
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:  # Skip self
            edges.append([i, j])
    return edges
```

**Method 3: Fully Connected**
```python
def build_edges_fully_connected(num_nodes):
    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edges.append([i, j])
    return edges
```

**Method 4: Tactical Relationships (Advanced)**
```python
def build_edges_tactical(players, ball_position):
    edges = []
    edge_types = []
    
    for i, player_i in enumerate(players):
        for j, player_j in enumerate(players):
            if i == j:
                continue
            
            # Distance edge
            dist = distance(player_i, player_j)
            if dist < 5.0:
                edges.append([i, j])
                edge_types.append('proximity')
            
            # Marking relationship (different teams, close)
            if player_i['team'] != player_j['team'] and dist < 2.0:
                edges.append([i, j])
                edge_types.append('marking')
            
            # Passing option (same team, clear line)
            if player_i['team'] == player_j['team']:
                if is_clear_path(player_i, player_j, players):
                    edges.append([i, j])
                    edge_types.append('pass_option')
    
    return edges, edge_types
```

**Recommendation:** Start with Method 1, move to Method 4 in Phase 2.

---

### GNN Architectures

#### GCN (Baseline)
```python
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = self.conv3(x, edge_index)
        return x
```

**Pros:** Simple, fast, good baseline  
**Cons:** All neighbors weighted equally

---

#### GAT (Recommended)
```python
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads)
        self.conv3 = GATConv(hidden_channels * heads, out_channels, heads=1)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = self.conv3(x, edge_index)
        return x
```

**Pros:** Learns neighbor importance, better performance  
**Cons:** More parameters, slower

---

#### GraphSAGE (Scalable)
```python
from torch_geometric.nn import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = self.conv3(x, edge_index)
        return x
```

**Pros:** Sampling-based (scalable), multiple aggregators  
**Cons:** More complex implementation

---

### Multi-Task Architecture

```python
class TacticAI_MultiTask(nn.Module):
    def __init__(self, node_features, edge_features, hidden_dim=128):
        super().__init__()
        
        # Shared encoder
        self.conv1 = GATConv(node_features, hidden_dim, heads=4, edge_dim=edge_features)
        self.conv2 = GATConv(hidden_dim*4, hidden_dim, heads=4, edge_dim=edge_features)
        self.conv3 = GATConv(hidden_dim*4, hidden_dim, heads=1, edge_dim=edge_features)
        
        # Task-specific heads
        
        # 1. Receiver prediction (node-level)
        self.receiver_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        
        # 2. Shot prediction (graph-level)
        self.shot_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        
        # 3. Goal prediction (graph-level)
        self.goal_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x, edge_index, edge_attr, batch):
        # Shared encoding
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = self.conv3(x, edge_index, edge_attr)
        
        # Task 1: Receiver (node-level)
        receiver_logits = self.receiver_head(x).squeeze(-1)
        
        # Task 2 & 3: Shot and Goal (graph-level)
        # Pool node features to graph features
        from torch_geometric.nn import global_mean_pool
        graph_features = global_mean_pool(x, batch)
        
        shot_logit = self.shot_head(graph_features).squeeze(-1)
        goal_logit = self.goal_head(graph_features).squeeze(-1)
        
        return {
            'receiver': receiver_logits,
            'shot': torch.sigmoid(shot_logit),
            'goal': torch.sigmoid(goal_logit)
        }

# Multi-task loss
def multi_task_loss(predictions, targets, weights={'receiver': 1.0, 'shot': 0.5, 'goal': 0.3}):
    losses = {}
    
    # Receiver loss (cross-entropy)
    losses['receiver'] = F.cross_entropy(
        predictions['receiver'], 
        targets['receiver']
    )
    
    # Shot loss (binary cross-entropy)
    losses['shot'] = F.binary_cross_entropy(
        predictions['shot'], 
        targets['shot']
    )
    
    # Goal loss (binary cross-entropy)
    losses['goal'] = F.binary_cross_entropy(
        predictions['goal'],
        targets['goal']
    )
    
    # Weighted sum
    total_loss = sum(weights[task] * loss for task, loss in losses.items())
    
    return total_loss, losses
```

---

## 🎯 Evaluation Metrics - Implementation

### Receiver Prediction Metrics

```python
def evaluate_receiver_prediction(model, test_loader, device='cuda'):
    """Comprehensive receiver prediction evaluation"""
    model.eval()
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch.y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Top-1 accuracy
    top1_acc = (all_preds == all_targets).mean()
    
    # Top-3 accuracy
    top3_preds = np.argsort(all_probs, axis=1)[:, -3:]
    top3_acc = np.mean([target in top3_preds[i] for i, target in enumerate(all_targets)])
    
    # Top-5 accuracy
    top5_preds = np.argsort(all_probs, axis=1)[:, -5:]
    top5_acc = np.mean([target in top5_preds[i] for i, target in enumerate(all_targets)])
    
    # Per-player metrics
    from sklearn.metrics import classification_report
    report = classification_report(all_targets, all_preds, output_dict=True)
    
    # Confidence calibration
    confidences = np.max(all_probs, axis=1)
    correct = (all_preds == all_targets)
    
    # Expected Calibration Error (ECE)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        bin_mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i+1])
        if bin_mask.sum() > 0:
            bin_acc = correct[bin_mask].mean()
            bin_conf = confidences[bin_mask].mean()
            ece += np.abs(bin_acc - bin_conf) * bin_mask.sum()
    ece /= len(all_targets)
    
    return {
        'top1_accuracy': top1_acc,
        'top3_accuracy': top3_acc,
        'top5_accuracy': top5_acc,
        'per_player_report': report,
        'expected_calibration_error': ece,
        'mean_confidence': confidences.mean(),
        'predictions': all_preds,
        'targets': all_targets,
        'probabilities': all_probs
    }
```

### Shot Prediction Metrics

```python
def evaluate_shot_prediction(model, test_loader, device='cuda'):
    """Evaluate shot prediction"""
    from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
    
    model.eval()
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outputs = model(batch)
            shot_prob = outputs['shot']
            
            all_probs.extend(shot_prob.cpu().numpy())
            all_targets.extend(batch.shot_target.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    all_preds = (all_probs > 0.5).astype(int)
    
    # Metrics
    auc_roc = roc_auc_score(all_targets, all_probs)
    auc_pr = average_precision_score(all_targets, all_probs)
    
    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Brier score
    brier = np.mean((all_probs - all_targets) ** 2)
    
    return {
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'brier_score': brier,
        'confusion_matrix': {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}
    }
```

### Visualization of Results

```python
def plot_evaluation_results(results, save_path='evaluation_results.png'):
    """Create comprehensive evaluation visualization"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Top-K Accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    accuracies = [
        results['receiver']['top1_accuracy'],
        results['receiver']['top3_accuracy'],
        results['receiver']['top5_accuracy']
    ]
    ax1.bar(['Top-1', 'Top-3', 'Top-5'], accuracies, color=['#FF6B6B', '#4ECDC4', '#95E1D3'])
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Receiver Prediction Accuracy')
    ax1.set_ylim([0, 1])
    for i, v in enumerate(accuracies):
        ax1.text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')
    
    # 2. Confusion Matrix (Shot Prediction)
    ax2 = fig.add_subplot(gs[0, 1])
    cm = results['shot']['confusion_matrix']
    cm_matrix = np.array([[cm['TN'], cm['FP']], [cm['FN'], cm['TP']]])
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=['No Shot', 'Shot'],
                yticklabels=['No Shot', 'Shot'])
    ax2.set_title('Shot Prediction Confusion Matrix')
    ax2.set_ylabel('True')
    ax2.set_xlabel('Predicted')
    
    # 3. ROC Curve
    ax3 = fig.add_subplot(gs[0, 2])
    # Plot ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(results['shot']['targets'], results['shot']['probabilities'])
    ax3.plot(fpr, tpr, linewidth=2, label=f"AUC = {results['shot']['auc_roc']:.3f}")
    ax3.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('ROC Curve (Shot Prediction)')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Calibration Plot
    ax4 = fig.add_subplot(gs[1, 0])
    confidences = np.max(results['receiver']['probabilities'], axis=1)
    correct = (results['receiver']['predictions'] == results['receiver']['targets'])
    
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_accs = []
    bin_confs = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if in_bin.sum() > 0:
            bin_acc = correct[in_bin].mean()
            bin_conf = confidences[in_bin].mean()
            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)
    
    ax4.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Calibration')
    ax4.plot(bin_confs, bin_accs, 'o-', linewidth=2, markersize=8, label='Model')
    ax4.set_xlabel('Confidence')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Calibration Plot')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # 5. Per-Player Performance
    ax5 = fig.add_subplot(gs[1, 1:])
    per_player = results['receiver']['per_player_report']
    players = [str(i) for i in range(22) if str(i) in per_player]
    precisions = [per_player[p]['precision'] for p in players]
    recalls = [per_player[p]['recall'] for p in players]
    
    x = np.arange(len(players))
    width = 0.35
    ax5.bar(x - width/2, precisions, width, label='Precision', alpha=0.8)
    ax5.bar(x + width/2, recalls, width, label='Recall', alpha=0.8)
    ax5.set_xlabel('Player ID')
    ax5.set_ylabel('Score')
    ax5.set_title('Per-Player Precision and Recall')
    ax5.set_xticks(x)
    ax5.set_xticklabels(players, rotation=45)
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Metrics Summary Table
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    metrics_data = [
        ['Metric', 'Receiver', 'Shot', 'Goal'],
        ['Accuracy', f"{results['receiver']['top1_accuracy']:.2%}", 
         f"{results['shot']['accuracy']:.2%}", '-'],
        ['Precision', '-', f"{results['shot']['precision']:.2%}", '-'],
        ['Recall', '-', f"{results['shot']['recall']:.2%}", '-'],
        ['F1 Score', '-', f"{results['shot']['f1_score']:.2%}", '-'],
        ['AUC-ROC', '-', f"{results['shot']['auc_roc']:.3f}", '-'],
    ]
    
    table = ax6.table(cellText=metrics_data, cellLoc='center', loc='center',
                     colWidths=[0.2, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved evaluation results to {save_path}")
    return fig
```

---

## 🔧 Useful Code Snippets

### Data Augmentation

```python
def augment_corner_graph(graph, augmentations=['horizontal', 'vertical', 'both']):
    """Apply symmetry-based augmentations"""
    augmented = [graph]
    
    if 'horizontal' in augmentations:
        # Flip horizontally (left ↔ right)
        g_h = graph.clone()
        g_h.x[:, 0] = 120 - g_h.x[:, 0]  # Flip x coordinate
        augmented.append(g_h)
    
    if 'vertical' in augmentations:
        # Flip vertically (top ↔ bottom)
        g_v = graph.clone()
        g_v.x[:, 1] = 80 - g_v.x[:, 1]  # Flip y coordinate
        augmented.append(g_v)
    
    if 'both' in augmentations:
        # Flip both
        g_hv = graph.clone()
        g_hv.x[:, 0] = 120 - g_hv.x[:, 0]
        g_hv.x[:, 1] = 80 - g_hv.x[:, 1]
        augmented.append(g_hv)
    
    return augmented
```

### Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

# Option 1: Reduce on plateau
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',          # Maximize validation accuracy
    factor=0.5,          # Reduce LR by half
    patience=5,          # Wait 5 epochs
    verbose=True
)

# Option 2: Cosine annealing
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=50,           # Period
    eta_min=1e-6        # Minimum LR
)

# Usage in training loop
for epoch in range(num_epochs):
    train_loss = train_epoch(...)
    val_acc = evaluate(...)
    
    scheduler.step(val_acc)  # For ReduceLROnPlateau
    # scheduler.step()       # For CosineAnnealingLR
```

### Gradient Clipping

```python
# Prevent exploding gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Early Stopping

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
        
        return self.early_stop

# Usage
early_stopping = EarlyStopping(patience=15)

for epoch in range(max_epochs):
    val_acc = evaluate(...)
    
    if early_stopping(val_acc):
        print(f"Early stopping at epoch {epoch}")
        break
```

---

## 📊 Experiment Tracking with Weights & Biases

```python
import wandb

# Initialize
wandb.init(
    project="tacticai",
    config={
        "architecture": "GAT",
        "hidden_dim": 128,
        "learning_rate": 0.001,
        "batch_size": 32,
        "num_epochs": 100,
        "dropout": 0.3,
        "edge_threshold": 5.0,
    }
)

# Log during training
for epoch in range(num_epochs):
    train_loss = train_epoch(...)
    val_acc = evaluate(...)
    
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_accuracy": val_acc,
        "learning_rate": optimizer.param_groups[0]['lr']
    })

# Log model
wandb.save('models/best_model.pth')

# Log visualizations
fig = plot_evaluation_results(...)
wandb.log({"evaluation": wandb.Image(fig)})

# Finish
wandb.finish()
```

---

## 🎯 Key Hyperparameters

Based on TacticAI paper and GNN best practices:

```python
HYPERPARAMETERS = {
    # Architecture
    'hidden_dim': 128,           # 64-256, try [64, 128, 256]
    'num_layers': 3,             # 2-5, typically 3-4
    'heads': 4,                  # For GAT, try [1, 2, 4, 8]
    
    # Regularization
    'dropout': 0.3,              # 0.2-0.5, try [0.2, 0.3, 0.4]
    'weight_decay': 1e-4,        # L2 regularization, try [1e-5, 1e-4, 1e-3]
    
    # Training
    'learning_rate': 0.001,      # try [1e-4, 5e-4, 1e-3]
    'batch_size': 32,            # 16-64, depends on GPU memory
    'max_epochs': 100,
    'early_stopping_patience': 15,
    
    # Graph construction
    'edge_threshold': 5.0,       # meters, try [3.0, 5.0, 7.0]
    'k_neighbors': 5,            # For KNN edges, try [3, 5, 7]
    
    # Multi-task weights
    'receiver_weight': 1.0,
    'shot_weight': 0.5,          # try [0.3, 0.5, 1.0]
    'goal_weight': 0.3,          # try [0.1, 0.3, 0.5]
}
```

---

## 💡 Pro Tips

### 1. Start with Minimal Feature Set
Don't add all features at once. Start with (x, y, teammate) and add incrementally.

### 2. Visualize Early and Often
After each change, visualize a few predictions to build intuition.

### 3. Use Validation Set Properly
Never touch test set until final evaluation. Use validation for hyperparameter tuning.

### 4. Check for Data Leakage
Ensure train/val/test splits are by match ID, not corner ID.

### 5. Monitor Both Tasks
In multi-task learning, track all task metrics. Don't optimize only for one.

### 6. Save Checkpoints Frequently
Save model every N epochs and keep best model based on validation.

### 7. Log Everything
Use Weights & Biases or similar. You'll thank yourself later.

### 8. Test with Synthetic Data First
Create a simple synthetic corner scenario and verify model learns it.

---

## 📖 Additional Reading

### Blogs & Tutorials
- Distill.pub: "A Gentle Introduction to Graph Neural Networks"
- PyTorch Geometric Tutorials
- StatsBomb's Technical Blog
- Friends of Tracking YouTube Channel

### Books
- "Graph Representation Learning" by William L. Hamilton (free online)
- "Geometric Deep Learning" textbook (geometricdeeplearning.com)
- "Soccermatics" by David Sumpter

### Courses
- Stanford CS224W: Machine Learning with Graphs
- Fast.ai Practical Deep Learning
- Coursera: Machine Learning Specialization

---

This reference document should serve as your go-to resource throughout the project. Bookmark it and refer back frequently!
