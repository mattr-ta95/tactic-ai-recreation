"""
Simple GNN model for corner kick analysis
Phase 1: Baseline receiver prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.nn import global_mean_pool, global_max_pool


class SimpleCornerGNN(nn.Module):
    """
    Baseline GNN for corner kick receiver prediction
    Uses GCN layers with simple architecture
    """
    
    def __init__(self, 
                 node_features: int = 3,
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 dropout: float = 0.3):
        """
        Initialize model
        
        Args:
            node_features: Number of input node features
            hidden_dim: Hidden dimension size
            num_layers: Number of GNN layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_features, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Prediction head (node-level)
        self.receiver_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, edge_index, batch=None, edge_attr=None):
        """
        Forward pass

        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment (for batched graphs)
            edge_attr: Edge features (accepted but unused by GCN layers)

        Returns:
            receiver_logits: Logits for each player [num_nodes]
        """
        # Apply GNN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Predict receiver probability for each node
        receiver_logits = self.receiver_head(x).squeeze(-1)

        return receiver_logits


class GATCornerNet(nn.Module):
    """
    Graph Attention Network for corner analysis with residual connections
    and optional edge feature support.

    Features:
    - Multi-head attention for learning important player relationships
    - Residual connections in middle layers for better gradient flow
    - Dropout for regularization
    - Optional edge features (distance, angle, same_team) for attention
    """

    def __init__(self,
                 node_features: int = 3,
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 heads: int = 4,
                 dropout: float = 0.3,
                 use_residual: bool = True,
                 edge_dim: int = None):
        """
        Initialize GAT model with optional edge features

        Args:
            node_features: Number of input node features
            hidden_dim: Hidden dimension per head
            num_layers: Number of GAT layers
            heads: Number of attention heads
            dropout: Dropout probability
            use_residual: Whether to use residual connections in middle layers
            edge_dim: Edge feature dimensionality (None = no edge features)
        """
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.use_residual = use_residual
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.edge_dim = edge_dim

        # GAT layers with optional edge features
        self.convs = nn.ModuleList()

        # First layer
        self.convs.append(
            GATConv(node_features, hidden_dim, heads=heads, dropout=dropout,
                    edge_dim=edge_dim)
        )

        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout,
                        edge_dim=edge_dim)
            )

        # Last layer (single head)
        self.convs.append(
            GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout,
                    edge_dim=edge_dim)
        )

        # Prediction head
        self.receiver_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x, edge_index, batch=None, edge_attr=None):
        """
        Forward pass with optional residual connections and edge features.

        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment (for batched graphs)
            edge_attr: Edge features [num_edges, edge_dim] (optional)

        Returns:
            receiver_logits: Logits for each player [num_nodes]
        """
        # First layer (no residual - dimension change)
        x = self.convs[0](x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Middle layers with residual connections
        for i, conv in enumerate(self.convs[1:-1], start=1):
            if self.use_residual:
                # Store input for residual connection
                residual = x
                # Apply conv with edge features
                x = conv(x, edge_index, edge_attr=edge_attr)
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                # Add residual (both have same shape: [N, hidden_dim * heads])
                x = x + residual
            else:
                x = conv(x, edge_index, edge_attr=edge_attr)
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Last layer (no residual - dimension change)
        x = self.convs[-1](x, edge_index, edge_attr=edge_attr)

        # Predict receiver
        receiver_logits = self.receiver_head(x).squeeze(-1)

        return receiver_logits


class MultiTaskCornerGNN(nn.Module):
    """
    Multi-task GNN predicting:
    1. Receiver (node-level)
    2. Shot probability (graph-level)
    3. Goal probability (graph-level)
    """
    
    def __init__(self,
                 node_features: int = 3,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.3):
        """Initialize multi-task model"""
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Shared GNN encoder
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_features, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Task-specific heads
        
        # 1. Receiver prediction (node-level)
        self.receiver_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        # 2. Shot prediction (graph-level)
        self.shot_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        # 3. Goal prediction (graph-level)
        self.goal_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x, edge_index, batch=None, edge_attr=None):
        """
        Forward pass

        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment (for batched graphs)
            edge_attr: Edge features (accepted but unused by GCN layers)

        Returns:
            Dictionary with receiver, shot, and goal predictions
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Shared encoding
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Node-level prediction
        receiver_logits = self.receiver_head(x).squeeze(-1)
        
        # Graph-level predictions (pool nodes)
        graph_features = global_mean_pool(x, batch)
        
        shot_logit = self.shot_head(graph_features).squeeze(-1)
        goal_logit = self.goal_head(graph_features).squeeze(-1)
        
        return {
            'receiver': receiver_logits,
            'shot': torch.sigmoid(shot_logit),
            'goal': torch.sigmoid(goal_logit)
        }


def get_model(model_type: str = 'simple', **kwargs):
    """
    Factory function to create models
    
    Args:
        model_type: 'simple', 'gat', or 'multitask'
        **kwargs: Model-specific arguments
    
    Returns:
        Model instance
    """
    models = {
        'simple': SimpleCornerGNN,
        'gat': GATCornerNet,
        'multitask': MultiTaskCornerGNN
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return models[model_type](**kwargs)


if __name__ == "__main__":
    # Test models
    print("Testing TacticAI models...")

    # Create dummy data
    x = torch.randn(22, 3)  # 22 players, 3 features
    edge_index = torch.randint(0, 22, (2, 50))  # Random edges
    edge_attr = torch.randn(50, 3)  # Edge features
    batch = torch.zeros(22, dtype=torch.long)  # Single graph

    # Test SimpleCornerGNN
    print("\n1. SimpleCornerGNN:")
    model1 = SimpleCornerGNN(node_features=3, hidden_dim=64)
    out1 = model1(x, edge_index, batch)
    print(f"   Output shape: {out1.shape}")  # Should be [22]
    print(f"   Parameters: {sum(p.numel() for p in model1.parameters())}")

    # Test GATCornerNet (without edge features)
    print("\n2. GATCornerNet (no edge features):")
    model2 = GATCornerNet(node_features=3, hidden_dim=64)
    out2 = model2(x, edge_index, batch)
    print(f"   Output shape: {out2.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model2.parameters())}")

    # Test GATCornerNet with edge features
    print("\n3. GATCornerNet (with edge features):")
    model2_edge = GATCornerNet(node_features=3, hidden_dim=64, edge_dim=3)
    out2_edge = model2_edge(x, edge_index, batch, edge_attr=edge_attr)
    print(f"   Output shape: {out2_edge.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model2_edge.parameters())}")

    # Test GATCornerNet with Phase 2 features (14 node features, 3 edge features)
    print("\n4. GATCornerNet (Phase 2 config):")
    x_phase2 = torch.randn(22, 14)  # 14 features
    model2_phase2 = GATCornerNet(node_features=14, hidden_dim=128, edge_dim=3)
    out2_phase2 = model2_phase2(x_phase2, edge_index, batch, edge_attr=edge_attr)
    print(f"   Output shape: {out2_phase2.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model2_phase2.parameters())}")

    # Test MultiTaskCornerGNN
    print("\n5. MultiTaskCornerGNN:")
    model3 = MultiTaskCornerGNN(node_features=3, hidden_dim=128)
    out3 = model3(x, edge_index, batch)
    print(f"   Receiver shape: {out3['receiver'].shape}")
    print(f"   Shot shape: {out3['shot'].shape}")
    print(f"   Goal shape: {out3['goal'].shape}")
    print(f"   Parameters: {sum(p.numel() for p in model3.parameters())}")

    print("\n✅ All models working correctly!")
