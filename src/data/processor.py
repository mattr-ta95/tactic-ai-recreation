"""
Data processing utilities for TacticAI
Converts corner kicks to graph representations

Note: This implementation works with shots that have freeze frames,
linking back to corner passes to extract receiver labels.
"""

import math

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional

class CornerKickProcessor:
    """Process corner kicks into graph representations"""
    
    def __init__(self,
                 distance_threshold: float = 5.0,
                 normalize_positions: bool = True,
                 use_enhanced_features: bool = False,
                 use_knn_edges: bool = False,
                 knn_k: int = 8,
                 use_role_features: bool = False,
                 use_positional_context: bool = False):
        """
        Initialize processor

        Args:
            distance_threshold: Max distance for edge connections (meters)
            normalize_positions: Whether to normalize coordinates to [0, 1]
            use_enhanced_features: Whether to add distance-to-goal, angle features, etc.
            use_knn_edges: Whether to use K-nearest neighbors instead of distance threshold
            knn_k: Number of nearest neighbors for KNN edges
            use_role_features: Whether to add player role (GK/DEF/MID/FWD) one-hot features
            use_positional_context: Whether to add positional context features
        """
        self.distance_threshold = distance_threshold
        self.normalize_positions = normalize_positions
        self.use_enhanced_features = use_enhanced_features
        self.use_knn_edges = use_knn_edges
        self.knn_k = knn_k
        self.use_role_features = use_role_features
        self.use_positional_context = use_positional_context

        # Pitch dimensions (StatsBomb coordinates)
        self.pitch_length = 120.0
        self.pitch_width = 80.0

    @staticmethod
    def _get_player_role(position_info) -> tuple:
        """
        Map StatsBomb position to general role (GK/DEF/MID/FWD)

        Args:
            position_info: Dictionary with 'id' and 'name' keys

        Returns:
            Tuple of (is_gk, is_def, is_mid, is_fwd) one-hot encoding
        """
        if not position_info or 'id' not in position_info:
            # Default to midfielder if position unknown
            return (0.0, 0.0, 1.0, 0.0)

        pos_id = position_info['id']

        # Map position ID to role:
        # 1: Goalkeeper
        # 2-8: Defenders (backs, wing backs)
        # 9-21: Midfielders (defensive, center, attacking, wings)
        # 22-25: Forwards (center forward, striker)

        if pos_id == 1:
            return (1.0, 0.0, 0.0, 0.0)  # GK
        elif 2 <= pos_id <= 8:
            return (0.0, 1.0, 0.0, 0.0)  # DEF
        elif 9 <= pos_id <= 21:
            return (0.0, 0.0, 1.0, 0.0)  # MID
        elif 22 <= pos_id <= 25:
            return (0.0, 0.0, 0.0, 1.0)  # FWD
        else:
            # Unknown position - default to midfielder
            return (0.0, 0.0, 1.0, 0.0)

    def _compute_positional_context(self, freeze_frame: List[Dict], player_idx: int) -> Tuple[float, float, float]:
        """
        Compute positional context features for a player.

        These features capture the player's spatial relationship to teammates and opponents:
        - Distance to nearest teammate (isolation indicator)
        - Distance to nearest opponent (pressure indicator)
        - Positional depth relative to team centroid (forward/back positioning)

        Args:
            freeze_frame: List of player dictionaries with 'location' and 'teammate' keys
            player_idx: Index of the player in freeze_frame

        Returns:
            Tuple of (dist_to_nearest_teammate, dist_to_nearest_opponent, positional_depth)
            All values are normalized by pitch_length to [0, ~1] range
        """
        player = freeze_frame[player_idx]
        player_pos = np.array(player['location'])
        is_teammate = player['teammate']

        # Separate teammates and opponents
        teammate_positions = []
        opponent_positions = []

        for i, p in enumerate(freeze_frame):
            if i == player_idx:
                continue  # Skip self
            pos = np.array(p['location'])
            if p['teammate'] == is_teammate:
                teammate_positions.append(pos)
            else:
                opponent_positions.append(pos)

        # Distance to nearest teammate (excluding self)
        if len(teammate_positions) > 0:
            teammate_dists = [np.linalg.norm(player_pos - tp) for tp in teammate_positions]
            dist_to_nearest_teammate = min(teammate_dists) / self.pitch_length
        else:
            dist_to_nearest_teammate = 1.0  # Max distance if no teammates

        # Distance to nearest opponent
        if len(opponent_positions) > 0:
            opponent_dists = [np.linalg.norm(player_pos - op) for op in opponent_positions]
            dist_to_nearest_opponent = min(opponent_dists) / self.pitch_length
        else:
            dist_to_nearest_opponent = 1.0  # Max distance if no opponents

        # Positional depth: how far forward/back relative to team centroid
        # Positive = ahead of team avg (more attacking), Negative = behind (more defensive)
        same_team_positions = [player_pos] + teammate_positions  # Include self
        team_centroid_x = np.mean([pos[0] for pos in same_team_positions])
        positional_depth = (player_pos[0] - team_centroid_x) / self.pitch_length

        return dist_to_nearest_teammate, dist_to_nearest_opponent, positional_depth

    def corner_to_graph(self, corner_row) -> Data:
        """
        Convert a corner kick/shot to a PyG graph

        Args:
            corner_row: Row from dataframe with 'freeze_frame_parsed'

        Returns:
            Data: PyTorch Geometric Data object
        """
        freeze_frame = corner_row['freeze_frame_parsed']

        if not isinstance(freeze_frame, list) or len(freeze_frame) == 0:
            raise ValueError("Invalid freeze frame data")

        # Extract node features
        # Get corner location (for shots linked to corners, use pass_end_location)
        corner_location = None
        
        # Try corner_pass_end_location first (from linked corners)
        if 'corner_pass_end_location' in corner_row.index:
            try:
                loc_val = corner_row['corner_pass_end_location']
                if pd.notna(loc_val):
                    if isinstance(loc_val, str):
                        try:
                            import ast
                            corner_location = ast.literal_eval(loc_val)
                        except:
                            corner_location = None
                    elif isinstance(loc_val, (list, tuple)):
                        corner_location = loc_val
            except:
                pass
        
        # Fallback to shot location if corner location not available
        if corner_location is None and 'location' in corner_row.index:
            try:
                loc_val = corner_row['location']
                if pd.notna(loc_val):
                    if isinstance(loc_val, str):
                        try:
                            import ast
                            corner_location = ast.literal_eval(loc_val)
                        except:
                            corner_location = None
                    elif isinstance(loc_val, (list, tuple)):
                        corner_location = loc_val
            except:
                corner_location = None
        
        # Goal center location (for scoring)
        goal_center = np.array([self.pitch_length, self.pitch_width / 2])
        
        node_features = []
        for player in freeze_frame:
            x, y = player['location']
            player_pos = np.array([x, y])

            if self.normalize_positions:
                x_norm = x / self.pitch_length
                y_norm = y / self.pitch_width
            else:
                x_norm = x
                y_norm = y

            # Basic features
            features = [
                x_norm,                          # Position x (normalized)
                y_norm,                          # Position y (normalized)
                float(player['teammate'])        # 1 for attacker, 0 for defender
            ]
            
            # Enhanced features (if enabled)
            if hasattr(self, 'use_enhanced_features') and self.use_enhanced_features:
                # Distance to goal center
                dist_to_goal = np.linalg.norm(player_pos - goal_center) / self.pitch_length  # Normalized
                features.append(dist_to_goal)

                # Distance to corner location (if available)
                if corner_location and isinstance(corner_location, (list, tuple)) and len(corner_location) >= 2:
                    corner_pos = np.array([corner_location[0], corner_location[1]])
                    dist_to_corner = np.linalg.norm(player_pos - corner_pos) / self.pitch_length
                    features.append(dist_to_corner)
                else:
                    features.append(0.0)  # Default if not available

                # Angle to goal (relative direction)
                angle_to_goal = np.arctan2(goal_center[1] - y, goal_center[0] - x) / np.pi  # Normalized to [-1, 1]
                features.append(angle_to_goal)

                # In penalty box (rough approximation)
                in_box = (x >= 102.0 and 18.0 <= y <= 62.0)  # Penalty box area
                features.append(float(in_box))

            # Player role features (if enabled)
            if hasattr(self, 'use_role_features') and self.use_role_features:
                position_info = player.get('position')
                is_gk, is_def, is_mid, is_fwd = self._get_player_role(position_info)
                features.extend([is_gk, is_def, is_mid, is_fwd])

            node_features.append(features)

        # Second pass: Compute positional context features (requires all positions)
        if hasattr(self, 'use_positional_context') and self.use_positional_context:
            for idx in range(len(freeze_frame)):
                dist_tm, dist_opp, depth = self._compute_positional_context(freeze_frame, idx)
                node_features[idx].extend([dist_tm, dist_opp, depth])

        x = torch.tensor(node_features, dtype=torch.float)

        # Create edges with features (connect nearby players)
        edge_index, edge_attr = self._build_edges(freeze_frame)

        # Create Data object with edge attributes
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # Add metadata
        graph.match_id = corner_row.get('match_id', -1)
        graph.num_players = len(freeze_frame)
        # Store corner location for augmentation feature recomputation
        if corner_location and isinstance(corner_location, (list, tuple)) and len(corner_location) >= 2:
            graph.corner_location = [float(corner_location[0]), float(corner_location[1])]
        else:
            graph.corner_location = [120.0, 0.0]  # Default right corner

        # Add receiver label
        # For shots linked to corners: use corner_pass_recipient_id if available
        # Otherwise: use pass_recipient_id or pass_end_location fallback
        if 'corner_pass_recipient_id' in corner_row.index and pd.notna(corner_row.get('corner_pass_recipient_id')):
            # This is a shot linked to a corner - use corner's recipient_id
            # Create temporary event row with recipient info for matching
            temp_event = corner_row.copy()
            temp_event['pass_recipient_id'] = corner_row['corner_pass_recipient_id']
            temp_event['pass_end_location'] = corner_row.get('corner_pass_end_location')
            receiver_idx = self._find_receiver(freeze_frame, temp_event)
        else:
            # Standard corner pass or shot - use existing method
            receiver_idx = self._find_receiver(freeze_frame, corner_row)
        
        if receiver_idx is not None:
            graph.y = torch.tensor([receiver_idx], dtype=torch.long)

        return graph
    
    def _find_receiver(self, freeze_frame: List[Dict], event_row) -> int:
        """
        Find receiver by matching pass_recipient_id to freeze frame player IDs.
        Uses multiple fallback strategies based on data availability.

        Args:
            freeze_frame: List of player dictionaries with 'player' -> {'id': ...}
            event_row: Event data with pass_recipient_id and pass_end_location

        Returns:
            Index of receiver player in freeze_frame, or None if not found
        """
        import pandas as pd
        
        # METHOD 1: Use pass_recipient_id if available (BEST - ground truth)
        recipient_id = event_row.get('pass_recipient_id')
        if pd.notna(recipient_id) and recipient_id is not None:
            recipient_id = int(recipient_id)
            
            # Search freeze frame for matching player ID
            for i, player in enumerate(freeze_frame):
                if isinstance(player, dict):
                    player_info = player.get('player', {})
                    if isinstance(player_info, dict):
                        player_id = player_info.get('id')
                        if player_id is not None and int(player_id) == recipient_id:
                            return i
            
            # Recipient not in freeze frame (off-camera or not captured)
            # This is acceptable - we'll return None and exclude this example
        
        # METHOD 2: Fallback - use pass_end_location + distance (HEURISTIC)
        pass_end = event_row.get('pass_end_location')
        if pass_end is not None and pd.notna(pass_end):
            # Parse if string
            if isinstance(pass_end, str):
                try:
                    import ast
                    pass_end = ast.literal_eval(pass_end)
                except:
                    pass_end = None
            
            if pass_end and isinstance(pass_end, (list, tuple)) and len(pass_end) >= 2:
                end_loc = np.array([pass_end[0], pass_end[1]])
                
                # Find closest teammate to pass end location
                min_dist = float('inf')
                receiver_idx = None
                
                for i, player in enumerate(freeze_frame):
                    if isinstance(player, dict) and player.get('teammate', False):
                        player_loc = np.array(player['location'])
                        dist = np.linalg.norm(end_loc - player_loc)
                        
                        if dist < min_dist:
                            min_dist = dist
                            receiver_idx = i
                
                # Use if reasonably close (relaxed threshold for shot freeze frames)
                # Note: Shot freeze frames capture players at moment of shot, which may be
                # different from corner pass moment, so we use a more lenient threshold
                threshold = 10.0  # Increased from 5.0 to handle temporal gap
                if receiver_idx is not None and min_dist < threshold:
                    return receiver_idx
        
        # METHOD 3: Final fallback - closest to corner location (LEAST ACCURATE)
        # We'll return None instead to maintain data quality
        # Better to exclude than use bad labels
        return None

    def _build_edges(self, freeze_frame: List[Dict]):
        """
        Build edge connections and features based on player distances or K-nearest neighbors

        Args:
            freeze_frame: List of player dictionaries

        Returns:
            edge_index: Tensor of shape [2, num_edges]
            edge_attr: Tensor of shape [num_edges, 3] with [distance, angle, same_team]
        """
        num_players = len(freeze_frame)
        
        if self.use_knn_edges:
            # Use K-nearest neighbors with edge features
            try:
                from sklearn.neighbors import NearestNeighbors

                positions = np.array([p['location'] for p in freeze_frame])

                # Use k+1 to include self, then skip it
                k = min(self.knn_k + 1, num_players)
                nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
                nn.fit(positions)
                distances, indices = nn.kneighbors(positions)

                edge_index = []
                edge_features = []
                for i, (neighbors, dists) in enumerate(zip(indices, distances)):
                    pos_i = positions[i]
                    teammate_i = freeze_frame[i]['teammate']

                    for j_idx, (j, dist) in enumerate(zip(neighbors[1:], dists[1:])):  # Skip self
                        j = int(j)
                        edge_index.append([i, j])

                        # Compute edge features
                        pos_j = positions[j]
                        dist_norm = dist / self.pitch_length
                        angle = np.arctan2(pos_j[1] - pos_i[1], pos_j[0] - pos_i[0]) / np.pi
                        same_team = float(teammate_i == freeze_frame[j]['teammate'])

                        edge_features.append([dist_norm, angle, same_team])

                if len(edge_index) == 0:
                    # Fallback: create self-loops with neutral features
                    edge_index = [[i, i] for i in range(num_players)]
                    edge_features = [[0.0, 0.0, 1.0] for _ in range(num_players)]

                edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                edge_attr_tensor = torch.tensor(edge_features, dtype=torch.float)

                return edge_index_tensor, edge_attr_tensor
            except ImportError:
                # Fallback to distance-based if sklearn not available
                print("Warning: sklearn not available, falling back to distance-based edges")
                self.use_knn_edges = False
        
        # Distance-based edges (default) with edge features
        edge_index = []
        edge_features = []

        for i in range(num_players):
            pos_i = np.array(freeze_frame[i]['location'])
            teammate_i = freeze_frame[i]['teammate']

            for j in range(num_players):
                if i == j:
                    continue

                pos_j = np.array(freeze_frame[j]['location'])
                teammate_j = freeze_frame[j]['teammate']
                distance = np.linalg.norm(pos_i - pos_j)

                if distance < self.distance_threshold:
                    edge_index.append([i, j])

                    # Compute edge features:
                    # 1. Normalized distance
                    dist_norm = distance / self.pitch_length

                    # 2. Angle between players (relative direction)
                    angle = np.arctan2(pos_j[1] - pos_i[1], pos_j[0] - pos_i[0]) / np.pi

                    # 3. Same team indicator (1 if same team, 0 if opponents)
                    same_team = float(teammate_i == teammate_j)

                    edge_features.append([dist_norm, angle, same_team])

        if len(edge_index) == 0:
            # If no edges, create self-loops with neutral features
            edge_index = [[i, i] for i in range(num_players)]
            edge_features = [[0.0, 0.0, 1.0] for _ in range(num_players)]  # Self-loop = same team

        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr_tensor = torch.tensor(edge_features, dtype=torch.float)

        return edge_index_tensor, edge_attr_tensor
    
    def create_dataset(self, corners_df) -> List[Data]:
        """
        Convert all corners to graphs
        
        Args:
            corners_df: DataFrame of corners with freeze_frame_parsed
        
        Returns:
            List of PyG Data objects
        """
        dataset = []
        failed = 0
        
        print(f"Processing {len(corners_df)} corners...")
        
        for idx, corner in corners_df.iterrows():
            try:
                graph = self.corner_to_graph(corner)
                dataset.append(graph)
            except Exception as e:
                failed += 1
                if failed <= 5:  # Only print first 5 errors
                    print(f"  Warning: Failed to process corner {idx}: {e}")
        
        print(f"✅ Created {len(dataset)} graphs ({failed} failed)")
        
        return dataset


def augment_graph(graph: Data,
                  augmentation_type: str = 'horizontal') -> Data:
    """
    Apply symmetry-based data augmentation with proper edge feature updates.

    IMPORTANT: This function updates both node features AND edge attributes
    to maintain graph consistency after coordinate flips.

    Args:
        graph: Original graph
        augmentation_type: 'horizontal', 'vertical', or 'both'

    Returns:
        Augmented graph with updated node features and edge attributes
    """
    aug_graph = graph.clone()
    num_features = aug_graph.x.shape[1]

    # --- Step 1: Flip coordinates ---
    if augmentation_type in ['horizontal', 'both']:
        aug_graph.x[:, 0] = 1.0 - aug_graph.x[:, 0]

    if augmentation_type in ['vertical', 'both']:
        aug_graph.x[:, 1] = 1.0 - aug_graph.x[:, 1]

    # --- Step 2: Recompute derived features from flipped positions ---
    if num_features > 3:
        # Denormalize to pitch coordinates
        x_pos = aug_graph.x[:, 0] * 120.0
        y_pos = aug_graph.x[:, 1] * 80.0
        positions = torch.stack([x_pos, y_pos], dim=1)

        # Feature 3: dist_to_goal (goal at 120, 40)
        goal = torch.tensor([120.0, 40.0], device=aug_graph.x.device)
        aug_graph.x[:, 3] = torch.norm(positions - goal, dim=1) / 120.0

    if num_features > 4:
        # Feature 4: dist_to_corner
        # Determine flipped corner location
        corner_loc = getattr(graph, 'corner_location', [120.0, 0.0])
        cx, cy = float(corner_loc[0]), float(corner_loc[1])
        if augmentation_type in ['horizontal', 'both']:
            cx = 120.0 - cx
        if augmentation_type in ['vertical', 'both']:
            cy = 80.0 - cy
        corner = torch.tensor([cx, cy], device=aug_graph.x.device)
        aug_graph.x[:, 4] = torch.norm(positions - corner, dim=1) / 120.0

    if num_features > 5:
        # Feature 5: angle_to_goal
        dx = 120.0 - x_pos
        dy = 40.0 - y_pos
        aug_graph.x[:, 5] = torch.atan2(dy, dx) / math.pi

    if num_features > 6:
        # Feature 6: in_box (unified with feature_recompute.py constants)
        aug_graph.x[:, 6] = ((x_pos >= 102.0) & (y_pos >= 18.0) & (y_pos <= 62.0)).float()

    if num_features > 13:
        # Features 11-13: positional context (nearest teammate/opponent, depth)
        teammate_mask = aug_graph.x[:, 2] > 0.5
        num_players = positions.shape[0]
        device = aug_graph.x.device

        # Pairwise distances
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)
        distances = torch.norm(diff, dim=2)
        distances = distances + torch.eye(num_players, device=device) * 1e6

        for i in range(num_players):
            is_tm = teammate_mask[i]
            same_team = (teammate_mask == is_tm)
            same_team[i] = False
            diff_team = (teammate_mask != is_tm)

            # Feature 11: dist to nearest teammate
            if same_team.any():
                aug_graph.x[i, 11] = distances[i, same_team].min() / 120.0
            else:
                aug_graph.x[i, 11] = 1.0

            # Feature 12: dist to nearest opponent
            if diff_team.any():
                aug_graph.x[i, 12] = distances[i, diff_team].min() / 120.0
            else:
                aug_graph.x[i, 12] = 1.0

        # Feature 13: positional depth relative to team centroid
        for is_attacker in [True, False]:
            team_mask = (teammate_mask == is_attacker)
            if team_mask.sum() > 0:
                centroid_x = positions[team_mask, 0].mean()
                aug_graph.x[team_mask, 13] = (positions[team_mask, 0] - centroid_x) / 120.0

    # --- Step 3: Update edge attributes ---
    if hasattr(aug_graph, 'edge_attr') and aug_graph.edge_attr is not None:
        # Angle (index 1) needs to be recomputed from flipped positions
        if augmentation_type == 'horizontal':
            angles = aug_graph.edge_attr[:, 1]
            aug_graph.edge_attr[:, 1] = torch.where(
                angles >= 0, 1.0 - angles, -1.0 - angles
            )
        elif augmentation_type == 'vertical':
            aug_graph.edge_attr[:, 1] = -aug_graph.edge_attr[:, 1]
        elif augmentation_type == 'both':
            angles = aug_graph.edge_attr[:, 1]
            new_angles = angles + 1.0
            new_angles = torch.where(new_angles > 1.0, new_angles - 2.0, new_angles)
            aug_graph.edge_attr[:, 1] = new_angles

    return aug_graph


def get_data_statistics(dataset: List[Data]) -> Dict:
    """
    Compute statistics about the dataset
    
    Args:
        dataset: List of graph Data objects
    
    Returns:
        Dictionary of statistics
    """
    num_players_list = [data.num_players for data in dataset]
    num_edges_list = [data.edge_index.shape[1] for data in dataset]
    
    # Count attackers vs defenders
    num_attackers = []
    num_defenders = []
    
    for data in dataset:
        teammates = data.x[:, 2].numpy()  # Teammate feature
        num_attackers.append(teammates.sum())
        num_defenders.append((1 - teammates).sum())
    
    stats = {
        'num_graphs': len(dataset),
        'avg_players': np.mean(num_players_list),
        'avg_edges': np.mean(num_edges_list),
        'avg_attackers': np.mean(num_attackers),
        'avg_defenders': np.mean(num_defenders),
        'min_players': np.min(num_players_list),
        'max_players': np.max(num_players_list),
    }
    
    return stats
