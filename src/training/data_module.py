"""PyTorch Lightning DataModule for TacticAI corner kick data."""

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import os
from typing import Optional, List

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.processor import CornerKickProcessor, augment_graph


class TacticAIDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for TacticAI corner kick data.

    Features:
    - Match-based train/val/test splitting (prevents data leakage)
    - Optional data augmentation (horizontal flip)
    - Support for combined real + synthetic data
    - K-fold cross-validation support

    Args:
        data_dir: Directory containing processed data
        batch_size: Batch size for DataLoaders
        num_workers: Number of data loading workers
        distance_threshold: Distance threshold for edge construction
        use_enhanced_features: Use enhanced node features
        use_role_features: Use player role features (GK/DEF/MID/FWD)
        use_positional_context: Use positional context features
        use_knn_edges: Use KNN for edge construction
        knn_k: K for KNN edges
        use_augmentation: Apply horizontal flip augmentation
        val_split: Validation split ratio
        test_split: Test split ratio
        seed: Random seed for reproducibility
        fold_idx: Fold index for cross-validation (None for standard split)
        num_folds: Number of folds for cross-validation
    """

    def __init__(
        self,
        data_dir: str = 'data/processed',
        batch_size: int = 32,
        num_workers: int = 0,
        distance_threshold: float = 5.0,
        use_enhanced_features: bool = True,
        use_role_features: bool = True,
        use_positional_context: bool = True,
        use_knn_edges: bool = False,
        knn_k: int = 8,
        use_augmentation: bool = True,
        val_split: float = 0.2,
        test_split: float = 0.2,
        seed: int = 42,
        fold_idx: Optional[int] = None,
        num_folds: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = data_dir
        self.processor = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def setup(self, stage: Optional[str] = None):
        """Load and prepare datasets."""
        # Initialize processor
        self.processor = CornerKickProcessor(
            distance_threshold=self.hparams.distance_threshold,
            normalize_positions=True,
            use_enhanced_features=self.hparams.use_enhanced_features,
            use_knn_edges=self.hparams.use_knn_edges,
            knn_k=self.hparams.knn_k,
            use_role_features=self.hparams.use_role_features,
            use_positional_context=self.hparams.use_positional_context
        )

        # Load data (prefer combined, fall back to training_shots)
        combined_path = os.path.join(self.data_dir, 'training_shots_combined.pkl')
        training_path = os.path.join(self.data_dir, 'training_shots.pkl')
        shots_path = os.path.join(self.data_dir, 'shots_freeze.pkl')

        if os.path.exists(combined_path):
            corners = pd.read_pickle(combined_path)
            print(f"Loaded combined data: {len(corners)} examples")
        elif os.path.exists(training_path):
            corners = pd.read_pickle(training_path)
            print(f"Loaded training data: {len(corners)} examples")
        elif os.path.exists(shots_path):
            corners = pd.read_pickle(shots_path)
            print(f"Loaded shots data: {len(corners)} examples")
        else:
            raise FileNotFoundError(f"No training data found in {self.data_dir}")

        # Convert to graphs
        print("Converting to graphs...")
        dataset = self.processor.create_dataset(corners)

        # Filter for labeled examples
        dataset = [g for g in dataset if hasattr(g, 'y') and g.y is not None]

        if len(dataset) == 0:
            raise ValueError("No labeled graphs in dataset")

        print(f"Created {len(dataset)} labeled graphs")

        # Get match IDs for splitting
        match_ids = list(set([g.match_id if hasattr(g, 'match_id') else i
                             for i, g in enumerate(dataset)]))

        if self.hparams.fold_idx is not None:
            # K-fold cross-validation split
            self._setup_kfold_split(dataset, match_ids)
        else:
            # Standard train/val/test split
            self._setup_standard_split(dataset, match_ids)

        print(f"Train: {len(self.train_data)}, Val: {len(self.val_data)}, Test: {len(self.test_data)}")

    def _setup_standard_split(self, dataset, match_ids):
        """Standard 60/20/20 split by match."""
        # First split: train vs (val + test)
        train_matches, temp_matches = train_test_split(
            match_ids,
            test_size=self.hparams.val_split + self.hparams.test_split,
            random_state=self.hparams.seed
        )

        # Second split: val vs test
        val_matches, test_matches = train_test_split(
            temp_matches,
            test_size=self.hparams.test_split / (self.hparams.val_split + self.hparams.test_split),
            random_state=self.hparams.seed
        )

        train_matches_set = set(train_matches)
        val_matches_set = set(val_matches)
        test_matches_set = set(test_matches)

        self.train_data = [g for g in dataset if
                          (g.match_id if hasattr(g, 'match_id') else dataset.index(g)) in train_matches_set]
        self.val_data = [g for g in dataset if
                        (g.match_id if hasattr(g, 'match_id') else dataset.index(g)) in val_matches_set]
        self.test_data = [g for g in dataset if
                         (g.match_id if hasattr(g, 'match_id') else dataset.index(g)) in test_matches_set]

        # Apply augmentation to training data only
        if self.hparams.use_augmentation:
            augmented = []
            for g in self.train_data:
                augmented.append(g)
                aug_g = augment_graph(g, 'horizontal')
                if aug_g is not None:
                    augmented.append(aug_g)
            self.train_data = augmented
            print(f"Augmented training set: {len(self.train_data)} graphs")

    def _setup_kfold_split(self, dataset, match_ids):
        """K-fold cross-validation split."""
        kfold = KFold(
            n_splits=self.hparams.num_folds,
            shuffle=True,
            random_state=self.hparams.seed
        )

        folds = list(kfold.split(match_ids))
        train_idx, val_idx = folds[self.hparams.fold_idx]

        train_matches = set([match_ids[i] for i in train_idx])
        val_matches = set([match_ids[i] for i in val_idx])

        self.train_data = [g for g in dataset if
                          (g.match_id if hasattr(g, 'match_id') else dataset.index(g)) in train_matches]
        self.val_data = [g for g in dataset if
                        (g.match_id if hasattr(g, 'match_id') else dataset.index(g)) in val_matches]
        self.test_data = self.val_data  # In CV, val is also test

        # Apply augmentation to training data
        if self.hparams.use_augmentation:
            augmented = []
            for g in self.train_data:
                augmented.append(g)
                aug_g = augment_graph(g, 'horizontal')
                if aug_g is not None:
                    augmented.append(aug_g)
            self.train_data = augmented

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers
        )
