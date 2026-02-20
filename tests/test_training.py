"""Tests for training utilities."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from multi_scale_temporal_fusion_speaker_diarization_with_uncertainty.training.trainer import (
    Trainer,
    EarlyStopping,
)
from multi_scale_temporal_fusion_speaker_diarization_with_uncertainty.models.model import (
    MultiScaleTemporalDiarizationModel,
)
from multi_scale_temporal_fusion_speaker_diarization_with_uncertainty.models.components import (
    DiarizationLoss,
)


class TestEarlyStopping:
    """Test EarlyStopping class."""

    def test_early_stopping_min_mode(self):
        """Test early stopping in min mode."""
        early_stopping = EarlyStopping(patience=3, mode='min')

        # Improving scores
        assert not early_stopping(1.0)
        assert not early_stopping(0.9)
        assert not early_stopping(0.8)

        # Non-improving scores
        assert not early_stopping(0.85)
        assert not early_stopping(0.85)
        assert early_stopping(0.85)  # Should trigger after patience

    def test_early_stopping_max_mode(self):
        """Test early stopping in max mode."""
        early_stopping = EarlyStopping(patience=2, mode='max')

        assert not early_stopping(0.7)
        assert not early_stopping(0.8)
        assert not early_stopping(0.75)  # Non-improving
        assert early_stopping(0.75)  # Should trigger


class TestTrainer:
    """Test Trainer class."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return MultiScaleTemporalDiarizationModel(
            input_dim=80,
            hidden_dim=32,
            num_speakers=4,
        )

    @pytest.fixture
    def simple_dataloader(self):
        """Create a simple dataloader for testing."""
        batch_size = 2
        num_samples = 10
        feature_dim = 80
        time_steps = 50

        features = torch.randn(num_samples, feature_dim, time_steps)
        speaker_labels = torch.randint(0, 4, (num_samples, time_steps))
        boundaries = torch.rand(num_samples, time_steps)

        dataset = TensorDataset(features, speaker_labels, boundaries)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Convert to dict format
        class DictDataLoader:
            def __init__(self, dataloader):
                self.dataloader = dataloader

            def __iter__(self):
                for features, speaker_labels, boundaries in self.dataloader:
                    yield {
                        'features': features,
                        'speaker_labels': speaker_labels,
                        'boundaries': boundaries,
                    }

            def __len__(self):
                return len(self.dataloader)

        return DictDataLoader(dataloader)

    def test_trainer_initialization(self, simple_model):
        """Test trainer initialization."""
        criterion = DiarizationLoss()
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)

        trainer = Trainer(
            model=simple_model,
            criterion=criterion,
            optimizer=optimizer,
            device=torch.device('cpu'),
        )

        assert trainer.device == torch.device('cpu')
        assert trainer.model is not None

    def test_train_epoch(self, simple_model, simple_dataloader):
        """Test single training epoch."""
        criterion = DiarizationLoss()
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)

        trainer = Trainer(
            model=simple_model,
            criterion=criterion,
            optimizer=optimizer,
            device=torch.device('cpu'),
            use_amp=False,
        )

        metrics = trainer.train_epoch(simple_dataloader)

        assert 'loss' in metrics
        assert metrics['loss'] > 0
