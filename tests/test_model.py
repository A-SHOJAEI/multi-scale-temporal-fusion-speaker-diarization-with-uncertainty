"""Tests for model components."""

import pytest
import torch

from multi_scale_temporal_fusion_speaker_diarization_with_uncertainty.models.model import (
    MultiScaleTemporalDiarizationModel,
)
from multi_scale_temporal_fusion_speaker_diarization_with_uncertainty.models.components import (
    MultiScaleTemporalFusion,
    UncertaintyEstimator,
    TemporalConvBlock,
    AttentionFusion,
    DiarizationLoss,
)


class TestMultiScaleTemporalFusion:
    """Test MultiScaleTemporalFusion module."""

    def test_initialization(self):
        """Test module initialization."""
        module = MultiScaleTemporalFusion(
            input_dim=80,
            hidden_dim=128,
        )

        assert module.num_scales == 3

    def test_forward(self):
        """Test forward pass."""
        module = MultiScaleTemporalFusion(
            input_dim=80,
            hidden_dim=128,
        )

        x = torch.randn(4, 80, 100)
        output = module(x)

        assert output.shape == (4, 128, 100)


class TestUncertaintyEstimator:
    """Test UncertaintyEstimator module."""

    def test_forward_without_uncertainty(self):
        """Test forward pass without uncertainty computation."""
        module = UncertaintyEstimator(
            input_dim=128,
            dropout=0.2,
        )

        x = torch.randn(4, 128, 100)
        output, uncertainty = module(x, compute_uncertainty=False)

        assert output.shape[0] == 4
        assert output.shape[2] == 100
        assert uncertainty is None

    def test_forward_with_uncertainty(self):
        """Test forward pass with uncertainty computation."""
        module = UncertaintyEstimator(
            input_dim=128,
            dropout=0.2,
            num_samples=5,
        )

        x = torch.randn(4, 128, 100)
        output, uncertainty = module(x, compute_uncertainty=True)

        assert output.shape[0] == 4
        assert output.shape[2] == 100
        assert uncertainty is not None
        assert uncertainty.shape == (4, 100)


class TestDiarizationModel:
    """Test MultiScaleTemporalDiarizationModel."""

    def test_initialization(self, model_config):
        """Test model initialization."""
        model = MultiScaleTemporalDiarizationModel(**model_config)

        assert model.input_dim == 80
        assert model.hidden_dim == 64
        assert model.num_speakers == 4

    def test_forward(self, model_config, sample_audio_features):
        """Test forward pass."""
        model = MultiScaleTemporalDiarizationModel(**model_config)
        model.eval()

        with torch.no_grad():
            outputs = model(sample_audio_features)

        assert 'speaker_logits' in outputs
        assert 'boundary_logits' in outputs
        assert outputs['speaker_logits'].shape[0] == 4  # batch size
        assert outputs['speaker_logits'].shape[1] == 4  # num speakers

    def test_predict(self, model_config, sample_audio_features):
        """Test prediction method."""
        model = MultiScaleTemporalDiarizationModel(**model_config)
        model.eval()

        predictions = model.predict(sample_audio_features)

        assert 'speaker_predictions' in predictions
        assert 'boundary_predictions' in predictions
        assert predictions['speaker_predictions'].shape[0] == 4


class TestDiarizationLoss:
    """Test DiarizationLoss module."""

    def test_loss_computation(self, sample_labels):
        """Test loss computation."""
        criterion = DiarizationLoss(
            speaker_weight=1.0,
            boundary_weight=2.0,
        )

        batch_size = 4
        num_speakers = 4
        time_steps = 100

        speaker_logits = torch.randn(batch_size, num_speakers, time_steps)
        boundary_logits = torch.randn(batch_size, time_steps)

        loss_dict = criterion(
            speaker_logits=speaker_logits,
            boundary_logits=boundary_logits,
            speaker_labels=sample_labels['speaker_labels'],
            boundary_labels=sample_labels['boundaries'],
        )

        assert 'loss' in loss_dict
        assert 'speaker_loss' in loss_dict
        assert 'boundary_loss' in loss_dict
        assert loss_dict['loss'].item() > 0
