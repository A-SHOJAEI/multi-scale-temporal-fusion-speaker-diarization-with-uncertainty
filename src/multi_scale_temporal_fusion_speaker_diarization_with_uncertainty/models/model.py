"""Main speaker diarization model with multi-scale fusion and uncertainty estimation."""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from multi_scale_temporal_fusion_speaker_diarization_with_uncertainty.models.components import (
    MultiScaleTemporalFusion,
    UncertaintyEstimator,
)

logger = logging.getLogger(__name__)


class MultiScaleTemporalDiarizationModel(nn.Module):
    """Multi-scale temporal fusion model for speaker diarization.

    This model combines:
    1. Multi-scale temporal feature extraction (frame, phoneme, word-level)
    2. Uncertainty estimation using Monte Carlo Dropout
    3. Joint speaker classification and boundary detection

    Args:
        input_dim: Input feature dimension (e.g., 80 for mel spectrogram)
        hidden_dim: Hidden dimension for temporal convolutions
        num_speakers: Maximum number of speakers to classify
        scales: List of temporal scales for multi-scale fusion
        dropout: Dropout probability
        mc_dropout: Dropout probability for uncertainty estimation
        mc_samples: Number of MC samples for uncertainty
    """

    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 256,
        num_speakers: int = 10,
        scales: Optional[list] = None,
        dropout: float = 0.1,
        mc_dropout: float = 0.2,
        mc_samples: int = 10,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_speakers = num_speakers

        # Multi-scale temporal fusion
        self.multi_scale_fusion = MultiScaleTemporalFusion(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            scales=scales,
            dropout=dropout,
        )

        # Uncertainty estimator
        self.uncertainty_estimator = UncertaintyEstimator(
            input_dim=hidden_dim,
            dropout=mc_dropout,
            num_samples=mc_samples,
        )

        # Feature dimension after uncertainty estimator
        self.feature_dim = hidden_dim // 4

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        # Output heads
        self.speaker_head = nn.Linear(hidden_dim * 2, num_speakers)
        self.boundary_head = nn.Linear(hidden_dim * 2, 1)

        logger.info(f"Initialized MultiScaleTemporalDiarizationModel with "
                   f"input_dim={input_dim}, hidden_dim={hidden_dim}, "
                   f"num_speakers={num_speakers}")

    def forward(
        self,
        features: torch.Tensor,
        compute_uncertainty: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            features: Input features of shape (batch, input_dim, time)
            compute_uncertainty: Whether to compute uncertainty estimates

        Returns:
            Dictionary containing:
                - speaker_logits: Speaker predictions (batch, num_speakers, time)
                - boundary_logits: Boundary predictions (batch, time)
                - uncertainty: Uncertainty estimates (batch, time) if computed
        """
        # Multi-scale temporal fusion
        fused_features = self.multi_scale_fusion(features)

        # Uncertainty estimation
        uncertainty_features, uncertainty = self.uncertainty_estimator(
            fused_features,
            compute_uncertainty=compute_uncertainty,
        )

        # Transpose for LSTM (batch, time, features)
        uncertainty_features = uncertainty_features.transpose(1, 2)

        # LSTM temporal modeling
        lstm_out, _ = self.lstm(uncertainty_features)

        # Speaker classification head
        speaker_logits = self.speaker_head(lstm_out)
        speaker_logits = speaker_logits.transpose(1, 2)  # (batch, num_speakers, time)

        # Boundary detection head
        boundary_logits = self.boundary_head(lstm_out)
        boundary_logits = boundary_logits.squeeze(-1)  # (batch, time)

        output = {
            'speaker_logits': speaker_logits,
            'boundary_logits': boundary_logits,
        }

        if compute_uncertainty:
            output['uncertainty'] = uncertainty

        return output

    def predict(
        self,
        features: torch.Tensor,
        uncertainty_threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """Generate predictions with uncertainty-based filtering.

        Args:
            features: Input features of shape (batch, input_dim, time)
            uncertainty_threshold: Threshold for filtering low-confidence predictions

        Returns:
            Dictionary containing filtered predictions
        """
        self.eval()

        with torch.no_grad():
            # Forward pass with uncertainty
            outputs = self.forward(features, compute_uncertainty=True)

            # Get predictions
            speaker_probs = F.softmax(outputs['speaker_logits'], dim=1)
            speaker_pred = torch.argmax(speaker_probs, dim=1)

            boundary_probs = torch.sigmoid(outputs['boundary_logits'])
            boundary_pred = (boundary_probs > 0.5).float()

            # Filter based on uncertainty
            if 'uncertainty' in outputs:
                uncertainty = outputs['uncertainty']

                # Mask high-uncertainty boundaries
                high_confidence = uncertainty < uncertainty_threshold
                boundary_pred = boundary_pred * high_confidence.float()

            return {
                'speaker_predictions': speaker_pred,
                'speaker_probabilities': speaker_probs,
                'boundary_predictions': boundary_pred,
                'boundary_probabilities': boundary_probs,
                'uncertainty': outputs.get('uncertainty'),
            }

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters.

        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
