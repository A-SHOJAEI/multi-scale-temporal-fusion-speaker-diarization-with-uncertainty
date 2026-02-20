"""Custom components for multi-scale temporal fusion and uncertainty estimation."""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TemporalConvBlock(nn.Module):
    """Temporal convolution block with residual connection.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        dilation: Dilation rate for temporal convolution
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)

        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, time)

        Returns:
            Output tensor of shape (batch, channels, time)
        """
        residual = self.residual(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual
        out = F.relu(out)

        return out


class AttentionFusion(nn.Module):
    """Attention-based fusion of multi-scale features.

    Args:
        feature_dim: Dimension of input features
        num_scales: Number of temporal scales
        hidden_dim: Hidden dimension for attention computation
    """

    def __init__(
        self,
        feature_dim: int,
        num_scales: int,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_scales = num_scales

        # Attention network
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_scales),
            nn.Softmax(dim=-1),
        )

    def forward(self, scale_features: List[torch.Tensor]) -> torch.Tensor:
        """Fuse multi-scale features using learned attention weights.

        Args:
            scale_features: List of feature tensors from different scales,
                          each of shape (batch, feature_dim, time)

        Returns:
            Fused features of shape (batch, feature_dim, time)
        """
        # Stack features along scale dimension
        stacked = torch.stack(scale_features, dim=1)  # (batch, num_scales, feature_dim, time)

        batch_size, num_scales, feature_dim, time_steps = stacked.shape

        # Compute attention weights for each time step
        # Average over feature dimension to get time-specific context
        context = torch.mean(stacked, dim=1)  # (batch, feature_dim, time)
        context = context.transpose(1, 2)  # (batch, time, feature_dim)

        # Compute attention weights
        attention_weights = self.attention(context)  # (batch, time, num_scales)
        attention_weights = attention_weights.transpose(1, 2).unsqueeze(2)  # (batch, num_scales, 1, time)

        # Apply attention weights
        fused = (stacked * attention_weights).sum(dim=1)  # (batch, feature_dim, time)

        return fused


class MultiScaleTemporalFusion(nn.Module):
    """Multi-scale temporal feature fusion module.

    This is the core novel component that extracts features at different temporal
    scales (frame, phoneme, word-level) and fuses them using learned attention.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension for convolutions
        scales: List of temporal scales (kernel sizes and dilations)
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        scales: Optional[List[Dict]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if scales is None:
            # Default scales: frame (5ms), phoneme (50ms), word (500ms)
            # Assuming hop_length=160 samples at 16kHz = 10ms per frame
            scales = [
                {'kernel_size': 3, 'dilation': 1},   # ~30ms (frame-level)
                {'kernel_size': 5, 'dilation': 2},   # ~100ms (phoneme-level)
                {'kernel_size': 7, 'dilation': 4},   # ~280ms (word-level)
            ]

        self.scales = scales
        self.num_scales = len(scales)

        # Project input to hidden dimension
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)

        # Multi-scale temporal convolution blocks
        self.scale_blocks = nn.ModuleList([
            TemporalConvBlock(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=scale['kernel_size'],
                dilation=scale['dilation'],
                dropout=dropout,
            )
            for scale in scales
        ])

        # Attention-based fusion
        self.fusion = AttentionFusion(
            feature_dim=hidden_dim,
            num_scales=self.num_scales,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features of shape (batch, input_dim, time)

        Returns:
            Fused multi-scale features of shape (batch, hidden_dim, time)
        """
        # Project input
        x = self.input_proj(x)

        # Extract features at different scales
        scale_features = []
        for scale_block in self.scale_blocks:
            scale_feat = scale_block(x)
            scale_features.append(scale_feat)

        # Fuse using attention
        fused = self.fusion(scale_features)

        return fused


class UncertaintyEstimator(nn.Module):
    """Monte Carlo Dropout-based uncertainty estimation module.

    This component estimates prediction uncertainty using MC Dropout,
    enabling adaptive rejection of low-confidence speaker boundaries.

    Args:
        input_dim: Input feature dimension
        dropout: Dropout probability for MC sampling
        num_samples: Number of MC samples for uncertainty estimation
    """

    def __init__(
        self,
        input_dim: int,
        dropout: float = 0.2,
        num_samples: int = 10,
    ):
        super().__init__()

        self.dropout = dropout
        self.num_samples = num_samples

        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, compute_uncertainty: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with optional uncertainty estimation.

        Args:
            x: Input features of shape (batch, input_dim, time)
            compute_uncertainty: Whether to compute uncertainty via MC Dropout

        Returns:
            Tuple of (features, uncertainty), where uncertainty is None if not computed
        """
        if not compute_uncertainty:
            # Standard forward pass
            x_transposed = x.transpose(1, 2)  # (batch, time, input_dim)
            out = self.layers(x_transposed)
            out = out.transpose(1, 2)  # (batch, input_dim // 4, time)
            return out, None

        # MC Dropout for uncertainty estimation
        self.train()  # Enable dropout

        x_transposed = x.transpose(1, 2)  # (batch, time, input_dim)

        mc_samples = []
        for _ in range(self.num_samples):
            sample = self.layers(x_transposed)
            mc_samples.append(sample)

        # Stack samples
        mc_samples = torch.stack(mc_samples, dim=0)  # (num_samples, batch, time, dim)

        # Compute mean and variance
        mean = torch.mean(mc_samples, dim=0)  # (batch, time, dim)
        variance = torch.var(mc_samples, dim=0)  # (batch, time, dim)

        # Aggregate uncertainty across feature dimension
        uncertainty = torch.mean(variance, dim=-1)  # (batch, time)

        mean = mean.transpose(1, 2)  # (batch, dim, time)

        return mean, uncertainty


class DiarizationLoss(nn.Module):
    """Custom loss function for speaker diarization with boundary detection.

    Combines speaker classification loss with boundary detection loss,
    with optional uncertainty-weighted terms.

    Args:
        speaker_weight: Weight for speaker classification loss
        boundary_weight: Weight for boundary detection loss
        use_focal_loss: Whether to use focal loss for boundaries
        focal_alpha: Alpha parameter for focal loss
        focal_gamma: Gamma parameter for focal loss
    """

    def __init__(
        self,
        speaker_weight: float = 1.0,
        boundary_weight: float = 2.0,
        use_focal_loss: bool = True,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()

        self.speaker_weight = speaker_weight
        self.boundary_weight = boundary_weight
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def focal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss for handling class imbalance.

        Args:
            pred: Predicted logits (not probabilities)
            target: Target labels

        Returns:
            Focal loss value
        """
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        pt = torch.exp(-bce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * bce_loss

        return focal_loss.mean()

    def forward(
        self,
        speaker_logits: torch.Tensor,
        boundary_logits: torch.Tensor,
        speaker_labels: torch.Tensor,
        boundary_labels: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined diarization loss.

        Args:
            speaker_logits: Speaker predictions of shape (batch, num_speakers, time)
            boundary_logits: Boundary predictions of shape (batch, time)
            speaker_labels: Speaker ground truth of shape (batch, time)
            boundary_labels: Boundary ground truth of shape (batch, time)
            uncertainty: Optional uncertainty estimates of shape (batch, time)

        Returns:
            Dictionary containing total loss and individual components
        """
        # Speaker classification loss
        speaker_logits_flat = speaker_logits.transpose(1, 2).reshape(-1, speaker_logits.shape[1])
        speaker_labels_flat = speaker_labels.reshape(-1)
        speaker_loss = F.cross_entropy(speaker_logits_flat, speaker_labels_flat)

        # Boundary detection loss
        if self.use_focal_loss:
            boundary_loss = self.focal_loss(boundary_logits, boundary_labels)
        else:
            boundary_loss = F.binary_cross_entropy_with_logits(boundary_logits, boundary_labels)

        # Uncertainty-weighted loss (optional)
        if uncertainty is not None:
            # Higher uncertainty reduces contribution to loss
            uncertainty_weight = 1.0 / (1.0 + uncertainty)
            boundary_loss = (boundary_loss * uncertainty_weight.mean()).mean()

        # Combine losses
        total_loss = (
            self.speaker_weight * speaker_loss +
            self.boundary_weight * boundary_loss
        )

        return {
            'loss': total_loss,
            'speaker_loss': speaker_loss,
            'boundary_loss': boundary_loss,
        }
