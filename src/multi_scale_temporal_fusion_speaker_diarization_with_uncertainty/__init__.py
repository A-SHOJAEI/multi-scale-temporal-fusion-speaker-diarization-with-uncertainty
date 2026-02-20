"""Multi-Scale Temporal Fusion Speaker Diarization with Uncertainty Quantification.

A novel speaker diarization system combining multi-scale temporal feature fusion
with uncertainty-aware clustering for robust speaker change detection.
"""

__version__ = "0.1.0"
__author__ = "Alireza Shojaei"

from multi_scale_temporal_fusion_speaker_diarization_with_uncertainty.models.model import (
    MultiScaleTemporalDiarizationModel,
)
from multi_scale_temporal_fusion_speaker_diarization_with_uncertainty.models.components import (
    MultiScaleTemporalFusion,
    UncertaintyEstimator,
)

__all__ = [
    "MultiScaleTemporalDiarizationModel",
    "MultiScaleTemporalFusion",
    "UncertaintyEstimator",
]
