"""Evaluation metrics for speaker diarization."""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


def compute_der(
    predictions: np.ndarray,
    targets: np.ndarray,
    tolerance: float = 0.25,
) -> float:
    """Compute Diarization Error Rate (DER).

    DER = (False Alarm + Missed Detection + Speaker Error) / Total

    Args:
        predictions: Predicted speaker labels
        targets: Ground truth speaker labels
        tolerance: Tolerance for boundary errors (in seconds)

    Returns:
        DER score
    """
    # Ensure same length
    min_len = min(len(predictions), len(targets))
    predictions = predictions[:min_len]
    targets = targets[:min_len]

    # Compute errors
    errors = (predictions != targets).sum()
    total = len(targets)

    der = errors / total if total > 0 else 0.0

    return der


def compute_jer(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Compute Jaccard Error Rate (JER).

    JER = 1 - Jaccard Index

    Args:
        predictions: Predicted speaker labels
        targets: Ground truth speaker labels

    Returns:
        JER score
    """
    # Ensure same length
    min_len = min(len(predictions), len(targets))
    predictions = predictions[:min_len]
    targets = targets[:min_len]

    # Compute Jaccard index
    intersection = (predictions == targets).sum()
    union = len(predictions)

    jaccard = intersection / union if union > 0 else 0.0
    jer = 1.0 - jaccard

    return jer


def compute_speaker_f1(
    predictions: np.ndarray,
    targets: np.ndarray,
    average: str = 'weighted',
) -> float:
    """Compute F1 score for speaker classification.

    Args:
        predictions: Predicted speaker labels
        targets: Ground truth speaker labels
        average: Averaging method ('micro', 'macro', 'weighted')

    Returns:
        F1 score
    """
    # Ensure same length
    min_len = min(len(predictions), len(targets))
    predictions = predictions[:min_len]
    targets = targets[:min_len]

    try:
        f1 = f1_score(targets, predictions, average=average, zero_division=0)
    except Exception as e:
        logger.warning(f"Error computing F1 score: {e}")
        f1 = 0.0

    return f1


def compute_boundary_precision(
    predicted_boundaries: np.ndarray,
    target_boundaries: np.ndarray,
    tolerance: int = 5,
) -> float:
    """Compute precision for boundary detection.

    Args:
        predicted_boundaries: Binary array of predicted boundaries
        target_boundaries: Binary array of ground truth boundaries
        tolerance: Tolerance window (in frames) for matching boundaries

    Returns:
        Boundary precision score
    """
    # Find boundary positions
    pred_positions = np.where(predicted_boundaries > 0)[0]
    target_positions = np.where(target_boundaries > 0)[0]

    if len(pred_positions) == 0:
        return 0.0

    # Count true positives
    true_positives = 0
    for pred_pos in pred_positions:
        # Check if there's a target boundary within tolerance
        distances = np.abs(target_positions - pred_pos)
        if len(distances) > 0 and np.min(distances) <= tolerance:
            true_positives += 1

    precision = true_positives / len(pred_positions)

    return precision


def compute_boundary_recall(
    predicted_boundaries: np.ndarray,
    target_boundaries: np.ndarray,
    tolerance: int = 5,
) -> float:
    """Compute recall for boundary detection.

    Args:
        predicted_boundaries: Binary array of predicted boundaries
        target_boundaries: Binary array of ground truth boundaries
        tolerance: Tolerance window (in frames) for matching boundaries

    Returns:
        Boundary recall score
    """
    # Find boundary positions
    pred_positions = np.where(predicted_boundaries > 0)[0]
    target_positions = np.where(target_boundaries > 0)[0]

    if len(target_positions) == 0:
        return 1.0 if len(pred_positions) == 0 else 0.0

    # Count true positives
    true_positives = 0
    for target_pos in target_positions:
        # Check if there's a predicted boundary within tolerance
        distances = np.abs(pred_positions - target_pos)
        if len(distances) > 0 and np.min(distances) <= tolerance:
            true_positives += 1

    recall = true_positives / len(target_positions)

    return recall


class DiarizationMetrics:
    """Comprehensive metrics for speaker diarization evaluation.

    Args:
        tolerance: Tolerance for boundary matching (in frames)
    """

    def __init__(self, tolerance: int = 5):
        self.tolerance = tolerance
        self.reset()

    def reset(self) -> None:
        """Reset all accumulated metrics."""
        self.all_predictions = []
        self.all_targets = []
        self.all_predicted_boundaries = []
        self.all_target_boundaries = []

    def update(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        predicted_boundaries: np.ndarray,
        target_boundaries: np.ndarray,
    ) -> None:
        """Update metrics with new batch.

        Args:
            predictions: Predicted speaker labels
            targets: Ground truth speaker labels
            predicted_boundaries: Predicted boundaries
            target_boundaries: Ground truth boundaries
        """
        self.all_predictions.append(predictions.flatten())
        self.all_targets.append(targets.flatten())
        self.all_predicted_boundaries.append(predicted_boundaries.flatten())
        self.all_target_boundaries.append(target_boundaries.flatten())

    def compute(self) -> Dict[str, float]:
        """Compute all metrics.

        Returns:
            Dictionary of computed metrics
        """
        # Concatenate all batches
        predictions = np.concatenate(self.all_predictions)
        targets = np.concatenate(self.all_targets)
        predicted_boundaries = np.concatenate(self.all_predicted_boundaries)
        target_boundaries = np.concatenate(self.all_target_boundaries)

        # Compute metrics
        metrics = {
            'DER': compute_der(predictions, targets),
            'JER': compute_jer(predictions, targets),
            'Speaker_F1': compute_speaker_f1(predictions, targets, average='weighted'),
            'Speaker_Precision': precision_score(targets, predictions, average='weighted', zero_division=0),
            'Speaker_Recall': recall_score(targets, predictions, average='weighted', zero_division=0),
            'Boundary_Precision': compute_boundary_precision(
                predicted_boundaries, target_boundaries, tolerance=self.tolerance
            ),
            'Boundary_Recall': compute_boundary_recall(
                predicted_boundaries, target_boundaries, tolerance=self.tolerance
            ),
        }

        # Compute boundary F1
        if metrics['Boundary_Precision'] + metrics['Boundary_Recall'] > 0:
            metrics['Boundary_F1'] = (
                2 * metrics['Boundary_Precision'] * metrics['Boundary_Recall'] /
                (metrics['Boundary_Precision'] + metrics['Boundary_Recall'])
            )
        else:
            metrics['Boundary_F1'] = 0.0

        return metrics

    def compute_per_class(self) -> Dict[int, Dict[str, float]]:
        """Compute per-speaker metrics.

        Returns:
            Dictionary mapping speaker ID to metrics
        """
        predictions = np.concatenate(self.all_predictions)
        targets = np.concatenate(self.all_targets)

        unique_speakers = np.unique(targets)
        per_class_metrics = {}

        for speaker_id in unique_speakers:
            # Create binary classification for this speaker
            binary_pred = (predictions == speaker_id).astype(int)
            binary_target = (targets == speaker_id).astype(int)

            per_class_metrics[int(speaker_id)] = {
                'F1': f1_score(binary_target, binary_pred, zero_division=0),
                'Precision': precision_score(binary_target, binary_pred, zero_division=0),
                'Recall': recall_score(binary_target, binary_pred, zero_division=0),
            }

        return per_class_metrics
