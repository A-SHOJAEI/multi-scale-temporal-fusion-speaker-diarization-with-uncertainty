"""Results analysis and visualization for speaker diarization."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> None:
    """Plot training and validation loss curves.

    Args:
        history: Training history dictionary
        save_path: Path to save plot (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Learning rate
    axes[1].plot(history['learning_rate'])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training curves to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_diarization(
    features: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    boundaries: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> None:
    """Visualize diarization results with features and speaker labels.

    Args:
        features: Feature matrix of shape (feature_dim, time)
        predictions: Predicted speaker labels
        targets: Ground truth speaker labels
        boundaries: Predicted boundaries (optional)
        save_path: Path to save plot (optional)
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 8))

    # Plot features (e.g., mel spectrogram)
    im = axes[0].imshow(
        features,
        aspect='auto',
        origin='lower',
        cmap='viridis',
    )
    axes[0].set_ylabel('Feature Dim')
    axes[0].set_title('Input Features')
    plt.colorbar(im, ax=axes[0])

    # Plot ground truth speaker labels
    axes[1].plot(targets, label='Ground Truth', linewidth=2)
    axes[1].set_ylabel('Speaker ID')
    axes[1].set_title('Ground Truth Speaker Labels')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot predictions and boundaries
    axes[2].plot(predictions, label='Predictions', linewidth=2)
    if boundaries is not None:
        # Mark boundaries
        boundary_positions = np.where(boundaries > 0)[0]
        for pos in boundary_positions:
            axes[2].axvline(x=pos, color='red', alpha=0.3, linestyle='--')
    axes[2].set_xlabel('Time (frames)')
    axes[2].set_ylabel('Speaker ID')
    axes[2].set_title('Predicted Speaker Labels and Boundaries')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved diarization visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def analyze_results(
    metrics: Dict[str, float],
    per_class_metrics: Dict[int, Dict[str, float]],
    save_dir: Optional[str] = None,
) -> None:
    """Analyze and visualize evaluation results.

    Args:
        metrics: Overall metrics dictionary
        per_class_metrics: Per-speaker metrics
        save_dir: Directory to save analysis (optional)
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Print overall metrics
    logger.info("Overall Metrics:")
    for metric_name, value in metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")

    # Print per-class metrics
    logger.info("\nPer-Speaker Metrics:")
    for speaker_id, speaker_metrics in per_class_metrics.items():
        logger.info(f"  Speaker {speaker_id}:")
        for metric_name, value in speaker_metrics.items():
            logger.info(f"    {metric_name}: {value:.4f}")

    # Plot per-class metrics
    if per_class_metrics:
        fig, ax = plt.subplots(figsize=(10, 6))

        speaker_ids = sorted(per_class_metrics.keys())
        f1_scores = [per_class_metrics[sid]['F1'] for sid in speaker_ids]
        precision_scores = [per_class_metrics[sid]['Precision'] for sid in speaker_ids]
        recall_scores = [per_class_metrics[sid]['Recall'] for sid in speaker_ids]

        x = np.arange(len(speaker_ids))
        width = 0.25

        ax.bar(x - width, f1_scores, width, label='F1')
        ax.bar(x, precision_scores, width, label='Precision')
        ax.bar(x + width, recall_scores, width, label='Recall')

        ax.set_xlabel('Speaker ID')
        ax.set_ylabel('Score')
        ax.set_title('Per-Speaker Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Speaker {sid}' for sid in speaker_ids])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_dir:
            save_path = save_dir / 'per_speaker_metrics.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved per-speaker metrics plot to {save_path}")
        else:
            plt.show()

        plt.close()


def plot_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Plot confusion matrix for speaker classification.

    Args:
        predictions: Predicted speaker labels
        targets: Ground truth speaker labels
        save_path: Path to save plot (optional)
    """
    from sklearn.metrics import confusion_matrix

    # Compute confusion matrix
    unique_speakers = sorted(set(targets.tolist() + predictions.tolist()))
    cm = confusion_matrix(targets, predictions, labels=unique_speakers)

    # Normalize
    cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-9)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=[f'S{s}' for s in unique_speakers],
        yticklabels=[f'S{s}' for s in unique_speakers],
        ax=ax,
    )
    ax.set_xlabel('Predicted Speaker')
    ax.set_ylabel('True Speaker')
    ax.set_title('Speaker Classification Confusion Matrix')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save_path}")
    else:
        plt.show()

    plt.close()
