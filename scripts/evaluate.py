#!/usr/bin/env python
"""Evaluation script for speaker diarization model."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from tqdm import tqdm

from multi_scale_temporal_fusion_speaker_diarization_with_uncertainty.data.loader import (
    create_dataloaders,
)
from multi_scale_temporal_fusion_speaker_diarization_with_uncertainty.models.model import (
    MultiScaleTemporalDiarizationModel,
)
from multi_scale_temporal_fusion_speaker_diarization_with_uncertainty.evaluation.metrics import (
    DiarizationMetrics,
)
from multi_scale_temporal_fusion_speaker_diarization_with_uncertainty.evaluation.analysis import (
    analyze_results,
    plot_confusion_matrix,
    visualize_diarization,
)
from multi_scale_temporal_fusion_speaker_diarization_with_uncertainty.utils.config import (
    load_config,
    set_seed,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate speaker diarization model'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to config file',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pt',
        help='Path to model checkpoint',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['val', 'test'],
        help='Dataset split to evaluate',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory to save evaluation results',
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualizations',
    )
    return parser.parse_args()


def evaluate_model(model, dataloader, device, uncertainty_threshold=0.5):
    """Evaluate model on dataset.

    Args:
        model: Model to evaluate
        dataloader: Data loader
        device: Device to use
        uncertainty_threshold: Threshold for uncertainty-based filtering

    Returns:
        Tuple of (metrics, predictions, targets)
    """
    model.eval()

    metrics_calculator = DiarizationMetrics()
    all_predictions = []
    all_targets = []
    all_features = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")

        for batch in pbar:
            # Move data to device
            features = batch['features'].to(device)
            speaker_labels = batch['speaker_labels'].to(device)
            boundary_labels = batch['boundaries'].to(device)

            # Forward pass with uncertainty
            predictions = model.predict(
                features,
                uncertainty_threshold=uncertainty_threshold,
            )

            # Convert to numpy
            speaker_pred = predictions['speaker_predictions'].cpu().numpy()
            boundary_pred = predictions['boundary_predictions'].cpu().numpy()
            speaker_target = speaker_labels.cpu().numpy()
            boundary_target = boundary_labels.cpu().numpy()

            # Update metrics
            metrics_calculator.update(
                predictions=speaker_pred,
                targets=speaker_target,
                predicted_boundaries=boundary_pred,
                target_boundaries=boundary_target,
            )

            # Store for visualization
            all_predictions.append(speaker_pred)
            all_targets.append(speaker_target)
            all_features.append(features.cpu().numpy())

    # Compute metrics
    metrics = metrics_calculator.compute()
    per_class_metrics = metrics_calculator.compute_per_class()

    return metrics, per_class_metrics, all_predictions, all_targets, all_features


def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Set random seed
    seed = config.get('seed', 42)
    set_seed(seed)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataloaders
    logger.info("Creating dataloaders...")
    data_config = config['data']
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_config.get('data_dir', './data/ami'),
        batch_size=data_config.get('batch_size', 16),
        num_workers=data_config.get('num_workers', 4),
        train_split=data_config.get('train_split', 0.7),
        val_split=data_config.get('val_split', 0.15),
        segment_length=data_config.get('segment_length', 10.0),
        sample_rate=data_config.get('sample_rate', 16000),
        seed=seed,
    )

    # Select dataloader
    if args.split == 'val':
        dataloader = val_loader
    else:
        dataloader = test_loader

    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model_config = config['model']
    model = MultiScaleTemporalDiarizationModel(
        input_dim=model_config.get('input_dim', 80),
        hidden_dim=model_config.get('hidden_dim', 256),
        num_speakers=model_config.get('num_speakers', 10),
        scales=model_config.get('scales', None),
        dropout=model_config.get('dropout', 0.1),
        mc_dropout=model_config.get('mc_dropout', 0.2),
        mc_samples=model_config.get('mc_samples', 10),
    )

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Evaluate model
    logger.info(f"Evaluating on {args.split} set...")
    eval_config = config.get('evaluation', {})
    metrics, per_class_metrics, predictions, targets, features = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        uncertainty_threshold=eval_config.get('uncertainty_threshold', 0.5),
    )

    # Print metrics
    logger.info("\nEvaluation Results:")
    logger.info("=" * 50)
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name:20s}: {value:.4f}")

    # Save metrics to JSON
    metrics_path = output_dir / f'metrics_{args.split}.json'
    with open(metrics_path, 'w') as f:
        json.dump({
            'overall': metrics,
            'per_speaker': {str(k): v for k, v in per_class_metrics.items()},
        }, f, indent=2)
    logger.info(f"\nSaved metrics to {metrics_path}")

    # Save metrics to CSV
    import pandas as pd
    metrics_df = pd.DataFrame([metrics])
    csv_path = output_dir / f'metrics_{args.split}.csv'
    metrics_df.to_csv(csv_path, index=False)
    logger.info(f"Saved metrics to {csv_path}")

    # Analyze results
    analyze_results(
        metrics=metrics,
        per_class_metrics=per_class_metrics,
        save_dir=str(output_dir),
    )

    # Generate visualizations
    if args.visualize:
        logger.info("Generating visualizations...")

        # Concatenate predictions and targets
        all_predictions = np.concatenate([p.flatten() for p in predictions])
        all_targets = np.concatenate([t.flatten() for t in targets])

        # Plot confusion matrix
        plot_confusion_matrix(
            predictions=all_predictions,
            targets=all_targets,
            save_path=str(output_dir / 'confusion_matrix.png'),
        )

        # Visualize sample diarization results
        if len(features) > 0 and len(predictions) > 0:
            sample_idx = 0
            visualize_diarization(
                features=features[sample_idx][0],  # First sample in first batch
                predictions=predictions[sample_idx][0],
                targets=targets[sample_idx][0],
                save_path=str(output_dir / 'sample_diarization.png'),
            )

        logger.info(f"Saved visualizations to {output_dir}")

    logger.info("\nEvaluation completed successfully!")


if __name__ == '__main__':
    main()
