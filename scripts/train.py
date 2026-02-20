#!/usr/bin/env python
"""Training script for multi-scale temporal fusion speaker diarization model."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau

from multi_scale_temporal_fusion_speaker_diarization_with_uncertainty.data.loader import (
    create_dataloaders,
)
from multi_scale_temporal_fusion_speaker_diarization_with_uncertainty.models.model import (
    MultiScaleTemporalDiarizationModel,
)
from multi_scale_temporal_fusion_speaker_diarization_with_uncertainty.models.components import (
    DiarizationLoss,
)
from multi_scale_temporal_fusion_speaker_diarization_with_uncertainty.training.trainer import (
    Trainer,
    EarlyStopping,
)
from multi_scale_temporal_fusion_speaker_diarization_with_uncertainty.utils.config import (
    load_config,
    save_config,
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
        description='Train multi-scale temporal fusion speaker diarization model'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to config file',
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda or cpu)',
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from',
    )
    return parser.parse_args()


def create_optimizer(model, config):
    """Create optimizer from config.

    Args:
        model: Model to optimize
        config: Training configuration

    Returns:
        Optimizer instance
    """
    optimizer_name = config.get('optimizer', 'adam').lower()
    lr = config.get('learning_rate', 0.001)
    weight_decay = config.get('weight_decay', 0.0001)

    if optimizer_name == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return optimizer


def create_scheduler(optimizer, config):
    """Create learning rate scheduler from config.

    Args:
        optimizer: Optimizer instance
        config: Training configuration

    Returns:
        Scheduler instance or None
    """
    scheduler_name = config.get('scheduler', None)

    if scheduler_name is None:
        return None

    scheduler_name = scheduler_name.lower()
    scheduler_params = config.get('scheduler_params', {})

    if scheduler_name == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=scheduler_params.get('T_max', 100),
            eta_min=scheduler_params.get('eta_min', 0.00001),
        )
    elif scheduler_name == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=scheduler_params.get('step_size', 30),
            gamma=scheduler_params.get('gamma', 0.1),
        )
    elif scheduler_name == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_params.get('factor', 0.5),
            patience=scheduler_params.get('patience', 10),
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    return scheduler


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Set random seed
    seed = config.get('seed', 42)
    set_seed(seed)

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create output directories
    checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
    results_dir = Path(config.get('results_dir', 'results'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    save_config(config, results_dir / 'config.yaml')

    # Initialize MLflow (optional, wrapped in try-except)
    try:
        import mlflow
        mlflow_config = config.get('mlflow', {})
        if mlflow_config.get('tracking_uri'):
            mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
        mlflow.set_experiment(mlflow_config.get('experiment_name', 'diarization'))
        mlflow.start_run()
        mlflow.log_params(config)
        use_mlflow = True
        logger.info("MLflow tracking enabled")
    except Exception as e:
        logger.warning(f"MLflow not available or failed to initialize: {e}")
        use_mlflow = False

    try:
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

        logger.info(f"Train batches: {len(train_loader)}, "
                   f"Val batches: {len(val_loader)}, "
                   f"Test batches: {len(test_loader)}")

        # Create model
        logger.info("Creating model...")
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

        logger.info(f"Model parameters: {model.get_num_parameters():,}")

        # Create loss function
        loss_config = config.get('loss', {})
        criterion = DiarizationLoss(
            speaker_weight=loss_config.get('speaker_weight', 1.0),
            boundary_weight=loss_config.get('boundary_weight', 2.0),
            use_focal_loss=loss_config.get('use_focal_loss', True),
            focal_alpha=loss_config.get('focal_alpha', 0.25),
            focal_gamma=loss_config.get('focal_gamma', 2.0),
        )

        # Create optimizer
        training_config = config['training']
        optimizer = create_optimizer(model, training_config)

        # Create scheduler
        scheduler = create_scheduler(optimizer, training_config)

        # Create trainer
        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            checkpoint_dir=str(checkpoint_dir),
            use_amp=training_config.get('use_amp', True),
            grad_clip=training_config.get('grad_clip', 1.0),
        )

        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)

        # Create early stopping
        early_stopping_config = training_config.get('early_stopping', {})
        early_stopping = EarlyStopping(
            patience=early_stopping_config.get('patience', 15),
            min_delta=early_stopping_config.get('min_delta', 0.0001),
            mode='min',
        ) if early_stopping_config else None

        # Train model
        logger.info("Starting training...")
        num_epochs = training_config.get('num_epochs', 100)
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            early_stopping=early_stopping,
        )

        # Save training history
        history_path = results_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"Saved training history to {history_path}")

        # Log final metrics to MLflow
        if use_mlflow:
            try:
                mlflow.log_metric('final_train_loss', history['train_loss'][-1])
                mlflow.log_metric('final_val_loss', history['val_loss'][-1])
                mlflow.log_metric('best_val_loss', min(history['val_loss']))
            except Exception as e:
                logger.warning(f"Failed to log metrics to MLflow: {e}")

        logger.info("Training completed successfully!")
        logger.info(f"Best validation loss: {min(history['val_loss']):.4f}")

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise

    finally:
        # End MLflow run
        if use_mlflow:
            try:
                mlflow.end_run()
            except:
                pass


if __name__ == '__main__':
    main()
