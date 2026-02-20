"""Training loop with learning rate scheduling, early stopping, and checkpointing."""

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve.

    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change in monitored metric to qualify as improvement
        mode: 'min' for loss, 'max' for accuracy/F1
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """Check if training should stop.

        Args:
            score: Current validation score

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class Trainer:
    """Trainer class for speaker diarization model.

    Args:
        model: The model to train
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        use_amp: Whether to use automatic mixed precision
        grad_clip: Gradient clipping value (optional)
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        device: torch.device = None,
        checkpoint_dir: str = "checkpoints",
        use_amp: bool = True,
        grad_clip: Optional[float] = 1.0,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.use_amp = use_amp and torch.cuda.is_available()
        self.grad_clip = grad_clip

        # Move model to device
        self.model.to(self.device)

        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
        }

        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Using AMP: {self.use_amp}")
        logger.info(f"Model parameters: {self.model.get_num_parameters():,}")

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        total_loss = 0.0
        total_speaker_loss = 0.0
        total_boundary_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc="Training")

        for batch in pbar:
            # Move data to device
            features = batch['features'].to(self.device)
            speaker_labels = batch['speaker_labels'].to(self.device)
            boundary_labels = batch['boundaries'].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(features)
                    loss_dict = self.criterion(
                        speaker_logits=outputs['speaker_logits'],
                        boundary_logits=outputs['boundary_logits'],
                        speaker_labels=speaker_labels,
                        boundary_labels=boundary_labels,
                    )
                    loss = loss_dict['loss']

                # Backward pass with scaled gradients
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(features)
                loss_dict = self.criterion(
                    speaker_logits=outputs['speaker_logits'],
                    boundary_logits=outputs['boundary_logits'],
                    speaker_labels=speaker_labels,
                    boundary_labels=boundary_labels,
                )
                loss = loss_dict['loss']

                # Backward pass
                loss.backward()

                # Gradient clipping
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                # Optimizer step
                self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            total_speaker_loss += loss_dict['speaker_loss'].item()
            total_boundary_loss += loss_dict['boundary_loss'].item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / num_batches,
                'spk_loss': total_speaker_loss / num_batches,
                'bnd_loss': total_boundary_loss / num_batches,
            })

        metrics = {
            'loss': total_loss / num_batches,
            'speaker_loss': total_speaker_loss / num_batches,
            'boundary_loss': total_boundary_loss / num_batches,
        }

        return metrics

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        total_speaker_loss = 0.0
        total_boundary_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")

            for batch in pbar:
                # Move data to device
                features = batch['features'].to(self.device)
                speaker_labels = batch['speaker_labels'].to(self.device)
                boundary_labels = batch['boundaries'].to(self.device)

                # Forward pass
                outputs = self.model(features)
                loss_dict = self.criterion(
                    speaker_logits=outputs['speaker_logits'],
                    boundary_logits=outputs['boundary_logits'],
                    speaker_labels=speaker_labels,
                    boundary_labels=boundary_labels,
                )

                # Update metrics
                total_loss += loss_dict['loss'].item()
                total_speaker_loss += loss_dict['speaker_loss'].item()
                total_boundary_loss += loss_dict['boundary_loss'].item()
                num_batches += 1

                # Update progress bar
                pbar.set_postfix({
                    'loss': total_loss / num_batches,
                })

        metrics = {
            'loss': total_loss / num_batches,
            'speaker_loss': total_speaker_loss / num_batches,
            'boundary_loss': total_boundary_loss / num_batches,
        }

        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        early_stopping: Optional[EarlyStopping] = None,
    ) -> Dict[str, list]:
        """Train the model for multiple epochs.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            early_stopping: Early stopping callback (optional)

        Returns:
            Training history
        """
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")

            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate(val_loader)

            # Get learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log metrics
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}, "
                       f"Val Loss: {val_metrics['loss']:.4f}, "
                       f"LR: {current_lr:.6f}")

            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['learning_rate'].append(current_lr)

            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_checkpoint(epoch, val_metrics['loss'], is_best=True)
                logger.info(f"Saved best model with val_loss: {val_metrics['loss']:.4f}")

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

            # Early stopping
            if early_stopping is not None:
                if early_stopping(val_metrics['loss']):
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

            # Save regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, val_metrics['loss'], is_best=False)

        return self.history

    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        is_best: bool = False,
    ) -> None:
        """Save model checkpoint.

        Args:
            epoch: Current epoch
            val_loss: Validation loss
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save best model
        if is_best:
            checkpoint_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved best checkpoint to {checkpoint_path}")

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pt'
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'history' in checkpoint:
            self.history = checkpoint['history']

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
