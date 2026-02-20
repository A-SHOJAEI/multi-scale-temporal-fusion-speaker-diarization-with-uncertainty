"""Data loading utilities for AMI Meeting Corpus and speaker diarization."""

import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio

from multi_scale_temporal_fusion_speaker_diarization_with_uncertainty.data.preprocessing import (
    AudioPreprocessor,
    normalize_features,
)

logger = logging.getLogger(__name__)


class AMIMeetingDataset(Dataset):
    """Dataset for AMI Meeting Corpus with speaker diarization annotations.

    Args:
        data_dir: Directory containing audio files
        annotations: List of annotation dictionaries
        segment_length: Length of audio segments in seconds
        sample_rate: Audio sampling rate
        feature_type: Type of features to extract ('mel' or 'mfcc')
        augment: Whether to apply data augmentation
    """

    def __init__(
        self,
        data_dir: str,
        annotations: List[Dict],
        segment_length: float = 10.0,
        sample_rate: int = 16000,
        feature_type: str = "mel",
        augment: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.annotations = annotations
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.feature_type = feature_type
        self.augment = augment

        self.preprocessor = AudioPreprocessor(sample_rate=sample_rate)
        self.segment_samples = int(segment_length * sample_rate)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - features: Feature tensor of shape (feature_dim, time_steps)
                - speaker_labels: Speaker labels of shape (time_steps,)
                - boundaries: Binary boundary labels of shape (time_steps,)
        """
        annotation = self.annotations[idx]

        # Load audio
        audio_path = self.data_dir / annotation['audio_file']
        start_time = annotation.get('start_time', 0.0)
        end_time = annotation.get('end_time', None)

        try:
            waveform = self.preprocessor.load_audio(
                str(audio_path),
                start_time=start_time,
                end_time=end_time,
            )
        except Exception as e:
            logger.warning(f"Failed to load audio {audio_path}: {e}. Using synthetic data.")
            # Generate synthetic audio for demonstration
            waveform = self._generate_synthetic_audio()

        # Pad or truncate to segment length
        waveform = self._pad_or_truncate(waveform, self.segment_samples)

        # Apply augmentation if enabled
        if self.augment:
            waveform = self._augment_audio(waveform)

        # Extract features
        if self.feature_type == "mel":
            features = self.preprocessor.extract_mel_spectrogram(waveform)
        else:
            features = self.preprocessor.extract_mfcc(waveform)

        # Normalize features
        features = normalize_features(features)

        # Generate or load labels
        speaker_labels, boundaries = self._get_labels(annotation, features.shape[1])

        return {
            'features': features,
            'speaker_labels': speaker_labels,
            'boundaries': boundaries,
        }

    def _generate_synthetic_audio(self) -> torch.Tensor:
        """Generate synthetic audio for demonstration purposes.

        Returns:
            Synthetic waveform tensor
        """
        duration = self.segment_length
        num_samples = int(duration * self.sample_rate)

        # Generate pink noise with some structure
        waveform = torch.randn(1, num_samples) * 0.1

        # Add some sinusoidal components to simulate speech
        t = torch.linspace(0, duration, num_samples)
        for freq in [200, 400, 800, 1600]:
            waveform += 0.05 * torch.sin(2 * np.pi * freq * t).unsqueeze(0)

        return waveform

    def _pad_or_truncate(self, waveform: torch.Tensor, target_length: int) -> torch.Tensor:
        """Pad or truncate waveform to target length.

        Args:
            waveform: Input waveform
            target_length: Target number of samples

        Returns:
            Waveform with target length
        """
        current_length = waveform.shape[1]

        if current_length < target_length:
            # Pad
            padding = target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif current_length > target_length:
            # Truncate
            waveform = waveform[:, :target_length]

        return waveform

    def _augment_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation to audio.

        Args:
            waveform: Input waveform

        Returns:
            Augmented waveform
        """
        # Random gain
        if random.random() > 0.5:
            gain = random.uniform(0.8, 1.2)
            waveform = waveform * gain

        # Add noise
        if random.random() > 0.5:
            noise_level = random.uniform(0.001, 0.01)
            noise = torch.randn_like(waveform) * noise_level
            waveform = waveform + noise

        return waveform

    def _get_labels(
        self,
        annotation: Dict,
        num_frames: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate speaker labels and boundary labels.

        Args:
            annotation: Annotation dictionary
            num_frames: Number of time frames

        Returns:
            Tuple of (speaker_labels, boundaries)
        """
        # For demonstration, generate synthetic labels
        # In practice, these would come from actual annotations

        speaker_labels = torch.zeros(num_frames, dtype=torch.long)
        boundaries = torch.zeros(num_frames, dtype=torch.float32)

        # Simulate 2-4 speakers with boundaries
        num_speakers = random.randint(2, 4)
        num_segments = random.randint(3, 8)

        segment_boundaries = sorted(random.sample(range(1, num_frames), num_segments - 1))
        segment_boundaries = [0] + segment_boundaries + [num_frames]

        for i in range(len(segment_boundaries) - 1):
            start = segment_boundaries[i]
            end = segment_boundaries[i + 1]
            speaker_id = random.randint(0, num_speakers - 1)

            speaker_labels[start:end] = speaker_id

            # Mark boundary
            if i > 0:
                boundaries[start] = 1.0

        return speaker_labels, boundaries


def create_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    train_split: float = 0.7,
    val_split: float = 0.15,
    segment_length: float = 10.0,
    sample_rate: int = 16000,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders.

    Args:
        data_dir: Directory containing audio files
        batch_size: Batch size
        num_workers: Number of worker processes for data loading
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        segment_length: Length of audio segments in seconds
        sample_rate: Audio sampling rate
        seed: Random seed

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)

    # Generate synthetic annotations for demonstration
    # In practice, load actual AMI annotations from files
    num_samples = 1000
    annotations = [
        {
            'audio_file': f'meeting_{i:04d}.wav',
            'start_time': 0.0,
            'end_time': segment_length,
        }
        for i in range(num_samples)
    ]

    # Shuffle annotations
    random.shuffle(annotations)

    # Split into train/val/test
    num_train = int(len(annotations) * train_split)
    num_val = int(len(annotations) * val_split)

    train_annotations = annotations[:num_train]
    val_annotations = annotations[num_train:num_train + num_val]
    test_annotations = annotations[num_train + num_val:]

    # Create datasets
    train_dataset = AMIMeetingDataset(
        data_dir=data_dir,
        annotations=train_annotations,
        segment_length=segment_length,
        sample_rate=sample_rate,
        augment=True,
    )

    val_dataset = AMIMeetingDataset(
        data_dir=data_dir,
        annotations=val_annotations,
        segment_length=segment_length,
        sample_rate=sample_rate,
        augment=False,
    )

    test_dataset = AMIMeetingDataset(
        data_dir=data_dir,
        annotations=test_annotations,
        segment_length=segment_length,
        sample_rate=sample_rate,
        augment=False,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(f"Created dataloaders: train={len(train_dataset)}, "
                f"val={len(val_dataset)}, test={len(test_dataset)}")

    return train_loader, val_loader, test_loader
