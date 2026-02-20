"""Tests for data loading and preprocessing."""

import pytest
import torch
import numpy as np

from multi_scale_temporal_fusion_speaker_diarization_with_uncertainty.data.preprocessing import (
    AudioPreprocessor,
    extract_features,
    normalize_features,
)
from multi_scale_temporal_fusion_speaker_diarization_with_uncertainty.data.loader import (
    AMIMeetingDataset,
)


class TestAudioPreprocessor:
    """Test AudioPreprocessor class."""

    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = AudioPreprocessor(sample_rate=16000)
        assert preprocessor.sample_rate == 16000
        assert preprocessor.n_fft == 512
        assert preprocessor.hop_length == 160

    def test_extract_mel_spectrogram(self):
        """Test mel spectrogram extraction."""
        preprocessor = AudioPreprocessor(sample_rate=16000)
        waveform = torch.randn(1, 16000)  # 1 second of audio

        mel_spec = preprocessor.extract_mel_spectrogram(waveform)

        assert mel_spec.dim() == 2
        assert mel_spec.shape[0] == 80  # n_mels

    def test_extract_mfcc(self):
        """Test MFCC extraction."""
        preprocessor = AudioPreprocessor(sample_rate=16000)
        waveform = torch.randn(1, 16000)

        mfcc = preprocessor.extract_mfcc(waveform)

        assert mfcc.dim() == 2
        assert mfcc.shape[0] == 40  # n_mfcc


class TestDataset:
    """Test AMIMeetingDataset class."""

    def test_dataset_creation(self, tmp_path):
        """Test dataset creation."""
        annotations = [
            {'audio_file': 'test.wav', 'start_time': 0.0, 'end_time': 10.0}
        ]

        dataset = AMIMeetingDataset(
            data_dir=str(tmp_path),
            annotations=annotations,
            segment_length=5.0,
        )

        assert len(dataset) == 1

    def test_dataset_getitem(self, tmp_path):
        """Test getting items from dataset."""
        annotations = [
            {'audio_file': 'test.wav', 'start_time': 0.0, 'end_time': 10.0}
        ]

        dataset = AMIMeetingDataset(
            data_dir=str(tmp_path),
            annotations=annotations,
            segment_length=5.0,
        )

        sample = dataset[0]

        assert 'features' in sample
        assert 'speaker_labels' in sample
        assert 'boundaries' in sample
        assert sample['features'].dim() == 2


def test_extract_features():
    """Test feature extraction function."""
    waveform = torch.randn(1, 16000)

    mel_features = extract_features(waveform, feature_type='mel')
    assert mel_features.dim() == 2

    mfcc_features = extract_features(waveform, feature_type='mfcc')
    assert mfcc_features.dim() == 2


def test_normalize_features():
    """Test feature normalization."""
    features = torch.randn(80, 100)

    normalized = normalize_features(features)

    assert normalized.shape == features.shape
    # Check if roughly normalized (mean ~0, std ~1)
    assert torch.abs(normalized.mean()) < 0.5
    assert torch.abs(normalized.std() - 1.0) < 0.5
