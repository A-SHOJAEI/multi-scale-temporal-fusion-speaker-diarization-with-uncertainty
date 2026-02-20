"""Pytest fixtures and configuration."""

import pytest
import torch
import numpy as np


@pytest.fixture
def device():
    """Get torch device for testing."""
    return torch.device('cpu')


@pytest.fixture
def sample_audio_features():
    """Generate sample audio features for testing."""
    batch_size = 4
    feature_dim = 80
    time_steps = 100

    features = torch.randn(batch_size, feature_dim, time_steps)
    return features


@pytest.fixture
def sample_labels():
    """Generate sample labels for testing."""
    batch_size = 4
    time_steps = 100
    num_speakers = 4

    speaker_labels = torch.randint(0, num_speakers, (batch_size, time_steps))
    boundaries = torch.zeros(batch_size, time_steps)

    # Add some boundaries
    for i in range(batch_size):
        num_boundaries = np.random.randint(2, 5)
        boundary_positions = np.random.choice(time_steps, num_boundaries, replace=False)
        boundaries[i, boundary_positions] = 1.0

    return {
        'speaker_labels': speaker_labels,
        'boundaries': boundaries,
    }


@pytest.fixture
def model_config():
    """Configuration for model testing."""
    return {
        'input_dim': 80,
        'hidden_dim': 64,
        'num_speakers': 4,
        'dropout': 0.1,
        'mc_dropout': 0.2,
        'mc_samples': 5,
    }
