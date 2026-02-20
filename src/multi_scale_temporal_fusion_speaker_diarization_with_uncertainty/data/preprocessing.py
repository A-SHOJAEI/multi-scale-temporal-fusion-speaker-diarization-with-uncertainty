"""Audio preprocessing and feature extraction for speaker diarization."""

import logging
from typing import Dict, Optional, Tuple

import librosa
import numpy as np
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, MFCC

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Preprocessor for audio files with feature extraction capabilities.

    Args:
        sample_rate: Target sampling rate for audio
        n_fft: FFT window size
        hop_length: Number of samples between successive frames
        n_mels: Number of mel filterbanks
        n_mfcc: Number of MFCC coefficients
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        n_mels: int = 80,
        n_mfcc: int = 40,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc

        self.mel_transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

        self.mfcc_transform = MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': n_fft,
                'hop_length': hop_length,
                'n_mels': n_mels,
            }
        )

    def load_audio(
        self,
        audio_path: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> torch.Tensor:
        """Load and resample audio file.

        Args:
            audio_path: Path to audio file
            start_time: Start time in seconds (optional)
            end_time: End time in seconds (optional)

        Returns:
            Audio waveform tensor of shape (1, num_samples)
        """
        try:
            waveform, sr = torchaudio.load(audio_path)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)

            # Extract segment if times provided
            if start_time is not None or end_time is not None:
                start_sample = int(start_time * self.sample_rate) if start_time else 0
                end_sample = int(end_time * self.sample_rate) if end_time else waveform.shape[1]
                waveform = waveform[:, start_sample:end_sample]

            return waveform

        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            raise

    def extract_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract mel spectrogram from waveform.

        Args:
            waveform: Audio waveform tensor

        Returns:
            Mel spectrogram tensor of shape (n_mels, time_steps)
        """
        mel_spec = self.mel_transform(waveform)
        # Apply log scaling
        mel_spec = torch.log(mel_spec + 1e-9)
        return mel_spec.squeeze(0)

    def extract_mfcc(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract MFCC features from waveform.

        Args:
            waveform: Audio waveform tensor

        Returns:
            MFCC tensor of shape (n_mfcc, time_steps)
        """
        mfcc = self.mfcc_transform(waveform)
        return mfcc.squeeze(0)


def extract_features(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    feature_type: str = "mel",
) -> torch.Tensor:
    """Extract features from audio waveform.

    Args:
        waveform: Audio waveform tensor
        sample_rate: Sampling rate
        feature_type: Type of features to extract ('mel' or 'mfcc')

    Returns:
        Feature tensor
    """
    preprocessor = AudioPreprocessor(sample_rate=sample_rate)

    if feature_type == "mel":
        return preprocessor.extract_mel_spectrogram(waveform)
    elif feature_type == "mfcc":
        return preprocessor.extract_mfcc(waveform)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")


def apply_vad(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    threshold: float = 0.5,
) -> np.ndarray:
    """Apply Voice Activity Detection to identify speech segments.

    Args:
        waveform: Audio waveform tensor
        sample_rate: Sampling rate
        threshold: Energy threshold for VAD

    Returns:
        Binary mask indicating speech activity
    """
    # Simple energy-based VAD
    frame_length = int(0.025 * sample_rate)
    hop_length = int(0.010 * sample_rate)

    waveform_np = waveform.squeeze().numpy()

    # Compute frame energy
    frames = librosa.util.frame(waveform_np, frame_length=frame_length, hop_length=hop_length)
    energy = np.sum(frames ** 2, axis=0)

    # Normalize energy
    energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-9)

    # Apply threshold
    vad_mask = (energy > threshold).astype(np.float32)

    return vad_mask


def normalize_features(features: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Normalize features using mean and standard deviation.

    Args:
        features: Feature tensor of shape (feature_dim, time_steps)
        eps: Small value to avoid division by zero

    Returns:
        Normalized features
    """
    mean = torch.mean(features, dim=-1, keepdim=True)
    std = torch.std(features, dim=-1, keepdim=True)
    normalized = (features - mean) / (std + eps)
    return normalized
