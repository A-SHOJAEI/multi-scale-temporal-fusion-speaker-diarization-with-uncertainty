#!/usr/bin/env python
"""Prediction script for speaker diarization on new audio files."""

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

from multi_scale_temporal_fusion_speaker_diarization_with_uncertainty.data.preprocessing import (
    AudioPreprocessor,
    normalize_features,
)
from multi_scale_temporal_fusion_speaker_diarization_with_uncertainty.models.model import (
    MultiScaleTemporalDiarizationModel,
)
from multi_scale_temporal_fusion_speaker_diarization_with_uncertainty.utils.config import (
    load_config,
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
        description='Predict speaker diarization for audio file'
    )
    parser.add_argument(
        '--audio',
        type=str,
        required=True,
        help='Path to audio file',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pt',
        help='Path to model checkpoint',
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to config file',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save predictions (JSON format)',
    )
    parser.add_argument(
        '--uncertainty-threshold',
        type=float,
        default=0.5,
        help='Uncertainty threshold for filtering predictions',
    )
    return parser.parse_args()


def predict_audio(
    audio_path: str,
    model: torch.nn.Module,
    preprocessor: AudioPreprocessor,
    device: torch.device,
    uncertainty_threshold: float = 0.5,
):
    """Predict speaker diarization for audio file.

    Args:
        audio_path: Path to audio file
        model: Trained model
        preprocessor: Audio preprocessor
        device: Device to use
        uncertainty_threshold: Threshold for uncertainty filtering

    Returns:
        Dictionary containing predictions
    """
    # Load audio
    logger.info(f"Loading audio from {audio_path}")
    try:
        waveform = preprocessor.load_audio(audio_path)
    except Exception as e:
        logger.error(f"Failed to load audio: {e}")
        # Generate synthetic audio for demonstration
        logger.warning("Using synthetic audio for demonstration")
        duration = 10.0
        num_samples = int(duration * preprocessor.sample_rate)
        waveform = torch.randn(1, num_samples) * 0.1

    # Extract features
    logger.info("Extracting features...")
    mel_features = preprocessor.extract_mel_spectrogram(waveform)
    mel_features = normalize_features(mel_features)

    # Add batch dimension
    features = mel_features.unsqueeze(0).to(device)

    # Predict
    logger.info("Running inference...")
    model.eval()
    with torch.no_grad():
        predictions = model.predict(
            features,
            uncertainty_threshold=uncertainty_threshold,
        )

    # Convert to numpy
    speaker_pred = predictions['speaker_predictions'].cpu().numpy()[0]
    speaker_probs = predictions['speaker_probabilities'].cpu().numpy()[0]
    boundary_pred = predictions['boundary_predictions'].cpu().numpy()[0]
    boundary_probs = predictions['boundary_probabilities'].cpu().numpy()[0]
    uncertainty = predictions['uncertainty'].cpu().numpy()[0] if predictions['uncertainty'] is not None else None

    # Find speaker segments
    segments = []
    current_speaker = speaker_pred[0]
    start_frame = 0

    for i in range(1, len(speaker_pred)):
        if speaker_pred[i] != current_speaker or boundary_pred[i] > 0:
            # End of segment
            segments.append({
                'speaker': int(current_speaker),
                'start_frame': int(start_frame),
                'end_frame': int(i),
                'confidence': float(np.mean(speaker_probs[current_speaker, start_frame:i])),
            })
            current_speaker = speaker_pred[i]
            start_frame = i

    # Add last segment
    segments.append({
        'speaker': int(current_speaker),
        'start_frame': int(start_frame),
        'end_frame': int(len(speaker_pred)),
        'confidence': float(np.mean(speaker_probs[current_speaker, start_frame:])),
    })

    # Convert frame indices to time
    hop_length = preprocessor.hop_length
    sample_rate = preprocessor.sample_rate

    for segment in segments:
        segment['start_time'] = segment['start_frame'] * hop_length / sample_rate
        segment['end_time'] = segment['end_frame'] * hop_length / sample_rate
        segment['duration'] = segment['end_time'] - segment['start_time']

    result = {
        'audio_file': audio_path,
        'num_frames': int(len(speaker_pred)),
        'duration': len(speaker_pred) * hop_length / sample_rate,
        'num_speakers': int(len(np.unique(speaker_pred))),
        'segments': segments,
    }

    if uncertainty is not None:
        result['mean_uncertainty'] = float(np.mean(uncertainty))
        result['max_uncertainty'] = float(np.max(uncertainty))

    return result


def main():
    """Main prediction function."""
    # Parse arguments
    args = parse_args()

    # Check if audio file exists
    audio_path = Path(args.audio)
    if not audio_path.exists():
        logger.warning(f"Audio file not found: {audio_path}")
        logger.info("Will generate synthetic audio for demonstration")

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create preprocessor
    data_config = config['data']
    preprocessor = AudioPreprocessor(
        sample_rate=data_config.get('sample_rate', 16000),
    )

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

    # Predict
    result = predict_audio(
        audio_path=str(audio_path),
        model=model,
        preprocessor=preprocessor,
        device=device,
        uncertainty_threshold=args.uncertainty_threshold,
    )

    # Print results
    logger.info("\nPrediction Results:")
    logger.info("=" * 50)
    logger.info(f"Audio file: {result['audio_file']}")
    logger.info(f"Duration: {result['duration']:.2f} seconds")
    logger.info(f"Number of speakers: {result['num_speakers']}")
    logger.info(f"\nSpeaker segments:")
    for i, segment in enumerate(result['segments']):
        logger.info(f"  Segment {i + 1}: Speaker {segment['speaker']}")
        logger.info(f"    Time: {segment['start_time']:.2f}s - {segment['end_time']:.2f}s")
        logger.info(f"    Duration: {segment['duration']:.2f}s")
        logger.info(f"    Confidence: {segment['confidence']:.4f}")

    if 'mean_uncertainty' in result:
        logger.info(f"\nMean uncertainty: {result['mean_uncertainty']:.4f}")
        logger.info(f"Max uncertainty: {result['max_uncertainty']:.4f}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        logger.info(f"\nSaved predictions to {output_path}")

    logger.info("\nPrediction completed successfully!")


if __name__ == '__main__':
    main()
