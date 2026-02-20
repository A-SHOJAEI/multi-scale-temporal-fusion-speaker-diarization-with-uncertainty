# Multi-Scale Temporal Fusion Speaker Diarization with Uncertainty Quantification

A novel speaker diarization system that combines multi-scale temporal feature fusion with uncertainty-aware clustering for robust speaker change detection in challenging acoustic environments.

## Overview

This system addresses the challenge of accurate speaker change detection in overlapping speech and noisy meeting scenarios where traditional fixed-scale approaches fail. The approach uses parallel temporal convolutions at different scales (frame-level ~30ms, phoneme-level ~100ms, word-level ~280ms) merged via learned attention weights, followed by Monte Carlo Dropout-based uncertainty estimation to adaptively reject low-confidence speaker boundaries.

## Key Features

- Multi-scale temporal fusion with attention-based feature aggregation
- Uncertainty quantification using Monte Carlo Dropout for boundary confidence estimation
- Custom focal loss for handling class imbalance in boundary detection
- Joint speaker classification and boundary detection framework
- Comprehensive evaluation metrics (DER, JER, Speaker F1, Boundary Precision)

## Installation

```bash
pip install -r requirements.txt
```

For development with testing tools:

```bash
pip install -e ".[dev]"
```

## Quick Start

### Training

Train the full model with multi-scale fusion and uncertainty estimation:

```bash
python scripts/train.py --config configs/default.yaml
```

Train the baseline model (single-scale, no uncertainty):

```bash
python scripts/train.py --config configs/ablation.yaml
```

### Evaluation

Evaluate the trained model on the test set:

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --split test --visualize
```

### Inference

Run inference on a new audio file:

```bash
python scripts/predict.py --audio /path/to/audio.wav --checkpoint checkpoints/best_model.pt --output predictions.json
```

## Methodology

### Problem Formulation

Speaker diarization is formulated as a joint task of speaker classification and boundary detection on temporal sequences. Given an audio input $X \in \mathbb{R}^{T \times F}$ where $T$ is the number of time frames and $F$ is the feature dimension, the model predicts:

1. Speaker labels $y_s \in \{1, ..., K\}^T$ for each time frame
2. Speaker boundaries $y_b \in \{0, 1\}^T$ indicating speaker changes
3. Uncertainty estimates $u \in \mathbb{R}^T$ for confidence-aware post-processing

### Multi-Scale Temporal Fusion

Traditional approaches use fixed-scale temporal modeling, which struggles when speaker changes occur at varying temporal granularities (e.g., rapid turn-taking vs. long monologues). This work introduces parallel temporal convolutions operating at three scales:

- **Frame-level (kernel=3, dilation=1)**: Captures ~30ms patterns for precise boundary localization
- **Phoneme-level (kernel=5, dilation=2)**: Captures ~100ms patterns for phonetic context
- **Word-level (kernel=7, dilation=4)**: Captures ~280ms patterns for lexical-level speaker characteristics

Each scale extracts complementary temporal features through residual temporal convolution blocks. Rather than simple concatenation, the scales are merged via a learned attention mechanism that weights each scale's contribution based on local temporal context:

$$\alpha_t = \text{softmax}(W_2 \cdot \text{ReLU}(W_1 \cdot h_t))$$

where $h_t$ is the concatenated multi-scale feature at time $t$, and $\alpha_t \in \mathbb{R}^3$ are the attention weights for the three scales. The final fused representation is:

$$z_t = \sum_{s=1}^{3} \alpha_{t,s} \cdot h_{t,s}$$

This allows the model to dynamically emphasize fine-grained features at speaker boundaries and coarse-grained features within homogeneous segments.

### Uncertainty-Aware Prediction

Speaker diarization in noisy environments often produces spurious boundaries. To address this, the model incorporates Monte Carlo Dropout-based uncertainty estimation. During inference, multiple forward passes are performed with dropout enabled:

$$\hat{y}^{(1)}, ..., \hat{y}^{(M)} \sim p(y | X, \theta)$$

The prediction uncertainty is quantified as the variance across samples:

$$u_t = \text{Var}(\hat{y}_t^{(1)}, ..., \hat{y}_t^{(M)})$$

Boundaries with uncertainty exceeding a threshold $\tau$ are rejected, reducing false alarms in challenging acoustic conditions.

### Custom Focal Loss

Speaker boundaries are sparse events (~2-5% of frames in typical conversations), creating severe class imbalance. To address this, the model uses focal loss for boundary detection:

$$\mathcal{L}_{\text{focal}} = -\alpha (1 - p_t)^\gamma \log(p_t)$$

where $p_t$ is the predicted probability for the true class, $\alpha=0.25$ balances positive/negative examples, and $\gamma=2$ down-weights easy examples. This focuses training on hard-to-classify boundary frames.

The total loss combines speaker classification (cross-entropy) and boundary detection (focal loss):

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{speaker}} + \lambda \mathcal{L}_{\text{boundary}}$$

with $\lambda=2.0$ to emphasize boundary accuracy.

## Model Architecture

The complete system pipeline:

1. **Feature Extraction**: 80-dimensional mel-spectrograms (16kHz, 25ms window, 10ms hop)
2. **Multi-Scale Temporal Fusion**: Parallel dilated convolutions at 3 scales with attention-based fusion (256-dim hidden)
3. **Uncertainty Estimator**: 2-layer MLP with MC Dropout (p=0.2, M=10 samples)
4. **Prediction Heads**:
   - Speaker classification: Linear layer → 10 speakers
   - Boundary detection: Linear layer → binary classification

## Configuration

All hyperparameters are configurable via YAML files in `configs/`. Key parameters include:

- `model.hidden_dim`: Hidden dimension for temporal convolutions (default: 256)
- `model.mc_dropout`: Dropout rate for uncertainty estimation (default: 0.2)
- `model.mc_samples`: Number of MC samples for uncertainty (default: 10)
- `loss.boundary_weight`: Weight for boundary detection loss (default: 2.0)
- `training.learning_rate`: Initial learning rate (default: 0.001)

## Results

Run `python scripts/train.py` to reproduce results. Target metrics on AMI Meeting Corpus:

| Metric | Target | Description |
|--------|--------|-------------|
| DER | 0.12 | Diarization Error Rate |
| JER | 0.18 | Jaccard Error Rate |
| Speaker F1 | 0.88 | Speaker classification F1 score |
| Boundary Precision | 0.82 | Boundary detection precision |

## Ablation Study

Compare the full model against the baseline by training both configurations:

```bash
python scripts/train.py --config configs/default.yaml
python scripts/train.py --config configs/ablation.yaml
```

The baseline removes multi-scale fusion (single scale only) and uncertainty estimation to isolate the contribution of the novel components.

## Project Structure

```
multi-scale-temporal-fusion-speaker-diarization-with-uncertainty/
├── src/                          # Source code
│   └── multi_scale_temporal_fusion_speaker_diarization_with_uncertainty/
│       ├── data/                 # Data loading and preprocessing
│       ├── models/               # Model architecture and components
│       ├── training/             # Training loop and utilities
│       ├── evaluation/           # Metrics and analysis
│       └── utils/                # Configuration and utilities
├── tests/                        # Unit tests
├── configs/                      # Configuration files
├── scripts/                      # Training, evaluation, and inference scripts
├── checkpoints/                  # Saved model checkpoints
└── results/                      # Evaluation results and visualizations
```

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

With coverage:

```bash
pytest tests/ --cov=src --cov-report=html
```

## License

MIT License - Copyright (c) 2026 Alireza Shojaei. See [LICENSE](LICENSE) for details.
