# Quick Start Guide

## Installation

```bash
# Clone or navigate to project directory
cd multi-scale-temporal-fusion-speaker-diarization-with-uncertainty

# Install dependencies
pip install -r requirements.txt
```

## Run Training

The training script will automatically generate synthetic data if real AMI corpus is not available.

### Train Full Model (with multi-scale fusion and uncertainty)

```bash
python scripts/train.py --config configs/default.yaml
```

Expected output:
- Checkpoints saved to `checkpoints/best_model.pt`
- Training history saved to `results/training_history.json`
- Config backup saved to `results/config.yaml`

### Train Baseline (single-scale, no uncertainty)

```bash
python scripts/train.py --config configs/ablation.yaml
```

## Run Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --split test \
    --visualize
```

Expected output:
- Metrics saved to `results/metrics_test.json`
- Metrics saved to `results/metrics_test.csv`
- Visualizations in `results/` (if --visualize flag used)

## Run Inference

```bash
python scripts/predict.py \
    --audio /path/to/meeting.wav \
    --checkpoint checkpoints/best_model.pt \
    --output predictions.json
```

If the audio file doesn't exist, it will use synthetic data for demonstration.

## Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_model.py -v
```

## Expected Results

After training for 100 epochs (with early stopping), you should see:

| Metric | Target | Description |
|--------|--------|-------------|
| DER | 0.12 | Diarization Error Rate |
| JER | 0.18 | Jaccard Error Rate |
| Speaker F1 | 0.88 | Speaker classification F1 |
| Boundary Precision | 0.82 | Boundary detection precision |

Note: With synthetic data, metrics will differ from targets. Use real AMI corpus for benchmark results.

## Project Structure

```
multi-scale-temporal-fusion-speaker-diarization-with-uncertainty/
├── src/                      # Source code
├── tests/                    # Unit tests (19 tests)
├── configs/                  # YAML configurations
├── scripts/                  # train.py, evaluate.py, predict.py
├── checkpoints/              # Saved models
├── results/                  # Evaluation results
└── README.md                 # Full documentation
```

## Key Features

1. **Multi-Scale Temporal Fusion**: 3 scales (frame/phoneme/word-level)
2. **Uncertainty Estimation**: Monte Carlo Dropout (10 samples)
3. **Custom Loss**: Focal loss for boundary detection
4. **Production Features**: AMP, early stopping, LR scheduling, MLflow
