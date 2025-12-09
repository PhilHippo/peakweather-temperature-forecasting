# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Spatiotemporal deep learning models for temperature prediction using the PeakWeather dataset from MeteoSwiss weather stations. Uses PyTorch Lightning with Hydra configuration management.

## Commands

### Installation
```bash
conda env create -f conda_env.yml
conda activate peakweather-env
```

### Training
```bash
# Basic training (runs TCN by default)
python -m experiments.run_temperature_prediction dataset=temperature model=tcn

# With custom hyperparameters
python -m experiments.run_temperature_prediction dataset=temperature model=stgnn epochs=100 batch_size=64 optimizer.hparams.lr=0.0005

# Probabilistic training with Energy Score loss
python -m experiments.run_temperature_prediction dataset=temperature model=tcn loss_fn=ens sampling.mc_samples_train=16
```

### Evaluation
```bash
# Load and evaluate saved checkpoint (skips training)
python -m experiments.run_temperature_prediction dataset=temperature model=tcn load_model_path=logs/Temperature/tcn/2025-11-20/17-53-24/epoch_XX-step_XXXXXX.ckpt
```

### Baselines (no training)
```bash
python -m experiments.run_temperature_prediction dataset=temperature model=naive
python -m experiments.run_temperature_prediction dataset=temperature model=moving_avg
```

### Batch Training
```bash
bash train_all_models.sh [optional_seed]
bash run_all_seeds.sh
```

### MLflow UI
```bash
mlflow ui --port 5000
```

## Architecture

### Entry Point
`experiments/run_temperature_prediction.py` - Main training/evaluation script. Contains `MODEL_REGISTRY` dict mapping model names to classes.

### Core Modules
- `lib/datasets/peakweather.py` - Dataset wrapper for torch-spatiotemporal, handles temporal resampling and missing data
- `lib/nn/models/learnable_models/` - Trainable models (TCN, RNN, STGNN, AttentionLongTermSTGNN)
- `lib/nn/models/baselines/` - Non-trainable baselines (Naive, MovingAverage, ICON)
- `lib/nn/predictors/` - Lightning predictors (`Predictor` for deterministic, `SamplingPredictor` for probabilistic)
- `lib/nn/layers/sampling_readout.py` - Gaussian readout layers for uncertainty quantification
- `lib/metrics/` - Energy Score (CRPS), MAE/MSE for samples

### Model Hierarchy
All models inherit from `BaseModel` (torch-spatiotemporal):
- **TCN**: Dilated causal convolutions, no graph structure
- **RNN**: GRU encoder with learnable node embeddings
- **STGNN**: Time-then-space architecture (GRU + graph convolutions)
- **AttentionLongTermSTGNN**: Patchified transformer + graph convolutions

### Configuration
Hydra configs in `config/`:
- `default.yaml` - Training defaults (epochs=120, patience=25, loss_fn=ens)
- `dataset/temperature.yaml` - Windowing (72h input, 24h horizon), connectivity (12-NN graph)
- `model/*.yaml` - Model-specific hyperparameters

### Data Format
- Tensors shape: `[batch, time, nodes, features]`
- Probabilistic samples: `[mc_samples, batch, time, nodes, features]`
- Nodes = weather stations; edges = spatial proximity (12-nearest neighbors)

### Data Splits
- Training: Before 2024-01-01
- Validation: 2024-01-01 to 2024-03-31
- Test: 2024-04-01 onwards

### Checkpoints
Stored in `logs/Temperature/{model_name}/{date}/{time}/epoch_XX-step_XXXXXX.ckpt`
