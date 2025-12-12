# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GDL (Graph Deep Learning) course project for temperature forecasting using the PeakWeather dataset from MeteoSwiss. The goal is to assess spatiotemporal graph neural networks (STGNNs) for temperature prediction and whether lower-quality rain gauge observations improve forecasting accuracy.

Built on PyTorch Lightning, tsl (torch-spatiotemporal), and Hydra for configuration.

## Commands

### Environment Setup
```bash
conda env create -f conda_env.yml
conda activate peakweather-env
```

### Training Models
```bash
# Basic training (uses Hydra config system)
python -m experiments.run_temperature_prediction dataset=temperature model=tcn

# With custom hyperparameters
python -m experiments.run_temperature_prediction dataset=temperature model=stgnn epochs=100 batch_size=64 optimizer.hparams.lr=0.0005

# Probabilistic training with Energy Score loss
python -m experiments.run_temperature_prediction dataset=temperature model=tcn loss_fn=ens sampling.mc_samples_train=16

# Train with specific seed (run 5 seeds for final evaluation)
python -m experiments.run_temperature_prediction dataset=temperature model=stgnn seed=42
```

### Evaluating Models
```bash
# Load and evaluate a saved checkpoint (skips training)
python -m experiments.run_temperature_prediction dataset=temperature model=tcn load_model_path=path/to/checkpoint.ckpt

# Run baselines (no training needed)
python -m experiments.run_temperature_prediction dataset=temperature model=naive
python -m experiments.run_temperature_prediction dataset=temperature model=moving_avg

# Evaluate ICON NWP baseline
python -m experiments.run_temperature_prediction dataset=temperature model=icon
```

### Experiment Tracking
```bash
mlflow ui --port 5000  # View at http://127.0.0.1:5000
```

## Dataset Details

### PeakWeather Dataset
- **Source**: SwissMetNet network operated by MeteoSwiss
- **Stations**: 302 total (160 meteo stations + 142 rain gauges)
- **Variables**: temperature, humidity, precipitation, sunshine, pressure, wind_speed, wind_gust, wind_direction
- **Resolution**: 10-minute intervals, resampled to hourly for this project
- **Time span**: Jan 2017 - Mar 2025 (8+ years)
- **Topographic features**: altitude, slope, aspect, TPI, STD at 2km and 10km scales

### Station Types
- **Meteo stations**: High-quality temperature sensors, full sensor suite
- **Rain gauges**: Lower-grade temperature sensors (primarily for precipitation type discrimination)

### Data Splits
- Training: until Dec 31, 2023
- Validation: Jan 1, 2024 - Mar 31, 2024
- Test: starts April 1, 2024

### NWP Baseline (ICON-CH1-EPS)
- 11-member ensemble (1 control + 10 perturbed)
- Forecasts every 3 hours, up to 33 hours ahead
- Available since May 14, 2024
- Use ensemble median for MAE, all members for EnergyScore

## Architecture

### Configuration (Hydra)
- `config/default.yaml`: Training defaults (epochs, batch_size, loss_fn, optimizer)
- `config/dataset/temperature.yaml`: Dataset settings (target_channels, covariates, connectivity)
- `config/model/*.yaml`: Model-specific hyperparameters

### Model Registry (`experiments/run_temperature_prediction.py`)
Learnable models: `tcn`, `rnn`, `stgnn`, `attn_longterm`
Baselines: `naive`, `moving_avg`, `icon`

### Model Descriptions
- **TCN**: Temporal Convolutional Network (no graph structure)
- **RNN**: GRU-based with node embeddings (no graph structure)
- **STGNN**: Spatiotemporal GNN with diffusion convolution + gated TCN
- **AttentionLongTermSTGNN** (`attn_longterm`): Two-stage architecture inspired by traffic prediction literature:
  1. Long-term feature extraction via masked autoencoding with Transformer
  2. Prediction via attention-based STGNN (STAWnet backbone)

### Core Components
- **lib/datasets/peakweather.py**: `PeakWeather` class wrapping the PeakWeatherDataset with tsl's `DatetimeDataset`
- **lib/nn/models/learnable_models/**: Neural network architectures
- **lib/nn/models/baselines/**: Non-trainable baselines (Naive, MovingAverage, ICON NWP)
- **lib/nn/predictors/predictor.py**: `Predictor` and `SamplingPredictor` (for probabilistic forecasting)
- **lib/metrics/**: `EnergyScore` for probabilistic loss, `SampleMAE`/`SampleMSE` for sample-based evaluation

### Probabilistic Forecasting
When `loss_fn=ens`, models use `SamplingPredictor` which generates Monte Carlo samples:
- `sampling.mc_samples_train`: Samples during training (default: 16)
- `sampling.mc_samples_eval`: Samples during validation (default: 11)
- `sampling.mc_samples_test`: Samples during testing (default: 20)

## Evaluation Requirements

### Metrics
- **MAE**: Mean Absolute Error (use SampleMAE with ensemble median for probabilistic models)
- **EnergyScore (CRPS)**: Continuous Ranked Probability Score for probabilistic forecasts

### Report metrics at specific horizons
t=1h, t=3h, t=6h, t=12h, t=18h, t=24h

### Experimental Protocol
1. Tune hyperparameters on validation MAE
2. Train best configuration with 5 different seeds
3. Report mean and std on test set

### Station Subset Evaluations
- `meteo_only`: Only meteo stations
- `all_with_temp`: All stations with temperature data
- When using rain gauges: compare train-all/test-meteo vs train-all/test-all

## Reference Documentation

See `docs/` folder:
- `peakweather-paper.pdf`: Dataset paper (arXiv:2506.13652)
- `peakweather-project.pdf`: Project deliverables and requirements
- `stgnn-att-paper.pdf`: Attention-based STGNN architecture reference
