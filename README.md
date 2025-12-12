# Temperature Forecasting with PeakWeather

Spatiotemporal deep learning models for temperature prediction using the [PeakWeather](https://huggingface.co/datasets/MeteoSwiss/PeakWeather) dataset.

## Acknowledgments

This project builds upon code and data from the [PeakWeather repository](https://github.com/Graph-Machine-Learning-Group/peakweather-baselines) by Zambon et al. If you use this code or the PeakWeather dataset, please cite:

```bibtex
@misc{zambon2025peakweather,
  title={PeakWeather: MeteoSwiss Weather Station Measurements for Spatiotemporal Deep Learning}, 
  author={Zambon, Daniele and Cattaneo, Michele and Marisca, Ivan and Bhend, Jonas and Nerini, Daniele and Alippi, Cesare},
  year={2025},
  eprint={2506.13652},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2506.13652}, 
}
```

## Getting Started

### Installation

Create the conda environment:

```bash
conda env create -f conda_env.yml
conda activate peakweather-env
```

> **Note for Apple Silicon (M1/M2/M3):** PyTorch Geometric conda packages are not available. Comment out PyG-related entries in `conda_env.yml` and install via pip:
> ```bash
> pip install torch_geometric==2.4 torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.2+cpu.html
> ```

### Project Structure

```
├── config/
│   ├── default.yaml           # Default training configuration
│   ├── dataset/
│   │   └── temperature.yaml   # Dataset settings
│   └── model/
│       ├── tcn.yaml           # TCN hyperparameters
│       ├── rnn.yaml           # RNN hyperparameters
│       ├── stgnn.yaml         # STGNN hyperparameters
│       ├── naive.yaml         # Naive baseline
│       ├── seasonal_naive.yaml # Seasonal naive baseline
│       └── moving_avg.yaml    # Moving average baseline
├── experiments/
│   └── run_temperature_prediction.py
├── lib/
│   ├── datasets/              # Dataset loaders
│   ├── metrics/               # Evaluation metrics
│   └── nn/                    # Neural network models
│       ├── models/
│       │   ├── learnable_models/
│       │   └── baselines/
│       └── predictors/
└── logs/                      # Training logs and checkpoints
```

## Baseline Models

Simple non-trainable baselines for benchmarking. These models require **no training** and are evaluated directly on the test set.

### Available Baselines

| Model | Config | Description |
|-------|--------|-------------|
| **Naive** | `model=naive` | Repeats the last observed value for all horizon steps |
| **Moving Average** | `model=moving_avg` | Predicts the average of the last 24 observations |

### Running Baselines

```bash
# Naive forecast
python -m experiments.run_temperature_prediction dataset=temperature model=naive

# Moving average
python -m experiments.run_temperature_prediction dataset=temperature model=moving_avg
```

### Customizing Baselines

```bash
# 48-hour moving average window
python -m experiments.run_temperature_prediction dataset=temperature model=moving_avg model.hparams.window_size=48
```

### Why Use Baselines?

Baselines establish a performance floor that learned models should beat. If a neural network doesn't outperform a simple moving average, it's not learning meaningful patterns.

For temperature forecasting:
- **Naive**: Tests if the model beats "no change" assumption
- **Moving Average**: Tests if the model beats simple smoothing

## Learnable Models

### Available Models

| Model | Config | Description | Uses Graph |
|-------|--------|-------------|------------|
| `tcn` | `model=tcn` | Temporal Convolutional Network | ❌ |
| `rnn` | `model=rnn` | GRU-based RNN with node embeddings | ❌ |
| `stgnn` | `model=stgnn` | Spatiotemporal Graph Neural Network | ✅ |
| `attn_longterm` | `model=attn_longterm` | Attention-based STGNN with long-term dependencies | ✅ |

### Training

```bash
# Basic training
python -m experiments.run_temperature_prediction dataset=temperature model=tcn

# Custom configuration
python -m experiments.run_temperature_prediction \
    dataset=temperature \
    model=tcn \
    epochs=100 \
    batch_size=64 \
    optimizer.hparams.lr=0.0005
```

### Probabilistic Training (Energy Score Loss)

For probabilistic forecasting with uncertainty estimation:

```bash
python -m experiments.run_temperature_prediction \
    dataset=temperature \
    model=tcn \
    loss_fn=ens \
    sampling.mc_samples_train=16 \
    sampling.mc_samples_eval=11 \
    sampling.mc_samples_test=20  # lower test MC samples to avoid OOM
```

> **Note:** When using `loss_fn=ens` (Energy Score), learnable models generate Monte Carlo samples to estimate prediction uncertainty. Baselines are deterministic only and should be evaluated with `loss_fn=mae`.

## Evaluation

### Load and Test a Saved Model

```bash
python -m experiments.run_temperature_prediction \
    dataset=temperature \
    model=tcn \
    load_model_path=TCN-all-nodes/epoch_68-step_66033.ckpt \
    sampling.mc_samples_test=20  # safe value for EnergyScore test
```

When `load_model_path` is specified, training is skipped and the model is evaluated directly on the test set.

## Experiment Tracking

Track experiments with MLflow:

```bash
mlflow ui --port 5000
```

Then open `http://127.0.0.1:5000` in your browser.

For remote tracking, specify the URI when running:

```bash
python -m experiments.run_temperature_prediction ++mlflow_tracking_uri=<YOUR_URI>
```

## Configuration Reference

### Dataset (`config/dataset/temperature.yaml`)

- `target_channels`: Variables to predict (e.g., `temperature`)
- `covariate_channels`: Input features (`other` uses all non-target variables)
- `freq`: Temporal resolution (`h` for hourly)
- `static_attributes`: Topographic features (altitude, slope, aspect, etc.)
- `connectivity`: Graph structure settings

### Training (`config/default.yaml`)

- `epochs`: Maximum training epochs
- `patience`: Early stopping patience
- `batch_size`: Training batch size
- `loss_fn`: Loss function (`mae` for deterministic, `ens` for probabilistic)
- `optimizer.hparams.lr`: Learning rate

### Model Hyperparameters (`config/model/*.yaml`)

Each model has its own config file with architecture-specific parameters (hidden size, number of layers, dropout, etc.).

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.
