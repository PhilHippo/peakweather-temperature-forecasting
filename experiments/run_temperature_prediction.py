import os
import sys
import numpy as np
import omegaconf
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Callback
from pytorch_lightning.loggers import MLFlowLogger
import pytorch_lightning as pl
import mlflow
from tsl import logger
from tsl.data import SpatioTemporalDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import scalers
from tsl.experiment import Experiment
from tsl.metrics import torch_metrics

import lib
from lib.datasets import PeakWeather
from lib.nn import models
import lib.metrics
from lib.nn.predictors import Predictor, SamplingPredictor

# Learnable models (require training)
LEARNABLE_MODELS = {
    'tcn': models.TCNModel,
    'rnn': models.RNNModel,
    'stgnn': models.STGNN,
    'attn_longterm': models.AttentionLongTermSTGNN,
    'model3': models.Model3,
    'model3_old': models.Model3Old,
}

# Baseline models (no training required)
BASELINE_MODELS = {
    'naive': models.NaiveModel,
    'moving_avg': models.MovingAverageModel,
    'icon': models.ICONDummyModel,
}

MODEL_REGISTRY = {**LEARNABLE_MODELS, **BASELINE_MODELS}


class MLflowFlushCallback(Callback):
    """Callback to flush MLflow metrics in real-time after each epoch."""
    
    def _flush_mlflow(self, logger):
        """Flush MLflow run to ensure metrics are written immediately."""
        if isinstance(logger, MLFlowLogger):
            try:
                # Get the MLflow run ID
                run_id = logger.run_id
                if run_id:
                    # Set tracking URI if needed
                    tracking_uri = logger.tracking_uri or './mlruns'
                    mlflow.set_tracking_uri(tracking_uri)
                    
                    # Get the active run and ensure it's synced
                    # Accessing the run forces a sync with the backend
                    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
                    run = client.get_run(run_id)
                    
                    # Force file system sync by accessing run data
                    # This ensures the file backend writes are flushed and visible in real-time
                    _ = run.info
                    _ = run.data
            except Exception:
                # Silently fail if flush doesn't work
                pass
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Flush MLflow after each training epoch."""
        for logger in trainer.loggers:
            self._flush_mlflow(logger)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Flush MLflow after each validation epoch."""
        for logger in trainer.loggers:
            self._flush_mlflow(logger)


def get_model_class(model_str: str):
    """Get model class from string identifier."""
    if model_str not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(f'Model "{model_str}" not available. Choose from: {available}')
    return MODEL_REGISTRY[model_str]


def is_baseline_model(model_str: str) -> bool:
    """Check if model is a non-trainable baseline."""
    return model_str in BASELINE_MODELS


def is_icon_model(model_str: str) -> bool:
    """Check if model is the ICON NWP baseline."""
    return model_str == 'icon'


def run(cfg: DictConfig):
    ########################################
    # Set Random Seed                      #
    ########################################
    
    # Get seed from config (can be set via command line: seed=1)
    # If not provided, Hydra will auto-generate one in cfg.run.seed
    seed = cfg.get('seed', None)
    if seed is None:
        # Fall back to Hydra's auto-generated seed if available
        seed = cfg.get('run', {}).get('seed', None)
    
    if seed is not None:
        # Set seed for reproducibility
        pl.seed_everything(seed, workers=True)
        logger.info(f"Seed set to {seed}")
    
    
    ########################################
    # Get Dataset                          #
    ########################################
    
    # Check if we need ICON data
    use_icon = is_icon_model(cfg.model.name) or cfg.get('nwp_test_set', False)
    
    # Prepare dataset params, adding ICON if needed
    dataset_params = dict(cfg.dataset.hparams)
    if use_icon:
        # Ensure temperature ICON data is loaded
        dataset_params['extended_nwp_vars'] = ['temperature']
    
    # Load dataset with explicit target and covariate separation
    dataset = PeakWeather(**dataset_params)
    
    # Get connectivity
    adj = dataset.get_connectivity(**cfg.dataset.connectivity)
    # Get mask
    mask = dataset.get_mask()

    # Get covariates
    # Similar to run_wind_prediction.py, but simplified for temperature
    u = []
    if cfg.dataset.get('covariates', {}).get('year', False):
        u.append(dataset.datetime_encoded('year').values)
    if cfg.dataset.get('covariates', {}).get('day', False):
        u.append(dataset.datetime_encoded('day').values)
    if cfg.dataset.get('covariates', {}).get('weekday', False):
        u.append(dataset.datetime_onehot('weekday').values)
    if cfg.dataset.get('covariates', {}).get('mask', False):
        u.append(mask.astype(np.float32))
        
    # Add 'u' from dataset (which contains other weather vars if configured)
    # Extract the actual numpy array using get_frame, similar to run_wind_prediction.py
    if 'u' in dataset.covariates:
        other_channels = dataset.get_frame('u', return_pattern=False)
        u.append(other_channels)
        if cfg.dataset.get('covariates', {}).get('u_mask', False):
            u_mask = dataset.get_frame('u_mask', return_pattern=False)
            u.append(u_mask)
        
    # Concatenate covariates
    if len(u):
        ndim = max(u_.ndim for u_ in u)
        u = np.concatenate([np.repeat(u_[:, None], dataset.n_nodes, 1)
                            if u_.ndim < ndim else u_
                            for u_ in u], axis=-1)
    else:
        u = None

    # Get static information
    covs = {}
    if u is not None:
        covs['u'] = u
        
    # Add static variables if present (e.g. topography)
    # In PeakWeather dataset.py, static vars might be in stations_table or separate.
    # run_wind_prediction.py added 'v' manually from stations_table.
    if cfg.dataset.get('static_attributes', None):
        v = dataset.stations_table[[*cfg.dataset.static_attributes]]
        # Standardize with safeguard against zero std
        v_mean = v.mean(0)
        v_std = v.std(0)
        # Replace zero std with 1.0 to avoid division by zero
        v_std = v_std.replace(0.0, 1.0)
        v = (v - v_mean) / v_std
        covs["v"] = v.values

    # Scale input features
    scaler_cfg = cfg.get('scaler')
    if scaler_cfg is not None:
        scale_axis = (0,) if scaler_cfg.axis == 'node' else (0, 1)
        scaler_cls = getattr(scalers, f'{scaler_cfg.method}Scaler')
        transform = dict(target=scaler_cls(axis=scale_axis))
        scale_target = scaler_cfg.get('scale_target', False)
    else:
        # Default scaling for temperature if not specified
        transform = dict(target=scalers.StandardScaler(axis=(0, 1)))
        scale_target = True

    torch_dataset = SpatioTemporalDataset(dataset.dataframe(),
                                          mask=dataset.mask,
                                          covariates=covs,
                                          connectivity=adj,
                                          horizon=cfg.horizon,
                                          window=cfg.window,
                                          stride=cfg.stride)
    
    # Sanity check for NaN/Inf in data
    data_df = dataset.dataframe()
    if np.isnan(data_df.values).any():
        logger.warning(f"Found {np.isnan(data_df.values).sum()} NaN values in target data")
    if np.isinf(data_df.values).any():
        logger.warning(f"Found {np.isinf(data_df.values).sum()} Inf values in target data")
    
    if 'u' in covs:
        if np.isnan(covs['u']).any():
            logger.warning(f"Found {np.isnan(covs['u']).sum()} NaN values in covariates")
        if np.isinf(covs['u']).any():
            raise ValueError(f"Found {np.isinf(covs['u']).sum()} Inf values in covariates - check data preprocessing!")
    
    if 'v' in covs:
        if np.isnan(covs['v']).any():
            logger.warning(f"Found {np.isnan(covs['v']).sum()} NaN values in static attributes")
        if np.isinf(covs['v']).any():
            raise ValueError(f"Found {np.isinf(covs['v']).sum()} Inf values in static attributes - check standardization!")

    print("Creating data module...")
    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=transform,
        splitter=dataset.get_splitter(
            first_val_ts=[2024, 1, 1],
            first_test_ts=[2024, 4, 1]
        ),
        mask_scaling=True,
        batch_size=cfg.batch_size,
        workers=cfg.get('workers', 0)
    )
    print("Setting up data module...")
    dm.setup()
    print("Data module setup complete.")

    print(f"Split sizes\n\tTrain: {len(dm.trainset)}\n"
          f"\tValidation: {len(dm.valset)}\n"
          f"\tTest: {len(dm.testset)}")

    ########################################
    # Create model                         #
    ########################################

    model_cls = get_model_class(cfg.model.name)

    d_exog = torch_dataset.input_map.u.shape[-1] if 'u' in covs else 0
    d_exog += torch_dataset.input_map.v.shape[-1] if 'v' in covs else 0
    
    model_kwargs = dict(n_nodes=torch_dataset.n_nodes,
                        input_size=torch_dataset.n_channels,
                        exog_size=d_exog,
                        output_size=torch_dataset.n_channels,
                        horizon=torch_dataset.horizon)

    # filter_model_args_ might not exist on custom models, so we might need to be careful
    if hasattr(model_cls, 'filter_model_args_'):
        model_cls.filter_model_args_(model_kwargs)
    model_kwargs.update(cfg.model.hparams)

    ########################################
    # predictor                            #
    ########################################

    # Determine if we're using a baseline model
    is_baseline_or_icon = is_baseline_model(cfg.model.name) or is_icon_model(cfg.model.name)
    
    # Use sample-based loss for probabilistic models, point-based for deterministic baselines
    if cfg.get('loss_fn') == "mae":
        # For deterministic baselines, use point MAE; for learned models, use sample MAE
        loss_fn = torch_metrics.MaskedMAE() if is_baseline_or_icon and not is_icon_model(cfg.model.name) else lib.metrics.SampleMAE()
    elif cfg.get('loss_fn') == "ens":
        loss_fn = lib.metrics.EnergyScore()
    else:
        # Default to SampleMAE for probabilistic evaluation
        loss_fn = lib.metrics.SampleMAE()
    
    mae_at = [1, 3, 6, 12, 18, 24]
    point_metrics = {'mae': torch_metrics.MaskedMAE(),
                     **{f'mae_{h:d}h': torch_metrics.MaskedMAE(at=h-1) for h in mae_at if h <= cfg.horizon},
                     'mse': torch_metrics.MaskedMSE(),
                     }
    
    sample_metrics = {'smae': lib.metrics.SampleMAE(),
                      **{f'smae_{h:d}h': lib.metrics.SampleMAE(at=h-1) for h in mae_at if h <= cfg.horizon},
                      'smse': lib.metrics.SampleMSE(),
                      'ens': lib.metrics.EnergyScore(),
                      **{f'ens_{h:d}h': lib.metrics.EnergyScore(at=h-1) for h in mae_at if h <= cfg.horizon},
                      }

    if cfg.get('lr_scheduler') is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler,
                                  cfg.lr_scheduler.name)
        scheduler_kwargs = dict(cfg.lr_scheduler.hparams)
    else:
        scheduler_class = scheduler_kwargs = None

    # Use regular Predictor for baseline models (deterministic), SamplingPredictor for learned models (probabilistic)
    # Note: is_baseline_or_icon is already computed above when setting loss_fn
    if is_baseline_or_icon and not isinstance(loss_fn, lib.metrics.SampleMetric):
        # Deterministic baseline models use regular Predictor with point metrics only
        predictor_class = Predictor
        log_metrics = point_metrics
        predictor_kwargs = {}
        monitored_metric = 'val_mae'
    else:
        # Learned models and ICON use SamplingPredictor for probabilistic evaluation
        predictor_class = SamplingPredictor
        assert not point_metrics.keys() & sample_metrics.keys()
        log_metrics = dict(**point_metrics, **sample_metrics)
        predictor_kwargs = dict(**cfg.get('sampling', {}))
        
        # Monitor validation metric based on loss type
        if isinstance(loss_fn, lib.metrics.SampleMetric):
            monitored_metric = 'val_smae'
        else:
            monitored_metric = 'val_mae'
    
    # setup predictor
    predictor = predictor_class(
        model_class=model_cls,
        model_kwargs=model_kwargs,
        optim_class=getattr(torch.optim, cfg.get('optimizer', {}).get('name', 'Adam')),
        optim_kwargs=dict(cfg.get('optimizer', {}).get('hparams', {'lr': 0.001})),
        loss_fn=loss_fn,
        metrics=log_metrics,
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
        scale_target=scale_target,
        **predictor_kwargs
    )

    ########################################
    # training                             #
    ########################################

    early_stop_callback = EarlyStopping(
        monitor=monitored_metric,
        patience=cfg.patience,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.run.dir,
        save_top_k=1,
        monitor=monitored_metric,
        mode='min',
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Will place the logs in ./mlruns
    mlflow_tracking_uri = cfg.get('mlflow_tracking_uri', './mlruns')
    exp_logger = MLFlowLogger(experiment_name=cfg.experiment_name, tracking_uri=mlflow_tracking_uri)
    
    # Create callback to flush MLflow in real-time
    mlflow_flush_callback = MLflowFlushCallback()

    # Print MLflow tracking URL (it's possible to use a remote or custom MLflow server)
    if mlflow_tracking_uri is None or mlflow_tracking_uri == './mlruns' or mlflow_tracking_uri.startswith('file://'):
        # Local file-based tracking
        abs_mlruns_path = os.path.abspath('./mlruns')
        mlflow_url = f"file://{abs_mlruns_path}"
        print(f"\n{'='*80}")
        print(f"MLflow Tracking:")
        print(f"  Tracking URI: {mlflow_url}")
        print(f"  Experiment: {cfg.experiment_name}")
        print(f"\n  To view results, run: mlflow ui --backend-store-uri {abs_mlruns_path}")
        print(f"  Then open: http://127.0.0.1:5000")
        print(f"{'='*80}\n")
    else:
        # Remote tracking server
        print(f"\n{'='*80}")
        print(f"MLflow Tracking:")
        print(f"  Tracking URI: {mlflow_tracking_uri}")
        print(f"  Experiment: {cfg.experiment_name}")
        print(f"  Access UI at: {mlflow_tracking_uri}")
        print(f"{'='*80}\n")

    trainer = Trainer(max_epochs=cfg.epochs,
                      default_root_dir=cfg.run.dir,
                      logger=exp_logger,
                      accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                      gradient_clip_val=cfg.grad_clip_val,
                      accumulate_grad_batches=cfg.get('accumulate_grad_batches', 1),
                      callbacks=[early_stop_callback, checkpoint_callback, lr_monitor, mlflow_flush_callback],
                      enable_progress_bar=True,  # Disable tqdm to keep logs concise
                      log_every_n_steps=50  # Log every 50 steps to see progress
                      )

    load_model_path = cfg.get('load_model_path')
    is_baseline = is_baseline_model(cfg.model.name)
    is_icon = is_icon_model(cfg.model.name)
    
    if is_baseline:
        # Baseline models don't require training
        print(f"\n{'='*80}")
        print(f"Baseline Model: {cfg.model.name}")
        print(f"  No training required - proceeding directly to evaluation")
        print(f"{'='*80}\n")
        result = dict()
    elif load_model_path is not None:
        print(f"Loading model from checkpoint: {load_model_path}")
        predictor.load_model(load_model_path)
        result = dict()
    else:
        print("\n" + "="*80)
        print("Starting training...")
        print("="*80)
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print("Calling trainer.fit()...")
        sys.stdout.flush()  # Force output flush
        
        trainer.fit(predictor,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader)
        best_checkpoint_path = checkpoint_callback.best_model_path
        predictor.load_model(best_checkpoint_path)
        
        # Print training summary
        print(f"\n{'='*80}")
        print(f"Training Complete!")
        print(f"  Best checkpoint: {best_checkpoint_path}")
        print(f"  Best validation metric ({monitored_metric}): {checkpoint_callback.best_model_score.item():.4f}")
        
        # Print MLflow run URL after training
        if hasattr(exp_logger, 'run_id') and exp_logger.run_id:
            if mlflow_tracking_uri is None or mlflow_tracking_uri == './mlruns' or mlflow_tracking_uri.startswith('file://'):
                abs_mlruns_path = os.path.abspath('./mlruns')
                print(f"\n  MLflow Run Details:")
                print(f"    Run ID: {exp_logger.run_id}")
                print(f"    Experiment: {cfg.experiment_name}")
                print(f"    View run: mlflow ui --backend-store-uri {abs_mlruns_path}")
                print(f"    Then navigate to: http://127.0.0.1:5000")
            else:
                print(f"\n  MLflow Run Details:")
                print(f"    Run ID: {exp_logger.run_id}")
                print(f"    Experiment: {cfg.experiment_name}")
                print(f"    View run at: {mlflow_tracking_uri}")
        print(f"{'='*80}\n")
        result = checkpoint_callback.best_model_score.item()

    predictor.freeze()

    ########################################
    # testing                              #
    ########################################
    
    # ICON NWP evaluation (for ICON model or when nwp_test_set is True)
    if use_icon and isinstance(predictor, SamplingPredictor):
        print(f"\n{'='*80}")
        print(f"ICON NWP Baseline Evaluation")
        print(f"{'='*80}\n")
        
        icon_data = models.ICONData(pw_dataset=dataset)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        icon_metrics = icon_data.test_set_eval(
            torch_dataset=torch_dataset,  # Use full dataset, not dm.testset
            metrics=sample_metrics,
            predictor=predictor,
            batch_size=cfg.batch_size,
            device=device
        )
        
        # Print ICON results
        print(f"\n{'='*80}")
        print(f"ICON NWP Test Results:")
        print(f"{'='*80}")
        for name, value in icon_metrics.compute().items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            print(f"  {name}: {value:.4f}")
            logger.info(f" - {name}: {value:.5f}")
        print(f"{'='*80}\n")
    
    # Standard test evaluation (skip for ICON model since it uses custom evaluation)
    if not is_icon:
        trainer.test(predictor, dataloaders=dm.test_dataloader())

    return result


if __name__ == '__main__':
    exp = Experiment(run_fn=run, config_path='../config/',
                     config_name='default')
    res = exp.run()
    logger.info(res)
