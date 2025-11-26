"""ICON NWP baseline models for evaluation.

ICON-CH1-EPS is a high-resolution ensemble numerical weather prediction model
from MeteoSwiss. These classes allow comparing learned models against NWP forecasts.
"""
import os
from torchmetrics import MetricCollection
from tqdm import tqdm
import xarray as xr
import pandas as pd
import torch
import numpy as np
from tsl.nn import models as tsl_models
import lib


class ICONData:
    """ICON ensemble data handler for temperature forecasting."""

    def __init__(self, pw_dataset, stations=None, root=None):
        self.stations = pw_dataset.nodes if stations is None else stations
        
        if pw_dataset.icon_data is None:
            raise ValueError(
                "No ICON data loaded. Set extended_nwp_vars=['temperature'] in dataset config."
            )
        
        if "temperature" not in pw_dataset.icon_data:
            raise ValueError(
                f"Temperature not in ICON data. Available: {list(pw_dataset.icon_data.keys())}"
            )
        
        self.ds_temp = pw_dataset.icon_data["temperature"]
        self.ds_temp.compute()
        
        if self.stations is not None:
            self.ds_temp = self.ds_temp.sel(nat_abbr=self.stations)
        
        self.reftimes = np.unique(self.ds_temp.reftime.values)

    def get_ensamble(self, reftime, horizon):
        """Get ICON ensemble predictions for given reference times.
        
        Args:
            reftime: Reference time(s) for the forecast
            horizon: Forecast horizon in hours
            
        Returns:
            Tensor of shape [n_samples, batch, horizon, nodes, features]
            where n_samples is the ensemble size (11 for ICON-CH1-EPS)
        """
        from einops import rearrange
        
        slice_temp = self.ds_temp.sel(
            reftime=reftime, 
            lead=slice(np.timedelta64(1, 'h'), np.timedelta64(horizon, 'h'))
        )
        
        # After slicing, xarray structure is:
        # - Data variable 'temperature' has dims: (reftime, lead, nat_abbr, realization)
        # - to_array() adds a 'variable' dimension at front
        y_hat = torch.from_numpy(slice_temp.to_array().values)
        
        # After to_array(), shape is: (variable, reftime, lead, nat_abbr, realization)
        # We need: (realization, reftime, lead, nat_abbr, variable)
        # Note: realization is the ensemble dimension (11 members)
        
        if y_hat.dim() == 5:
            # Shape: (variable, reftime, lead, nat_abbr, realization)
            # Rearrange to: (realization, reftime, lead, nat_abbr, variable)
            y_hat = rearrange(y_hat, "v b h n s -> s b h n v")
        elif y_hat.dim() == 4:
            # Single reftime case: (variable, lead, nat_abbr, realization)
            # Rearrange to: (realization, 1, lead, nat_abbr, variable)
            y_hat = rearrange(y_hat, "v h n s -> s 1 h n v")
        
        return y_hat
    
    def test_set_idx(self, torch_dataset):
        """Get indices of test set samples that have ICON data available."""
        assert torch_dataset.delay == 0

        rt_dti = pd.DatetimeIndex(self.reftimes, tz="UTC")
        ds_dti = pd.DatetimeIndex(torch_dataset.data_timestamps()["window"][:, -1])
        intersect = ds_dti.isin(rt_dti)
        test_ds_indices = intersect.nonzero()[0]
        test_dt = ds_dti[intersect]

        return test_ds_indices, test_dt
    
    def test_set_eval(self, torch_dataset, metrics, predictor=None, batch_size=32, device="cpu"):
        """Evaluate on test set using ICON predictions or model predictions.
        
        Args:
            torch_dataset: The full SpatioTemporalDataset (not Subset)
            metrics: Dictionary of metrics to compute
            predictor: The predictor (model wrapper)
            batch_size: Batch size for evaluation
            device: Device to use for computation
            
        Returns:
            MetricCollection with computed metrics
        """
        test_ds_idx, icon_reftimes = self.test_set_idx(torch_dataset)
        
        if len(test_ds_idx) == 0:
            raise ValueError("No overlapping samples between test set and ICON data")
        
        print(f"Evaluating on {len(test_ds_idx)} samples with ICON data available")
        
        metrics = MetricCollection(
            {n: m for n, m in metrics.items() if isinstance(m, lib.metrics.SampleMetric)},
            prefix="nwp/test_")
        metrics.to(device)
        metrics.reset()

        predictor.in_testing_step = True  # sets the correct number of test samples
        predictor.to(device)

        test_ds_idx = np.array(test_ds_idx)
        icon_reftimes = icon_reftimes.tz_localize(None)
        
        for i in tqdm(range(0, len(test_ds_idx), batch_size), desc='Eval on NWP test set'):
            batch_idx = test_ds_idx[i: i + batch_size]
            batch_reftimes = icon_reftimes[i: i + batch_size]
            
            data = torch_dataset[batch_idx]
            data.to(device)
            
            if isinstance(predictor.model, ICONDummyModel):
                y_hat = self.get_ensamble(
                    reftime=batch_reftimes, 
                    horizon=torch_dataset.horizon
                )
                y_hat = y_hat.to(device).float()
            else:
                y_hat = predictor.predict_batch(data, preprocess=False, postprocess=True)

            metrics.update(y=data.y, y_hat=y_hat, mask=data.mask)

        predictor.in_testing_step = False
        return metrics


class ICONDummyModel(tsl_models.BaseModel):
    """Dummy model placeholder for ICON evaluation.
    
    This model doesn't actually compute anything - predictions come from
    the ICON ensemble data loaded separately.
    """
    
    def __init__(self, **kwargs):
        super().__init__()
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "ICONDummyModel doesn't compute predictions. "
            "Use ICONData.get_ensamble() instead."
        )
