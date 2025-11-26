"""Simple non-trainable baseline models for time series forecasting.

These baselines are useful for benchmarking learnable models:
- NaiveModel: Last value repeated (persistence)
- MovingAverageModel: Average of last n observations
"""
from __future__ import annotations

from torch import Tensor
from tsl.nn.models import BaseModel


class NaiveModel(BaseModel):
    """Naive forecast: repeats the last observed value for all horizon steps.
    
    Also known as persistence forecast or random walk forecast.
    
    Args:
        horizon: Number of prediction steps.
    """
    
    def __init__(self, horizon: int, **kwargs):
        super().__init__()
        self.horizon = horizon
    
    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """
        Args:
            x: Input tensor of shape [batch, time, nodes, features].
        
        Returns:
            Predictions of shape [batch, horizon, nodes, features].
        """
        # Take the last timestep and repeat for entire horizon
        last_value = x[:, -1:, :, :]  # [batch, 1, nodes, features]
        return last_value.expand(-1, self.horizon, -1, -1)


class MovingAverageModel(BaseModel):
    """Moving average forecast: predicts the average of the last n observations.
    
    This baseline captures the recent trend by averaging recent values.
    The same averaged value is used for all horizon steps.
    
    Args:
        horizon: Number of prediction steps.
        window_size: Number of past observations to average (default: 24 for daily).
    """
    
    def __init__(self, horizon: int, window_size: int = 24, **kwargs):
        super().__init__()
        self.horizon = horizon
        self.window_size = window_size
    
    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """
        Args:
            x: Input tensor of shape [batch, time, nodes, features].
        
        Returns:
            Predictions of shape [batch, horizon, nodes, features].
        """
        t = x.size(1)
        
        # Use minimum of window_size and available timesteps
        actual_window = min(self.window_size, t)
        
        # Compute mean of last `actual_window` timesteps
        mean_value = x[:, -actual_window:, :, :].mean(dim=1, keepdim=True)
        return mean_value.expand(-1, self.horizon, -1, -1)
