"""Temporal Convolutional Network for time series forecasting.

This model uses dilated causal convolutions for temporal modeling.
It does NOT use graph structure - each node is processed independently.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor


class TemporalBlock(nn.Module):
    """Residual block with dilated causal convolutions."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation

        # First conv layer
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size, padding=self.padding, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second conv layer
        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size, padding=self.padding, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        # Causal convolution: remove future padding
        out = self.conv1(x)[:, :, : -self.padding] if self.padding > 0 else self.conv1(x)
        out = self.dropout1(self.relu1(self.bn1(out)))

        out = self.conv2(out)[:, :, : -self.padding] if self.padding > 0 else self.conv2(out)
        out = self.dropout2(self.relu2(self.bn2(out)))

        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNModel(nn.Module):
    """Temporal Convolutional Network for spatiotemporal forecasting.

    This model processes each node independently using dilated causal convolutions.
    It does NOT use graph structure.

    Architecture:
    1. Linear projection of input features
    2. Stack of TemporalBlocks with exponentially increasing dilation
    3. MLP decoder for multi-step prediction

    Args:
        input_size: Number of input features per node.
        n_nodes: Number of nodes (not used for graph, just for shape).
        horizon: Number of prediction steps.
        exog_size: Size of exogenous features.
        output_size: Number of output features per node.
        hidden_size: Hidden dimension size.
        num_layers: Number of temporal blocks.
        kernel_size: Convolution kernel size.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_size: int,
        n_nodes: int,
        horizon: int,
        exog_size: int = 0,
        output_size: Optional[int] = None,
        hidden_size: int = 32,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.horizon = horizon
        self.output_size = output_size if output_size is not None else 1

        # Input projection
        self.input_proj = nn.Linear(input_size + exog_size, hidden_size)

        # Stack of temporal blocks with exponentially increasing dilation
        layers = [
            TemporalBlock(hidden_size, hidden_size, kernel_size, 2**i, dropout)
            for i in range(num_layers)
        ]
        self.tcn = nn.Sequential(*layers)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, horizon * self.output_size),
        )

    def forward(
        self,
        x: Tensor,
        u: Optional[Tensor] = None,
        edge_index=None,
        edge_weight: Optional[Tensor] = None,
        mc_samples: Optional[int] = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [batch, time, nodes, features].
            u: Exogenous features of shape [batch, time, nodes, exog_features].
            edge_index: Not used (no graph structure).
            edge_weight: Not used (no graph structure).
            mc_samples: Number of Monte Carlo samples for probabilistic output.

        Returns:
            Predictions of shape [batch, horizon, nodes, output_size] or
            [mc_samples, batch, horizon, nodes, output_size] if mc_samples is set.
        """
        # Concatenate exogenous features
        if u is not None:
            if u.shape[1] > x.shape[1]:
                u = u[:, : x.shape[1]]
            x = torch.cat([x, u], dim=-1)

        b, t, n, f = x.shape

        # Process each node independently
        x = rearrange(x, 'b t n f -> (b n) t f')
        x = self.input_proj(x)
        x = rearrange(x, 'bn t h -> bn h t')

        # Apply TCN
        x = self.tcn(x)

        # Take last timestep and project to output
        x = x[:, :, -1]
        x = self.output_proj(x)
        x = rearrange(x, '(b n) (t f) -> b t n f', b=b, n=n, t=self.horizon, f=self.output_size)

        # Generate samples if mc_samples is provided
        if mc_samples is not None:
            sigma = x.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
            noise_shape = (mc_samples, b, self.horizon, n, self.output_size)
            x = x.unsqueeze(0) + sigma * torch.randn(*noise_shape, device=x.device)

        return x
