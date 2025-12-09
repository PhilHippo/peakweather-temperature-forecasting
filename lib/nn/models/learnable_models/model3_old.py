"""Model3: Attention-based STGNN for Traffic Speed Prediction.

Implementation based on the Traffic Speed Prediction Stage from the paper:
"Attention-Based Spatial-Temporal Graph Neural Network With Long-Term Dependencies
for Traffic Speed Prediction" (IEEE T-ITS, 2025)

Key components:
- Graph Structure Learning with Gumbel-Sigmoid for differentiable binary edges
- STAWnet backbone (Gated TCN + Dynamic Attention Network)
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tsl.nn.models import BaseModel
from lib.nn.layers.sampling_readout import SamplingReadoutLayer


class _CausalConv1d(nn.Module):
    """1D causal convolution with proper padding."""

    def __init__(self, channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.pad > 0:
            x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class GatedTCN(nn.Module):
    """Gated Temporal Convolutional Network block.

    H_out = tanh(W_f * H_in) ⊙ sigmoid(W_g * H_in)
    """

    def __init__(self, hidden_dim: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.filter_conv = _CausalConv1d(hidden_dim, kernel_size, dilation)
        self.gate_conv = _CausalConv1d(hidden_dim, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input [batch, time, nodes, channels]

        Returns:
            Output [batch, time, nodes, channels]
        """
        b, t, n, c = x.shape
        # Reshape for 1D conv: [batch * nodes, channels, time]
        x_flat = x.view(b * n, t, c).transpose(1, 2)

        # Gated activation
        h_filter = torch.tanh(self.filter_conv(x_flat))
        h_gate = torch.sigmoid(self.gate_conv(x_flat))
        out = h_filter * h_gate

        # Reshape back: [batch, time, nodes, channels]
        out = out.transpose(1, 2).contiguous().view(b, t, n, c)
        return self.dropout(out)


class DynamicAttentionNetwork(nn.Module):
    """Dynamic Attention Network for spatial aggregation.

    Computes attention scores between nodes incorporating learned adjacency.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_state: Tensor, adjacency: Optional[Tensor]) -> Tensor:
        """
        Args:
            node_state: Node representations [batch, nodes, hidden_dim]
            adjacency: Learned adjacency [batch, nodes, nodes]

        Returns:
            Updated node states [batch, nodes, hidden_dim]
        """
        # Handle NaN in input
        node_state = torch.where(torch.isnan(node_state), torch.zeros_like(node_state), node_state)

        b, n, _ = node_state.shape

        # Multi-head projections
        q = self.q_proj(node_state).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(node_state).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(node_state).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        # Incorporate adjacency as attention bias
        if adjacency is not None:
            adj_clamped = adjacency.clamp(min=1e-4, max=1.0)
            log_adj = torch.log(adj_clamped)
            # Clamp log values to prevent extreme negative values
            log_adj = torch.clamp(log_adj, min=-10.0)
            scores = scores + log_adj.unsqueeze(1)

        # Clamp scores before softmax to prevent overflow/underflow
        scores = torch.clamp(scores, min=-50.0, max=50.0)
        attn = torch.softmax(scores, dim=-1)

        # Replace any NaN in attention
        nan_mask = torch.isnan(attn)
        if nan_mask.any():
            uniform_val = 1.0 / n
            attn = torch.where(nan_mask, torch.full_like(attn, uniform_val), attn)

        attn = self.dropout(attn)

        # Apply attention
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, self.hidden_dim)
        return self.out_proj(out)


class STBlock(nn.Module):
    """Spatial-Temporal block combining Gated TCN and Dynamic Attention."""

    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int,
        dilation: int,
        attn_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.tcn = GatedTCN(hidden_dim, kernel_size, dilation, dropout)
        self.dan = DynamicAttentionNetwork(hidden_dim, attn_heads, dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, adjacency: Optional[Tensor]) -> Tensor:
        """
        Args:
            x: Input [batch, time, nodes, hidden_dim]
            adjacency: Adjacency matrix [batch, nodes, nodes]

        Returns:
            Output [batch, time, nodes, hidden_dim]
        """
        # Temporal processing with residual
        h = self.tcn(x) + x

        # Spatial processing on last time step
        node_state = h[:, -1, :, :].clone()
        attended = self.dan(node_state, adjacency)

        # Update last time step with spatial info
        h = h.clone()
        h[:, -1, :, :] = self.norm(node_state + attended)

        return h


class STAWnet(nn.Module):
    """Spatial-Temporal Attention Wavenet.

    Backbone network combining stacked Gated TCN blocks with Dynamic Attention
    for effective spatial-temporal dependency modeling.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int,
        dilations: Sequence[int],
        attn_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # ST-Blocks with skip connections
        self.blocks = nn.ModuleList([
            STBlock(hidden_dim, kernel_size, dilation, attn_heads, dropout)
            for dilation in dilations
        ])

        # Skip connection projections
        self.skip_projs = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, 1)
            for _ in dilations
        ])

        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, adjacency: Optional[Tensor]) -> Tensor:
        """
        Args:
            x: Input [batch, time, nodes, features]
            adjacency: Adjacency matrix [batch, nodes, nodes]

        Returns:
            Node representations [batch, nodes, hidden_dim]
        """
        h = self.input_proj(x)
        skip_sum = None

        for block, skip_proj in zip(self.blocks, self.skip_projs):
            h = block(h, adjacency)
            # Skip connection from last time step
            skip = h[:, -1, :, :]  # [batch, nodes, hidden]
            skip_t = skip.transpose(1, 2)  # [batch, hidden, nodes]
            skip_out = skip_proj(skip_t).transpose(1, 2)  # [batch, nodes, hidden]

            if skip_sum is None:
                skip_sum = skip_out
            else:
                skip_sum = skip_sum + skip_out

        # Combine skip connections
        out = self.output_norm(skip_sum)
        return out


class GumbelSigmoidGraphLearner(nn.Module):
    """Graph structure learner using Gumbel-Sigmoid for differentiable sampling.

    Learns a binary adjacency matrix through a differentiable relaxation,
    with KNN regularization for sparsity control.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_nodes: int,
        knn: int = 10,
        tau: float = 1.0,
        hard: bool = False,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.knn = knn
        self.tau = tau
        self.hard = hard

        # Learnable graph parameters
        self.theta = nn.Parameter(torch.zeros(n_nodes, n_nodes))
        nn.init.xavier_uniform_(self.theta)

        # Feature extractors for KNN computation
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)

    def compute_knn_graph(self, node_repr: Tensor) -> Tensor:
        """Compute KNN graph from node representations.

        Args:
            node_repr: Node representations [batch, nodes, hidden_dim]

        Returns:
            Binary KNN adjacency matrix [batch, nodes, nodes]
        """
        q = self.q_proj(node_repr)
        k = self.k_proj(node_repr)

        # Compute similarity scores
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))
        # Clamp scores to prevent extreme values
        scores = torch.clamp(scores, min=-50.0, max=50.0)

        # Get top-k neighbors
        _, topk_indices = torch.topk(scores, self.knn, dim=-1)

        # Create binary adjacency
        adj = torch.zeros_like(scores)
        adj.scatter_(-1, topk_indices, 1.0)

        return adj

    def forward(self, node_repr: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass for graph learning.

        Args:
            node_repr: Node representations [batch, nodes, hidden_dim]

        Returns:
            adj: Learned adjacency matrix (soft or hard binary edges)
            knn_adj: KNN adjacency for regularization loss
        """
        # Handle NaN in input
        node_repr = torch.where(torch.isnan(node_repr), torch.zeros_like(node_repr), node_repr)

        batch_size = node_repr.size(0)

        # Compute KNN graph for regularization
        knn_adj = self.compute_knn_graph(node_repr)

        # Gumbel-Sigmoid for differentiable binary edge sampling
        # (Each edge is sampled independently, not normalized across neighbors)
        # Expand theta for batch processing
        theta = self.theta.unsqueeze(0).expand(batch_size, -1, -1)

        # Sample from Gumbel distribution for binary edge decisions
        if self.training:
            # Gumbel noise for binary sampling (logistic distribution difference)
            u = torch.rand_like(theta).clamp(min=1e-10, max=1 - 1e-10)
            gumbel_noise = torch.log(u) - torch.log(1 - u)
            logits = (theta + gumbel_noise) / max(self.tau, 1e-6)
        else:
            logits = theta / max(self.tau, 1e-6)

        # Sigmoid to get edge probabilities (binary, not normalized)
        adj = torch.sigmoid(logits)

        if self.hard:
            # Straight-through estimator for hard binary sampling
            adj_hard = (adj > 0.5).float()
            adj = adj_hard - adj.detach() + adj

        return adj, knn_adj


class Model3Old(BaseModel):
    """Model3: Attention-based STGNN for Traffic Speed Prediction.

    Traffic Speed Prediction Stage architecture based on the paper:
    "Attention-Based Spatial-Temporal Graph Neural Network With Long-Term
    Dependencies for Traffic Speed Prediction" (IEEE T-ITS, 2025)

    Architecture:
        - Graph Structure Learning with Gumbel-Sigmoid for differentiable binary edges
        - STAWnet backbone (Gated TCN + Dynamic Attention Network)
        - Probabilistic output via Gaussian sampling layer

    The model learns spatial dependencies dynamically through attention-based
    graph structure learning, while temporal dependencies are captured through
    gated temporal convolutions with multiple dilation rates.
    """

    def __init__(
        self,
        input_size: int,
        horizon: int,
        n_nodes: Optional[int] = None,
        output_size: Optional[int] = None,
        exog_size: int = 0,
        hidden_dim: int = 128,
        staw_kernel_size: int = 3,
        staw_dilations: Sequence[int] = (1, 2, 4),
        staw_attn_heads: int = 4,
        graph_knn: int = 10,
        graph_tau: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size or input_size
        self.horizon = horizon
        self.n_nodes = n_nodes
        self.exog_size = exog_size
        self.hidden_dim = hidden_dim

        # Feature projection
        total_input = input_size + exog_size
        self.feature_proj = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.GELU(),
        )

        # Graph Structure Learning
        if n_nodes is not None:
            self.graph_learner = GumbelSigmoidGraphLearner(
                hidden_dim=hidden_dim,
                n_nodes=n_nodes,
                knn=graph_knn,
                tau=graph_tau,
            )
        else:
            self.graph_learner = None

        # STAWnet backbone
        self.stawnet = STAWnet(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            kernel_size=staw_kernel_size,
            dilations=staw_dilations,
            attn_heads=staw_attn_heads,
            dropout=dropout,
        )

        # Readout
        readout_hidden = (hidden_dim + self.output_size) // 2

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon * readout_hidden),
        )
        self.readout_hidden = readout_hidden

        # Probabilistic output layer
        self.sample_decoder = SamplingReadoutLayer(
            input_size=readout_hidden,
            output_size=self.output_size,
            n_nodes=n_nodes,
            horizon=horizon,
            noise_mode='lin',
            pre_activation='elu',
        )

        # For storing graph regularization loss
        self.graph_reg_loss = None

    @staticmethod
    def _merge_features(x: Tensor, u: Optional[Tensor]) -> Tensor:
        """Merge input with exogenous features."""
        # Replace NaN values with 0 in input
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        if u is None:
            return x
        # Replace NaN in exogenous features as well
        u = torch.where(torch.isnan(u), torch.zeros_like(u), u)

        if u.dim() == 3:
            u = u.unsqueeze(2).expand(-1, -1, x.size(2), -1)
        elif u.dim() == 4 and u.size(2) == 1:
            u = u.expand(-1, -1, x.size(2), -1)
        return torch.cat([x, u], dim=-1)

    def compute_graph_reg_loss(self, adj: Tensor, knn_adj: Tensor) -> Tensor:
        """Compute graph regularization loss (cross-entropy with KNN graph).

        Uses binary cross-entropy to encourage the learned adjacency to match
        the KNN graph structure, providing supervised signal for graph learning.
        """
        return F.binary_cross_entropy(adj, knn_adj, reduction='mean')

    def forward(
        self,
        x: Tensor,
        edge_index,
        edge_weight: Optional[Tensor] = None,
        u: Optional[Tensor] = None,
        v: Optional[Tensor] = None,
        node_idx: Optional[Tensor] = None,
        mc_samples: Optional[int] = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor [batch, time, nodes, features]
            edge_index: Edge indices (not used, graph is learned internally)
            edge_weight: Edge weights (not used)
            u: Exogenous features
            v: Additional exogenous features (not used)
            node_idx: Node indices for subset selection
            mc_samples: Number of Monte Carlo samples for probabilistic output

        Returns:
            Predictions [batch, horizon, nodes, output_size] or
            [mc_samples, batch, horizon, nodes, output_size] if mc_samples is set
        """
        del edge_index, edge_weight, v

        # Merge features and project
        x = self._merge_features(x, u)
        x = self.feature_proj(x)
        b, _, n, _ = x.shape

        # Graph Structure Learning
        # Use mean-pooled temporal features for graph learning
        node_repr = x.mean(dim=1)  # [batch, nodes, hidden]

        if self.graph_learner is not None:
            adjacency, knn_adj = self.graph_learner(node_repr)
            # Store regularization loss for training
            if self.training:
                self.graph_reg_loss = self.compute_graph_reg_loss(adjacency, knn_adj)
        else:
            adjacency = None

        # STAWnet for spatial-temporal processing
        staw_feat = self.stawnet(x, adjacency)  # [batch, nodes, hidden]

        if node_idx is not None:
            staw_feat = staw_feat[:, node_idx, :]
            n = staw_feat.size(1)

        # Readout
        out = self.readout(staw_feat.view(-1, staw_feat.size(-1)))
        out = out.view(b, n, self.horizon, self.readout_hidden)
        out = out.permute(0, 2, 1, 3)  # [batch, horizon, nodes, readout_hidden]

        # Probabilistic output
        out = self.sample_decoder(out, mc_samples=mc_samples)

        return out
