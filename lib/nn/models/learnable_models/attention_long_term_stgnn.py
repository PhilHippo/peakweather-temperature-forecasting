from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tsl.nn.models import BaseModel


class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm variant used in the reference architecture."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        norm = torch.mean(x.pow(2), dim=-1, keepdim=True)
        return self.scale * x * torch.rsqrt(norm + self.eps)


class RMSNormTransformerLayer(nn.Module):
    """Transformer encoder block where LayerNorm is replaced by RMSNorm."""

    def __init__(self, dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + self.dropout(attn_out)
        x = x + self.ff(self.norm2(x))
        return x


class LongTermFeatureEncoder(nn.Module):
    """Patchified Transformer encoder that captures long-range patterns."""

    def __init__(
        self,
        hidden_dim: int,
        patch_len: int,
        max_patches: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.patch_len = patch_len
        self.max_patches = max_patches
        self.patch_embed = nn.Linear(patch_len * hidden_dim, hidden_dim)
        self.positional = nn.Parameter(torch.randn(1, max_patches, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            RMSNormTransformerLayer(hidden_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, int]:
        b, t, n, d = x.shape
        patch_len = self.patch_len
        num_patches = min(math.ceil(t / patch_len), self.max_patches)
        target = num_patches * patch_len
        if t < target:
            pad = torch.zeros(b, target - t, n, d, device=x.device, dtype=x.dtype)
            x = torch.cat([pad, x], dim=1)
        elif t > target:
            x = x[:, -target:, :, :]
        x = x.view(b, num_patches, patch_len, n, d)
        x = x.permute(0, 3, 1, 2, 4).contiguous()
        patches = x.view(b * n, num_patches, patch_len * d)
        tokens = self.patch_embed(patches)
        tokens = tokens + self.positional[:, :num_patches, :]
        tokens = self.dropout(tokens)
        for layer in self.layers:
            tokens = layer(tokens)
        tokens = tokens.view(b, n, num_patches, self.hidden_dim)
        return tokens, num_patches


class GraphStructureLearner(nn.Module):
    """Differentiable KNN-style learner that infers dependencies among nodes."""

    def __init__(self, hidden_dim: int, knn: Optional[int] = 8, tau: float = 1.0, symmetric: bool = True):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.knn = knn
        self.tau = tau
        self.symmetric = symmetric

    def forward(self, node_repr: Tensor) -> Tensor:
        q = self.q_proj(node_repr)
        k = self.k_proj(node_repr)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))
        attention = torch.softmax(scores / max(self.tau, 1e-6), dim=-1)
        if self.knn is not None and self.knn < attention.size(-1):
            topk = torch.topk(attention, self.knn, dim=-1).indices
            mask = torch.zeros_like(attention)
            mask.scatter_(-1, topk, 1.0)
            attention = attention * mask
            attention = attention / (attention.sum(dim=-1, keepdim=True) + 1e-6)
        if self.symmetric:
            attention = 0.5 * (attention + attention.transpose(-1, -2))
        return attention


class _CausalConv1d(nn.Module):
    """1D convolution with manual left padding to preserve causality."""

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


class GatedTemporalConv(nn.Module):
    """Temporal gated TCN block operating independently on each node."""

    def __init__(self, hidden_dim: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.filter_conv = _CausalConv1d(hidden_dim, kernel_size, dilation)
        self.gate_conv = _CausalConv1d(hidden_dim, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        b, t, n, c = x.shape
        x_flat = x.view(b * n, t, c).transpose(1, 2)
        h_filter = torch.tanh(self.filter_conv(x_flat))
        h_gate = torch.sigmoid(self.gate_conv(x_flat))
        out = h_filter * h_gate
        out = out.transpose(1, 2).contiguous().view(b, t, n, c)
        return self.dropout(out)


class DynamicNodeAttention(nn.Module):
    """Attention mechanism that incorporates the learned adjacency as a bias."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_state: Tensor, adjacency: Optional[Tensor]) -> Tensor:
        b, n, _ = node_state.shape
        q = self.q_proj(node_state).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(node_state).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(node_state).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if adjacency is not None:
            log_adj = torch.log(adjacency + 1e-6)
            scores = scores + log_adj.unsqueeze(1)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, self.hidden_dim)
        return self.out_proj(out)


class STAWnet(nn.Module):
    """Spatial-Temporal Attention Wavenet backbone."""

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
        self.blocks = nn.ModuleList(
            GatedTemporalConv(hidden_dim, kernel_size, dilation, dropout) for dilation in dilations
        )
        self.node_attention = DynamicNodeAttention(hidden_dim, attn_heads, dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, adjacency: Optional[Tensor]) -> Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h) + h
        node_state = h[:, -1, :, :]
        attended = self.node_attention(node_state, adjacency)
        return self.norm(node_state + attended)


class AttentionLongTermSTGNN(BaseModel):
    """Attention-based STGNN with long-term Transformer encoder and STAWnet head.

    This model follows the design described in
    \"Attention-Based Spatial-Temporal Graph Neural Network With Long-Term
    Dependencies for Traffic Speed Prediction\" (IEEE T-ITS, 2025) and adapts it
    to PeakWeather. It consumes the enriched meteorological feature set
    (ew_wind, humidity, nw_wind, precipitation, pressure, sunshine, temperature,
    wind_gust) plus the static topo-geomorphological attributes merged in ``u``.
    """

    def __init__(
        self,
        input_size: int,
        horizon: int,
        n_nodes: Optional[int] = None,
        output_size: Optional[int] = None,
        exog_size: int = 0,
        hidden_dim: int = 128,
        patch_len: int = 12,
        max_patches: int = 32,
        long_layers: int = 4,
        long_heads: int = 4,
        long_ff_size: int = 256,
        staw_kernel_size: int = 3,
        staw_dilations: Sequence[int] = (1, 2, 4),
        staw_attn_heads: int = 4,
        graph_knn: Optional[int] = 8,
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

        self.feature_proj = nn.Sequential(nn.LazyLinear(hidden_dim), nn.GELU())

        self.long_term_encoder = LongTermFeatureEncoder(
            hidden_dim=hidden_dim,
            patch_len=patch_len,
            max_patches=max_patches,
            num_layers=long_layers,
            num_heads=long_heads,
            ff_dim=long_ff_size,
            dropout=dropout,
        )
        self.graph_learner = GraphStructureLearner(
            hidden_dim=hidden_dim,
            knn=graph_knn,
            tau=graph_tau,
            symmetric=True,
        )
        self.stawnet = STAWnet(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            kernel_size=staw_kernel_size,
            dilations=staw_dilations,
            attn_heads=staw_attn_heads,
            dropout=dropout,
        )

        fusion_dim = hidden_dim * 2
        self.readout = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, horizon * self.output_size),
        )

    @staticmethod
    def _merge_features(x: Tensor, u: Optional[Tensor]) -> Tensor:
        if u is None:
            return x
        if u.dim() == 3:
            u = u.unsqueeze(2).expand(-1, -1, x.size(2), -1)
        elif u.dim() == 4 and u.size(2) == 1:
            u = u.expand(-1, -1, x.size(2), -1)
        return torch.cat([x, u], dim=-1)

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
        del edge_index, edge_weight, v, mc_samples
        x = self._merge_features(x, u)
        x = self.feature_proj(x)
        b, _, n, _ = x.shape

        tokens, _ = self.long_term_encoder(x)
        long_feat = tokens[:, :, -1, :]
        adjacency = self.graph_learner(long_feat)
        staw_feat = self.stawnet(x, adjacency)

        features = torch.cat([long_feat, staw_feat], dim=-1)
        if node_idx is not None:
            features = features[:, node_idx, :]
            n = features.size(1)

        out = self.readout(features.view(-1, features.size(-1)))
        out = out.view(b, n, self.horizon, self.output_size)
        out = out.permute(0, 2, 1, 3)
        return out

