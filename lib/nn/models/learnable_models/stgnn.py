"""Spatiotemporal Graph Neural Network for forecasting.

This model uses a GRU-based RNN for temporal encoding followed by 
graph convolutions for spatial message passing.
"""
from __future__ import annotations

from typing import List, Literal, Union

from .time_then_graph_isotropic import TimeThenGraphIsoModel


class STGNN(TimeThenGraphIsoModel):
    """Spatiotemporal Graph Neural Network.
    
    Architecture: Time-Then-Space
    1. Temporal encoding with GRU-based RNN
    2. Spatial encoding with graph convolutions (DiffConv or GraphConv)
    3. MLP decoder for multi-step prediction
    
    Uses learnable node embeddings for station-specific representations.
    
    Args:
        input_size: Number of input features per node.
        horizon: Number of prediction steps.
        n_nodes: Number of nodes in the graph.
        output_size: Number of output features per node.
        exog_size: Size of exogenous features.
        hidden_size: Hidden dimension size.
        emb_size: Node embedding dimension.
        add_embedding_before: Where to add embeddings ('encoding', 'message_passing', 'decoding').
        time_layers: Number of RNN layers.
        graph_layers: Number of graph convolution layers.
        norm: Graph normalization ('none', 'sym', 'asym').
        activation: Activation function.
        noise_mode: Sampling noise mode ('lin', 'multi', 'add', 'none').
        time_skip_connect: Whether to add skip connection from temporal to output.
    """

    def __init__(
        self,
        input_size: int,
        horizon: int,
        n_nodes: int = None,
        output_size: int = None,
        exog_size: int = 0,
        hidden_size: int = 64,
        emb_size: int = 32,
        add_embedding_before: Union[str, List[str]] = 'encoding',
        time_layers: int = 2,
        graph_layers: int = 2,
        norm: str = 'none',
        activation: str = 'elu',
        noise_mode: Literal['lin', 'multi', 'add', 'none'] = 'lin',
        time_skip_connect: bool = True,
    ):
        super().__init__(
            input_size=input_size,
            horizon=horizon,
            n_nodes=n_nodes,
            output_size=output_size,
            exog_size=exog_size,
            hidden_size=hidden_size,
            emb_size=emb_size,
            add_embedding_before=add_embedding_before,
            use_local_weights=None,
            time_layers=time_layers,
            graph_layers=graph_layers,
            root_weight=True,
            norm=norm,
            add_backward=False,
            cached=False,
            activation=activation,
            noise_mode=noise_mode,
            time_skip_connect=time_skip_connect,
        )
