"""RNN model with node embeddings for spatiotemporal forecasting.

This model uses a GRU-based recurrent neural network with learnable node 
embeddings. It processes temporal patterns but does not use graph structure
(equivalent to TimeThenGraphIsoModel with graph_layers=0).
"""
from __future__ import annotations

from typing import List, Literal, Union

from .time_then_graph_isotropic import TimeThenGraphIsoModel


class RNNModel(TimeThenGraphIsoModel):
    """RNN with node embeddings for spatiotemporal forecasting.
    
    This is a temporal-only model that uses:
    - Learnable node embeddings for station-specific representations
    - GRU-based RNN for temporal encoding
    - MLP decoder for multi-step prediction
    
    It does NOT use graph structure (graph_layers=0).
    
    Args:
        input_size: Number of input features per node.
        horizon: Number of prediction steps.
        n_nodes: Number of nodes in the graph.
        output_size: Number of output features per node.
        exog_size: Size of exogenous features.
        hidden_size: Hidden dimension size.
        emb_size: Node embedding dimension.
        add_embedding_before: Where to add embeddings ('encoding', 'decoding').
        time_layers: Number of RNN layers.
        activation: Activation function.
        noise_mode: Sampling noise mode ('lin', 'multi', 'add', 'none').
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
        add_embedding_before: Union[str, List[str]] = ('encoding', 'decoding'),
        time_layers: int = 2,
        activation: str = 'elu',
        noise_mode: Literal['lin', 'multi', 'add', 'none'] = 'lin',
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
            graph_layers=0,  # No graph processing
            root_weight=False,
            norm='none',
            add_backward=False,
            cached=False,
            activation=activation,
            noise_mode=noise_mode,
            time_skip_connect=False,
        )
