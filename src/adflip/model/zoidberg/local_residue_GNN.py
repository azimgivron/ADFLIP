from __future__ import annotations, print_function

import itertools
import math
import sys
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionWiseFeedForward(torch.nn.Module):
    """Implement the position wise feed forward component."""

    def __init__(self, num_hidden: int, num_ff: int) -> None:
        """Initialize the PositionWiseFeedForward.

        Args:
            num_hidden: Number of hidden.
            num_ff: Number of ff.
        """
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = torch.nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = torch.nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.

        Args:
            h_V: H v value.

        Returns:
            Computed tensor values.
        """
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h


class PositionalEncodings(torch.nn.Module):
    """Implement the positional encodings component."""

    def __init__(self, num_embeddings: int, max_relative_feature: int = 32) -> None:
        """Initialize the PositionalEncodings.

        Args:
            num_embeddings: Number of embeddings.
            max_relative_feature: Max relative feature value.
        """
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = torch.nn.Linear(2 * max_relative_feature + 1 + 1, num_embeddings)

    def forward(self, offset: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.

        Args:
            offset: Offset value.
            mask: Boolean mask for mask.

        Returns:
            Computed tensor values.
        """
        d = torch.clip(
            offset + self.max_relative_feature, 0, 2 * self.max_relative_feature
        ) * mask + (1 - mask) * (2 * self.max_relative_feature + 1)
        d_onehot = torch.nn.functional.one_hot(d, 2 * self.max_relative_feature + 1 + 1)
        E = self.linear(d_onehot.float())
        return E


class EncLayer(torch.nn.Module):
    """Implement the enc layer component."""

    def __init__(
        self,
        num_hidden: int,
        num_in: int,
        dropout: float = 0.1,
        num_heads: Optional[int] = None,
        scale: int = 30,
        time_embedder: Optional[nn.Module] = None,
    ) -> None:
        """Initialize the EncLayer.

        Args:
            num_hidden: Number of hidden.
            num_in: Number of in.
            dropout: Dropout value.
            num_heads: Number of heads.
            scale: Scale value.
            time_embedder: Time embedder value.
        """
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)
        self.norm1 = torch.nn.LayerNorm(num_hidden)
        self.norm2 = torch.nn.LayerNorm(num_hidden)
        self.norm3 = torch.nn.LayerNorm(num_hidden)

        self.W1 = torch.nn.Linear(num_hidden + 2 * num_in, num_hidden, bias=True)
        self.W2 = torch.nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = torch.nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = torch.nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = torch.nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = torch.nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)
        if time_embedder is not None:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(num_hidden, 2 * num_hidden, bias=True)
            )

    def forward(
        self,
        h_V: torch.Tensor,
        h_V_atom: torch.Tensor,
        h_E: torch.Tensor,
        E_idx: torch.Tensor,
        mask_V: Optional[torch.Tensor] = None,
        mask_attend: Optional[torch.Tensor] = None,
        time: Optional[torch.Tensor] = None,
    ) -> Tuple[Any, ...]:
        """Parallel computation of full transformer layer

        Args:
            h_V: H v value.
            h_V_atom: H v atom value.
            h_E: H e value.
            E_idx: E idx value.
            mask_V: Boolean mask for mask V.
            mask_attend: Boolean mask for mask attend.
            time: Time value.

        Returns:
            Computed result values.
        """

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_EV_atom = cat_neighbors_nodes(h_V_atom.clone(), h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV, h_EV_atom], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))

        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        if time is not None:
            scale, shift = self.adaLN_modulation(time).chunk(2, dim=-1)
            dh = modulate(dh, shift, scale)
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        # h_EV_atom = cat_neighbors_nodes(h_V_atom, h_E, E_idx)

        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E


# The following gather functions
def gather_edges(edges: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    """Execute the gather edges operation.

    Args:
        edges: Edges value.
        neighbor_idx: Neighbor idx value.

    Returns:
        Result of the gather edges operation.
    """
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features


def gather_nodes(nodes: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    """Execute the gather nodes operation.

    Args:
        nodes: Nodes value.
        neighbor_idx: Neighbor idx value.

    Returns:
        Result of the gather nodes operation.
    """
    neighbors_flat = neighbor_idx.reshape((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features


def gather_nodes_t(nodes: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    """Execute the gather nodes t operation.

    Args:
        nodes: Nodes value.
        neighbor_idx: Neighbor idx value.

    Returns:
        Result of the gather nodes t operation.
    """
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features


def cat_neighbors_nodes(
    h_nodes: torch.Tensor, h_neighbors: torch.Tensor, E_idx: torch.Tensor
) -> torch.Tensor:
    """Execute the cat neighbors nodes operation.

    Args:
        h_nodes: H nodes value.
        h_neighbors: H neighbors value.
        E_idx: E idx value.

    Returns:
        Result of the cat neighbors nodes operation.
    """
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: float) -> torch.Tensor:
    """Execute the modulate operation.

    Args:
        x: Input tensor.
        shift: Shift value.
        scale: Scale value.

    Returns:
        Computed tensor values.
    """
    return x * (1 + scale) + shift
