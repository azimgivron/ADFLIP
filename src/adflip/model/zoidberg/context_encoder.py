from __future__ import annotations, print_function

import itertools
import math
import sys
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class DecLayer(torch.nn.Module):
    """Implement the dec layer component."""

    def __init__(
        self,
        num_hidden: int,
        num_in: int,
        dropout: float = 0.1,
        num_heads: Optional[int] = None,
        scale: int = 30,
        time_embedder: Optional[nn.Module] = None,
    ) -> None:
        """Initialize the DecLayer.

        Args:
            num_hidden: Number of hidden.
            num_in: Number of in.
            dropout: Dropout value.
            num_heads: Number of heads.
            scale: Scale value.
            time_embedder: Time embedder value.
        """
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.norm1 = torch.nn.LayerNorm(num_hidden)
        self.norm2 = torch.nn.LayerNorm(num_hidden)

        self.W1 = torch.nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = torch.nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = torch.nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)
        if time_embedder is not None:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(num_hidden, 2 * num_hidden, bias=True)
            )

    def forward(
        self,
        h_V: torch.Tensor,
        h_E: torch.Tensor,
        mask_V: Optional[torch.Tensor] = None,
        mask_attend: Optional[torch.Tensor] = None,
        time: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Parallel computation of full transformer layer

        Args:
            h_V: H v value.
            h_E: H e value.
            mask_V: Boolean mask for mask V.
            mask_attend: Boolean mask for mask attend.
            time: Time value.

        Returns:
            Computed tensor values.
        """

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if time is not None:
            scale, shift = self.adaLN_modulation(time).chunk(2, dim=-1)
            h_message = modulate(h_message, shift, scale)

        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V


class DecLayerJ(torch.nn.Module):
    """Implement the dec layer j component."""

    def __init__(
        self,
        num_hidden: int,
        num_in: int,
        dropout: float = 0.1,
        num_heads: Optional[int] = None,
        scale: int = 30,
    ) -> None:
        """Initialize the DecLayerJ.

        Args:
            num_hidden: Number of hidden.
            num_in: Number of in.
            dropout: Dropout value.
            num_heads: Number of heads.
            scale: Scale value.
        """
        super(DecLayerJ, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.norm1 = torch.nn.LayerNorm(num_hidden)
        self.norm2 = torch.nn.LayerNorm(num_hidden)

        self.W1 = torch.nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = torch.nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = torch.nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(
        self,
        h_V: torch.Tensor,
        h_E: torch.Tensor,
        mask_V: Optional[torch.Tensor] = None,
        mask_attend: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Parallel computation of full transformer layer

        Args:
            h_V: H v value.
            h_E: H e value.
            mask_V: Boolean mask for mask V.
            mask_attend: Boolean mask for mask attend.

        Returns:
            Computed tensor values.
        """

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(
            -1, -1, -1, h_E.size(-2), -1
        )  # the only difference
        h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V


class ContextBlock(torch.nn.Module):
    """Implement the context block component."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the ContextBlock.

        Args:
            hidden_dim: Dimension for hidden.
            dropout: Dropout value.
        """
        super(ContextBlock, self).__init__()
        self.y_context_encoder_layers = DecLayerJ(
            hidden_dim, hidden_dim, dropout=dropout
        )
        self.context_encoder_layers = DecLayer(
            hidden_dim, hidden_dim * 2, dropout=dropout
        )

    def forward(
        self,
        Y_nodes: torch.Tensor,
        Y_edges: torch.Tensor,
        Y_m: torch.Tensor,
        h_V_C: torch.Tensor,
        h_E_context: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[Any, ...]:
        """Run the forward pass.

        Args:
            Y_nodes: Y nodes value.
            Y_edges: Y edges value.
            Y_m: Y m value.
            h_V_C: H v c value.
            h_E_context: H e context value.
            mask: Boolean mask for mask.

        Returns:
            Computed result values.
        """
        Y_m_edges = Y_m[:, :, :, None] * Y_m[:, :, None, :]
        Y_nodes = self.y_context_encoder_layers(Y_nodes, Y_edges, Y_m, Y_m_edges)
        h_E_context_cat = torch.cat([h_E_context, Y_nodes], -1)
        h_V_C = self.context_encoder_layers(h_V_C, h_E_context_cat, mask, Y_m)
        return Y_nodes, h_V_C
