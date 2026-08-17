from __future__ import annotations

from typing import Any, Tuple

import torch
import torch.nn as nn

from adflip.model.zoidberg.utils import gather_residue_average_from_atoms


class AtomToNodeEmbedder(nn.Module):
    """Implement the atom to node embedder component."""

    def __init__(self, dim: int) -> None:
        """Initialize the AtomToNodeEmbedder.

        Args:
            dim: Dimension for dim.
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
        )

    def forward(
        self,
        x: torch.Tensor,
        is_center: torch.Tensor,
        unique_residue_index: torch.Tensor,
        not_pad_mask: torch.Tensor,
    ) -> Tuple[Any, ...]:
        """Run the forward pass.

        Args:
            x: Input tensor.
            is_center: Is center value.
            unique_residue_index: Unique residue index value.
            not_pad_mask: Boolean mask for not pad.

        Returns:
            Computed result values.
        """
        x = self.proj(x)
        node_emb, residue_mask = gather_residue_average_from_atoms(
            x, unique_residue_index, not_pad_mask
        )
        return node_emb, residue_mask
