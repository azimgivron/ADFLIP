from __future__ import annotations

import torch
import torch.nn as nn

from adflip.model.zoidberg.utils import scatter_residue_feature_over_atoms


class NodeToAtomEmbedder(nn.Module):
    """Implement the node to atom embedder component."""

    def __init__(self, dim: int) -> None:
        """Initialize the NodeToAtomEmbedder.

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
        residue_features: torch.Tensor,
        unique_residue_index: torch.Tensor,
        not_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the forward pass.

        Args:
            x: Input tensor.
            residue_features: Residue features value.
            unique_residue_index: Unique residue index value.
            not_pad_mask: Boolean mask for not pad.

        Returns:
            Computed tensor values.
        """
        residue_features_scattered = scatter_residue_feature_over_atoms(
            residue_features, unique_residue_index, not_pad_mask
        )
        residue_features_scattered = self.proj(residue_features_scattered)
        return x + residue_features_scattered
