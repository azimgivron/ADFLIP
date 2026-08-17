from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.pool import knn_graph

from adflip.model.zoidberg.transition_block import TransitionBlock


class DistanceEmbedding(nn.Module):
    """
    Inspired by AF3 Atom attention encoder
    """

    def __init__(self, dim: int, max_dist_ang: float = 32.0) -> None:
        """Initialize the DistanceEmbedding.

        Args:
            dim: Dimension for dim.
            max_dist_ang: Max dist ang value.
        """
        super().__init__()
        self.dim = dim
        self.distance_embedding = nn.Embedding(dim, embedding_dim=dim)
        self.register_buffer(
            "distance_bins",
            torch.linspace(
                start=0,
                end=max_dist_ang,
                steps=dim,
            ),
        )

    def forward(self, distance_matrix: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.

        Args:
            distance_matrix: Distance matrix value.

        Returns:
            Computed tensor values.
        """
        distance_matrix = distance_matrix[..., None]
        distance_idx = torch.argmin(
            torch.abs(distance_matrix - self.distance_bins[None, None]), dim=-1
        )
        distance_emb = self.distance_embedding(distance_idx)
        return distance_emb


class LocalAtomAttention(nn.Module):
    """Implement the local atom attention component."""

    def __init__(
        self, dim: int, num_heads: int, k: int, max_distance_ang: float = 32.0
    ) -> None:
        """Initialize the LocalAtomAttention.

        Args:
            dim: Dimension for dim.
            num_heads: Number of heads.
            k: K value.
            max_distance_ang: Max distance ang value.
        """
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.k = k
        self.max_distance_ang = max_distance_ang
        self.v = nn.Linear(dim, dim)
        self.distance_embedding = DistanceEmbedding(dim, max_distance_ang)
        self.affinity_pred = nn.Linear(dim, self.num_heads)
        self.fc = TransitionBlock(dim)

    def forward(
        self,
        x: torch.Tensor,
        batch: Any,
        positions: torch.Tensor,
        not_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the forward pass.

        Args:
            x: Input tensor.
            batch: Batch value.
            positions: Positions value.
            not_pad_mask: Boolean mask for not pad.

        Returns:
            Computed tensor values.
        """
        B, N, _ = positions.shape
        positions = positions.reshape(B * N, -1)
        knn_graph_indices = knn_graph(
            positions, k=self.k, batch=batch.flatten(), loop=True
        )
        distance_matrix = torch.norm(
            positions[knn_graph_indices[0]] - positions[knn_graph_indices[1]],
            dim=-1,
        )

        distance_matrix = distance_matrix.reshape(B, N, self.k)
        distance_matrix = self.distance_embedding(distance_matrix)
        affinities = torch.softmax(
            self.affinity_pred(distance_matrix).reshape(B, N, self.num_heads, self.k),
            dim=-1,
        )  # B, N, h, k

        # Apply padding mask to affinities
        not_pad_mask_matrix = not_pad_mask.reshape(B * N).float()
        not_pad_mask_matrix = (
            not_pad_mask_matrix[knn_graph_indices[0]]
            * not_pad_mask_matrix[knn_graph_indices[1]]
        )
        not_pad_mask_matrix = not_pad_mask_matrix.reshape(B, N, 1, self.k)
        affinities = affinities * not_pad_mask_matrix
        # Renormalize
        affinities = affinities / (affinities.sum(dim=-1, keepdim=True) + 1e-6)

        # Calculate values
        values = self.v(x)
        values = values.reshape(B, N, self.num_heads, self.head_dim, 1)
        updated_values = torch.sum(
            affinities[..., None, :] * values, dim=-1
        )  # B, N, h, d
        updated_values = updated_values.reshape(B, N, self.dim)
        x = self.fc(updated_values) + x
        return x
