from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn


class FourierEmbedding(nn.Module):
    """Implement the fourier embedding component."""

    def __init__(self, dim: int) -> None:
        """AlphaFold3's Fourier Embedding

        Args:
            dim: Dimension for dim.
        """
        super().__init__()
        self.dim = dim
        self.register_buffer("weight", torch.randn(dim))
        self.register_buffer("bias", torch.randn(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.

        Args:
            x: Input tensor.

        Returns:
            Computed tensor values.
        """
        return torch.cos(2 * torch.pi * (x[..., None] * self.weight + self.bias))


class AdaptiveLayerNorm(nn.Module):
    """Implement the adaptive layer norm component."""

    def __init__(self, dim: int, condition_dim: int = None, eps: float = 1e-6) -> None:
        """Initialize the AdaptiveLayerNorm.

        Args:
            dim: Dimension for dim.
            condition_dim: Dimension for condition.
            eps: Eps value.
        """
        super().__init__()
        if condition_dim is None:
            condition_dim = dim
        self.eps = eps
        self.ln1 = nn.LayerNorm(dim, elementwise_affine=False, bias=False, eps=eps)
        self.ln2 = nn.LayerNorm(condition_dim, bias=False, eps=eps)
        self.linear_nb = nn.Linear(condition_dim, dim, bias=False)
        self.linear_sig = nn.Linear(condition_dim, dim)

        nn.init.constant_(self.linear_sig.bias, -2.0)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.

        Args:
            x: Input tensor.
            condition: Condition value.

        Returns:
            Computed tensor values.
        """
        x = self.ln1(x)
        y = self.ln2(condition)
        b = torch.sigmoid(self.linear_sig(y)) * x + self.linear_nb(y)
        return b


class FrameAveraging(nn.Module):
    """Implement the frame averaging component."""

    def __init__(self) -> None:
        """Initialize the FrameAveraging."""
        super().__init__()
        self.ops = torch.tensor(
            [[i, j, k] for i in [-1, 1] for j in [-1, 1] for k in [-1, 1]]
        )

    def create_frame(self, X: torch.Tensor, mask: torch.Tensor) -> Tuple[Any, ...]:
        """Create frame.

        Args:
            X: Input tensor.
            mask: Boolean mask for mask.

        Returns:
            Computed result values.
        """
        device = X.device
        mask = mask.unsqueeze(-1)
        center = (X * mask).sum(dim=1) / mask.sum(dim=1)
        X = X - center.unsqueeze(1) * mask  # [B,N,3]
        C = torch.bmm(X.transpose(1, 2), X)  # [B,3,3] (Cov)
        _, V = torch.linalg.eigh(C.detach(), UPLO="U")  # [B,3,3]
        self.ops = self.ops.to(device)
        F_ops = self.ops.unsqueeze(1).unsqueeze(0) * V.unsqueeze(
            1
        )  # [1,8,1,3] x [B,1,3,3] -> [B,8,3,3]
        h = torch.einsum(
            "boij,bpj->bopi", F_ops.transpose(2, 3), X
        )  # transpose is inverse [B,8,N,3]
        h = h.view(X.size(0) * 8, X.size(1), 3)
        return h, F_ops.detach(), center

    def invert_frame(
        self,
        X: torch.Tensor,
        mask: torch.Tensor,
        F_ops: torch.Tensor,
        center: torch.Tensor,
    ) -> torch.Tensor:
        """Execute the invert frame operation.

        Args:
            X: Input tensor.
            mask: Boolean mask for mask.
            F_ops: F ops value.
            center: Center value.

        Returns:
            Computed tensor values.
        """
        X = torch.einsum("boij,bopj->bopi", F_ops, X)
        X = X.mean(dim=1)  # frame averaging
        X = X + center.unsqueeze(1)
        return X * mask.unsqueeze(-1)


def transform_residue_index(
    residue_index: torch.Tensor, atom_mask: torch.Tensor
) -> Any:
    """Transform residue index.

    Args:
        residue_index: Residue index value.
        atom_mask: Boolean mask for atom.

    Returns:
        Result of the transform residue index operation.
    """
    B, N = residue_index.shape
    device = residue_index.device

    # Create a tensor to hold the transformed indices
    transformed_index = torch.full_like(residue_index, -1)

    for b in range(B):
        # Get the residue indices and mask for this batch
        batch_indices = residue_index[b]
        batch_mask = atom_mask[b] & (batch_indices != -1)  # Exclude -1 indices

        # Find the unique residue indices, respecting order
        unique_indices, inverse_indices = torch.unique_consecutive(
            batch_indices[batch_mask], return_inverse=True
        )

        # Assign the new indices
        transformed_index[b, batch_mask] = inverse_indices

    return transformed_index


def transform_atom_index(
    atom_index: torch.Tensor, atom_mask: torch.Tensor, batch_indices: torch.Tensor
) -> Any:
    """Transform atom index.

    Args:
        atom_index: Atom index value.
        atom_mask: Boolean mask for atom.
        batch_indices: Batch indices value.

    Returns:
        Result of the transform atom index operation.
    """
    B, N = atom_index.shape
    device = atom_index.device

    # Find the maximum residue index for each batch
    max_atom_index = atom_index.max(dim=1).values
    max_K = torch.zeros_like(max_atom_index)
    max_K[1:] = (torch.cumsum(max_atom_index, dim=0) + 1)[:-1]
    max_K = max_K[:, None]

    # Create batch indices
    batch_indices = torch.ones(B, N, device=device, dtype=torch.long)

    # Create a linear index for each residue
    linear_index = batch_indices * max_K + atom_index

    # Create a mask for valid atoms (not masked and have a valid residue index)
    valid_mask = atom_mask & (atom_index >= 0)

    linear_index = torch.where(
        valid_mask, linear_index, torch.full_like(linear_index, -1)
    )
    return linear_index


def gather_residue_average_from_atoms(
    x: torch.Tensor, residue_index: torch.Tensor, atom_mask: torch.Tensor
) -> Tuple[Any, ...]:
    """Execute the gather residue average from atoms operation.

    Args:
        x: Input tensor.
        residue_index: Residue index value.
        atom_mask: Boolean mask for atom.

    Returns:
        Computed result values.
    """
    B, N, D = x.shape
    device = x.device
    # Apply atom mask to x
    x_masked = x * atom_mask.unsqueeze(-1)
    # Find the maximum residue index for each batch
    max_residue_index = residue_index.max()

    max_K = max_residue_index.item() + 1

    # Create batch indices
    batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, N)

    # Create a linear index for each residue
    linear_index = batch_indices * max_K + residue_index

    # Create a mask for valid atoms (not masked and have a valid residue index)
    valid_mask = atom_mask & (residue_index >= 0)

    # Sum the vectors for each residue and count the atoms
    residue_sum = torch.zeros(B * max_K, D, device=device)
    residue_count = torch.zeros(B * max_K, 1, device=device)

    residue_sum.index_add_(0, linear_index[valid_mask], x_masked[valid_mask])
    residue_count.index_add_(
        0, linear_index[valid_mask], torch.ones(valid_mask.sum(), 1, device=device)
    )

    # Calculate the average
    residue_avg = residue_sum / (
        residue_count + 1e-10
    )  # Add small epsilon to avoid division by zero

    # Reshape to [B, max_K, D]
    residue_avg = residue_avg.view(B, max_K, D)

    # Create residue mask
    residue_mask = residue_count.view(B, max_K) > 0

    return residue_avg, residue_mask


def scatter_residue_feature_over_atoms(
    residue_avg: torch.Tensor, residue_index: torch.Tensor, atom_mask: torch.Tensor
) -> torch.Tensor:
    """Execute the scatter residue feature over atoms operation.

    Args:
        residue_avg: Residue avg value.
        residue_index: Residue index value.
        atom_mask: Boolean mask for atom.

    Returns:
        Result of the scatter residue feature over atoms operation.
    """
    B, K, D = residue_avg.shape
    N = residue_index.shape[1]
    device = residue_avg.device

    # Create a tensor to hold the reconstructed atom features
    x_reconstructed = torch.zeros((B, N, D), device=device)

    # Create batch indices
    batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, N)

    # Use advanced indexing to distribute the residue averages back to atoms
    x_reconstructed = residue_avg[batch_indices, residue_index]

    # Apply the atom mask
    x_reconstructed = x_reconstructed * atom_mask.unsqueeze(-1)

    return x_reconstructed


def find_first_padded(not_pad_mask: torch.Tensor) -> Any:
    """Courtesy of Claude 3.5

    Args:
        not_pad_mask: Boolean mask for not pad.

    Returns:
        Result of the find first padded operation.
    """
    # Find the first False in each sequence
    first_padded = (~not_pad_mask).long().argmax(dim=1)

    # If a sequence has no padding, argmax will return 0
    # We need to set it to the length of the sequence
    seq_lengths = not_pad_mask.sum(dim=1)
    no_padding = (first_padded == 0) & (seq_lengths == not_pad_mask.shape[1])
    first_padded[no_padding] = not_pad_mask.shape[1]

    return first_padded


class StartEndPad(nn.Module):
    """Implement the start end pad component."""

    def __init__(self, dim: int) -> None:
        """Initialize the StartEndPad.

        Args:
            dim: Dimension for dim.
        """
        super().__init__()
        self.start = nn.Parameter(torch.randn(dim))
        self.end = nn.Parameter(torch.randn(dim))

    def forward(self, x: torch.Tensor, protein_mask: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.

        Args:
            x: Input tensor.
            protein_mask: Boolean mask for protein.

        Returns:
            Computed tensor values.
        """
        B, N, D = x.shape
        x = nn.functional.pad(x, (0, 0, 1, 1), value=0)  # shape: [B, N+1, D]
        protein_mask = nn.functional.pad(
            protein_mask, (0, 1), value=False
        )  # shape: [B, N+1]
        first_padded = find_first_padded(protein_mask)
        x[:, 0] = self.start[None].repeat(B, 1)
        x[torch.arange(B), first_padded] = self.end[None].repeat(B, 1)
        return x


# timm SwiGLU
class SwiGLULayer(nn.Module):
    """Implement the swi glulayer component."""

    def __init__(
        self,
        in_features: torch.Tensor,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Any = nn.SiLU,
        norm_layer: Optional[nn.Module] = None,
        bias: bool = True,
        drop: float = 0.0,
    ) -> None:
        """Initialize the SwiGLULayer.

        Args:
            in_features: In features value.
            hidden_features: Hidden features value.
            out_features: Out features value.
            act_layer: Act layer value.
            norm_layer: Norm layer value.
            bias: Bias value.
            drop: Drop value.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = [bias, bias]
        drop_probs = [drop, drop]

        self.fc1_g = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.fc1_x = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self) -> None:
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        """Execute the init weights operation."""
        nn.init.ones_(self.fc1_g.bias)
        nn.init.normal_(self.fc1_g.weight, std=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.

        Args:
            x: Input tensor.

        Returns:
            Computed tensor values.
        """
        x_gate = self.fc1_g(x)
        x = self.fc1_x(x)
        x = self.act(x_gate) * x
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
