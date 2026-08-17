"""Protein batch-mask and side-chain-packing helpers."""

from __future__ import annotations

import os
from importlib import resources
from importlib.abc import Traversable
from typing import Any, Mapping, Union

import torch

from adflip.data.all_atom_parse import RESIDUE_TOKENS, RESTYPE_1TO3


def pippack_model_weight_path(filename: str) -> Traversable:
    """Execute the pippack model weight path operation.

    Args:
        filename: Filename value.

    Returns:
        Result of the pippack model weight path operation.
    """
    return resources.files("PIPPack").joinpath("model_weights", filename)


def batch_value(data: Union[Mapping[str, torch.Tensor], Any], key: str) -> torch.Tensor:
    """Read a batch field from either a mapping or an attribute container.

    Args:
        data: Data value.
        key: Key value.

    Returns:
        Computed tensor values.
    """
    if isinstance(data, Mapping):
        return data[key]
    return getattr(data, key)


def has_batch_value(data: Union[Mapping[str, torch.Tensor], Any], key: str) -> bool:
    """Execute the has batch value operation.

    Args:
        data: Data value.
        key: Key value.

    Returns:
        Whether the has batch value condition is satisfied.
    """
    if isinstance(data, Mapping):
        return key in data
    return hasattr(data, key)


def protein_center_mask(
    data: Union[Mapping[str, torch.Tensor], Any],
    *,
    require_backbone: bool = True,
) -> torch.Tensor:
    """Select protein center atoms, optionally requiring a complete backbone.

    Args:
        data: Data value.
        require_backbone: Require backbone value.

    Returns:
        Computed tensor values.
    """
    center_mask = (
        batch_value(data, "is_center").bool() & batch_value(data, "is_protein").bool()
    )
    if require_backbone and has_batch_value(data, "backbone_mask"):
        center_mask = center_mask & batch_value(data, "backbone_mask").bool()
    return center_mask


def designable_center_mask(
    data: Union[Mapping[str, torch.Tensor], Any],
    *,
    require_backbone: bool = True,
    mask_field: str = "mask_chain",
) -> torch.Tensor:
    """Select protein center atoms that are marked as designable.

    Args:
        data: Data value.
        require_backbone: Require backbone value.
        mask_field: Boolean mask for mask field.

    Returns:
        Computed tensor values.
    """
    center_mask = protein_center_mask(data, require_backbone=require_backbone)
    if has_batch_value(data, mask_field):
        center_mask = center_mask & batch_value(data, mask_field).bool()
    return center_mask


def center_loss_mask(
    data: Union[Mapping[str, torch.Tensor], Any],
    *,
    require_backbone: bool = True,
    mask_field: str = "mask_chain",
) -> torch.Tensor:
    """Execute the center loss mask operation.

    Args:
        data: Data value.
        require_backbone: Require backbone value.
        mask_field: Boolean mask for mask field.

    Returns:
        Computed tensor values.
    """
    center_mask = protein_center_mask(data, require_backbone=require_backbone)
    if not has_batch_value(data, mask_field):
        return torch.ones(
            int(center_mask.sum().item()),
            dtype=torch.bool,
            device=center_mask.device,
        )
    return batch_value(data, mask_field)[center_mask].bool()


def center_interaction_mask(
    data: Union[Mapping[str, torch.Tensor], Any],
    *,
    interaction_field: str = "interact_non_protein_res",
    require_backbone: bool = True,
) -> torch.Tensor:
    """Execute the center interaction mask operation.

    Args:
        data: Data value.
        interaction_field: Interaction field value.
        require_backbone: Require backbone value.

    Returns:
        Computed tensor values.
    """
    center_mask = protein_center_mask(data, require_backbone=require_backbone)
    if not has_batch_value(data, interaction_field):
        return torch.zeros(
            int(center_mask.sum().item()),
            dtype=torch.bool,
            device=center_mask.device,
        )
    return batch_value(data, interaction_field)[center_mask].bool()


def interacting_protein_residue_indices(
    data: Union[Mapping[str, torch.Tensor], Any],
    *,
    interaction_field: str = "interact_non_protein_res",
) -> torch.Tensor:
    """Execute the interacting protein residue indices operation.

    Args:
        data: Data value.
        interaction_field: Interaction field value.

    Returns:
        Computed tensor values.
    """
    is_protein = batch_value(data, "is_protein").bool()
    if not has_batch_value(data, interaction_field):
        return torch.empty(0, dtype=torch.long, device=is_protein.device)
    interaction_mask = batch_value(data, interaction_field).bool()
    residue_index = batch_value(data, "residue_index")
    return residue_index[is_protein][interaction_mask[is_protein]].unique()


def sampled_residue_sequence(samples: torch.Tensor) -> str:
    """Decode ADFLIP residue-token samples to a one-letter protein sequence.

    Args:
        samples: Samples value.

    Returns:
        Result of the sampled residue sequence operation.
    """
    decode_mapping = {token_id: name for name, token_id in RESIDUE_TOKENS.items()}
    residue_to_one_letter = {name: code for code, name in RESTYPE_1TO3.items()}
    samples = samples.clone()
    samples[samples > 21] = 10
    return "".join(
        residue_to_one_letter[decode_mapping[token.item()]]
        for token in samples.flatten()
    )


def write_packed_sidechains(
    protein_pdb: str,
    pdb_path: str,
    sample_save_folder: str,
    time: float,
) -> str:
    """Write a packed PDB to the original ADFLIP time-specific location.

    Args:
        protein_pdb: Protein pdb value.
        pdb_path: Path for pdb.
        sample_save_folder: Sample save folder value.
        time: Time value.

    Returns:
        Result of the write packed sidechains operation.
    """
    protein_name = os.path.basename(pdb_path).replace(".pdb", "") + "_0"
    save_path = os.path.join(sample_save_folder, f"side_chain_t={round(time, 3)}")
    os.makedirs(save_path, exist_ok=True)
    output_path = os.path.join(save_path, protein_name + ".pdb")
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(protein_pdb)
    return output_path
