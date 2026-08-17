"""Focused PIPPack loading and side-chain-packing functions.

The functions are stateless apart from the explicitly supplied model ensemble,
which lets Hessian Flow decide how to schedule or parallelize packing work.
"""

from __future__ import annotations

import pickle
from importlib.abc import Traversable
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from adflip.data.all_atom_parse import StructureData, make_merged_pdb, pdb2data
from adflip.model.utils import (
    pippack_model_weight_path,
    sampled_residue_sequence,
    write_packed_sidechains,
)
from PIPPack.data.protein import from_pdb_file
from PIPPack.data.top2018_dataset import collate_fn, transform_structure
from PIPPack.ensembled_inference import sample_epoch
from PIPPack.inference import pdbs_from_prediction, replace_protein_sequence
from PIPPack.model.modules import PIPPackFineTune

_MODEL_NAMES = (
    "pippack_model_1",
    "pippack_model_2",
    "pippack_model_3",
)


def _weight_path(
    weights_dir: Optional[Path], filename: str
) -> Union[Path, Traversable]:
    """Execute the weight path operation.

    Args:
        weights_dir: Path for weights.
        filename: Filename value.

    Returns:
        Result of the weight path operation.
    """
    if weights_dir is None:
        return pippack_model_weight_path(filename)
    return weights_dir / filename


def load_sidechain_models(
    device: Union[torch.device, str],
    *,
    num_models: int = 3,
    weights_dir: Optional[Union[str, Path]] = None,
) -> Tuple[List[nn.Module], Any]:
    """Load the bundled PIPPack ensemble and its inference configuration.

    Args:
        device: Device used for tensor operations.
        num_models: Number of models.
        weights_dir: Path for weights.

    Returns:
        Computed result values.
    """
    if not 1 <= num_models <= len(_MODEL_NAMES):
        raise ValueError(f"num_models must be between 1 and {len(_MODEL_NAMES)}.")

    resolved_device = torch.device(device)
    resolved_weights_dir = None if weights_dir is None else Path(weights_dir)
    inference_path = _weight_path(resolved_weights_dir, "inference.pickle")
    with inference_path.open("rb") as inference_file:
        inference_config = pickle.load(inference_file)

    models: List[nn.Module] = []
    for model_name in _MODEL_NAMES[:num_models]:
        config_path = _weight_path(resolved_weights_dir, f"{model_name}_config.pickle")
        checkpoint_path = _weight_path(resolved_weights_dir, f"{model_name}_ckpt.pt")
        with config_path.open("rb") as config_file:
            config = pickle.load(config_file)

        model = PIPPackFineTune(
            node_features=config.model.node_features,
            edge_features=config.model.edge_features,
            hidden_dim=config.model.hidden_dim,
            num_mpnn_layers=config.model.num_mpnn_layers,
            k_neighbors=config.model.k_neighbors,
            augment_eps=config.model.augment_eps,
            use_ipmp=config.model.use_ipmp,
            use_ipmp_ipa=config.model.use_ipmp_ipa,
            n_points=config.model.n_points,
            dropout=config.model.dropout,
            act=config.model.act,
            predict_bin_chi=config.model.predict_bin_chi,
            n_chi_bins=config.model.n_chi_bins,
            predict_offset=config.model.predict_offset,
            position_scale=config.model.position_scale,
            recycle_strategy=config.model.recycle_strategy,
            recycle_SC_D_sc=config.model.recycle_SC_D_sc,
            recycle_SC_D_probs=config.model.recycle_SC_D_probs,
            recycle_X=config.model.recycle_X,
            mask_distances=config.model.mask_distances,
            loss=config.model.loss,
        )
        state = torch.load(
            str(checkpoint_path),
            map_location=resolved_device,
            weights_only=False,
        )
        model.load_state_dict(state["model_state_dict"])
        models.append(model.to(resolved_device))

    return models, inference_config


def pack_sidechains(
    samples: torch.Tensor,
    pdb_path: Union[str, Path],
    *,
    models: Sequence[nn.Module],
    inference_config: Dict[str, Any],
    output_dir: Union[str, Path],
    time: float,
    device: Optional[Union[torch.device, str]] = None,
) -> StructureData:
    """Pack a sampled sequence and return it as ADFLIP all-atom data.

    Args:
        samples: Samples value.
        pdb_path: Path for pdb.
        models: Models value.
        inference_config: Configuration values.
        output_dir: Path for output.
        time: Time value.
        device: Device used for tensor operations.

    Returns:
        Result of the pack sidechains operation.
    """
    if not models:
        raise ValueError("At least one side-chain model is required.")
    resolved_device = (
        torch.device(device)
        if device is not None
        else next(models[0].parameters()).device
    )
    pdb_path = Path(pdb_path)
    sequence = sampled_residue_sequence(samples)
    proteins = replace_protein_sequence(
        vars(from_pdb_file(str(pdb_path), mse_to_met=True, ignore_non_std=False)),
        str(pdb_path),
        [[sequence]],
    )
    transformed = transform_structure(proteins[0][1], sc_d_mask_from_seq=True)
    batch = collate_fn([transformed])
    results = sample_epoch(
        list(models),
        batch,
        inference_config["temperature"],
        resolved_device,
        n_recycle=1,
        resample=False,
        resample_args=inference_config["resample_args"],
    )
    packed_pdb = pdbs_from_prediction(results)[0]
    output_path = write_packed_sidechains(
        packed_pdb,
        str(pdb_path),
        str(output_dir),
        time,
    )
    make_merged_pdb(str(pdb_path), output_path)
    return pdb2data(output_path, resolved_device)


__all__ = ["load_sidechain_models", "pack_sidechains"]
