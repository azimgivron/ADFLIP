import os
from importlib import resources
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from adflip.data.all_atom_parse import residue_tokens, restype_1to3


def is_abnormal_pdb_line(line: str) -> bool:
    """Return *True* if the ATOM / HETATM / ANISOU record is shifted left."""

    if not line.startswith(("ATOM  ", "HETATM", "ANISOU")):
        return False

    # 1. residue name length > 3 (columns 18‑20 plus possible spill‑over)
    res_field = line[17:21]  # cols 18‑21 (0‑based slice 17:21)
    if len(res_field.strip()) > 3:
        return True

    # 2. column 21 (index 20) must be blank in a valid PDB
    if len(line) > 20 and line[20] != " ":
        return True

    # 3. column 23 (index 22) is the first of the 4‑char residue number field
    #    If it contains a minus sign, the record is shifted left.
    if len(line) > 22 and line[22] == "-":
        return True

    return False


def pippack_model_weight_path(filename: str):
    return resources.files("PIPPack").joinpath("model_weights", filename)


def _batch_value(data: Mapping[str, torch.Tensor] | Any, key: str) -> torch.Tensor:
    if isinstance(data, Mapping):
        return data[key]
    return getattr(data, key)


def _has_batch_value(data: Mapping[str, torch.Tensor] | Any, key: str) -> bool:
    if isinstance(data, Mapping):
        return key in data
    return hasattr(data, key)


def protein_center_mask(
    data: Mapping[str, torch.Tensor] | Any,
    *,
    require_backbone: bool = True,
) -> torch.Tensor:
    center_mask = (
        _batch_value(data, "is_center").bool() & _batch_value(data, "is_protein").bool()
    )
    if require_backbone and _has_batch_value(data, "backbone_mask"):
        center_mask = center_mask & _batch_value(data, "backbone_mask").bool()
    return center_mask


def designable_center_mask(
    data: Mapping[str, torch.Tensor] | Any,
    *,
    require_backbone: bool = True,
    mask_field: str = "mask_chain",
) -> torch.Tensor:
    center_mask = protein_center_mask(data, require_backbone=require_backbone)
    if _has_batch_value(data, mask_field):
        center_mask = center_mask & _batch_value(data, mask_field).bool()
    return center_mask


def center_residue_targets(
    data: Mapping[str, torch.Tensor] | Any,
    *,
    require_backbone: bool = True,
) -> torch.Tensor:
    return _batch_value(data, "residue_token")[
        protein_center_mask(data, require_backbone=require_backbone)
    ]


def center_loss_mask(
    data: Mapping[str, torch.Tensor] | Any,
    *,
    require_backbone: bool = True,
    mask_field: str = "mask_chain",
) -> torch.Tensor:
    center_mask = protein_center_mask(data, require_backbone=require_backbone)
    if not _has_batch_value(data, mask_field):
        return torch.ones(
            int(center_mask.sum().item()),
            dtype=torch.bool,
            device=center_mask.device,
        )
    return _batch_value(data, mask_field)[center_mask].bool()


def center_interaction_mask(
    data: Mapping[str, torch.Tensor] | Any,
    *,
    interaction_field: str = "interact_non_protein_res",
    require_backbone: bool = True,
) -> torch.Tensor:
    center_mask = protein_center_mask(data, require_backbone=require_backbone)
    if not _has_batch_value(data, interaction_field):
        return torch.zeros(
            int(center_mask.sum().item()),
            dtype=torch.bool,
            device=center_mask.device,
        )
    return _batch_value(data, interaction_field)[center_mask].bool()


def interacting_protein_residue_indices(
    data: Mapping[str, torch.Tensor] | Any,
    *,
    interaction_field: str = "interact_non_protein_res",
) -> torch.Tensor:
    is_protein = _batch_value(data, "is_protein").bool()
    if not _has_batch_value(data, interaction_field):
        return torch.empty(0, dtype=torch.long, device=is_protein.device)
    interaction_mask = _batch_value(data, interaction_field).bool()
    residue_index = _batch_value(data, "residue_index")
    return residue_index[is_protein][interaction_mask[is_protein]].unique()


def masked_residue_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    label_smoothing: bool | float = False,
) -> torch.Tensor:
    if mask is not None:
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        logits = logits[mask]
        targets = targets[mask]

    targets = targets.long()
    smoothing = float(label_smoothing) if label_smoothing else 0.0
    if smoothing <= 0.0:
        return F.cross_entropy(logits, targets)

    target_onehot = F.one_hot(targets, num_classes=logits.size(-1)).to(
        dtype=logits.dtype
    )
    target_onehot = target_onehot + smoothing / float(target_onehot.size(-1))
    target_onehot = target_onehot / target_onehot.sum(-1, keepdim=True)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_onehot * log_probs).sum(-1).mean()


def protein_residue_cross_entropy(
    logits: torch.Tensor,
    data: Mapping[str, torch.Tensor] | Any,
    *,
    label_smoothing: bool | float = False,
    require_backbone: bool = True,
) -> torch.Tensor:
    targets = center_residue_targets(data, require_backbone=require_backbone)
    loss_mask = center_loss_mask(data, require_backbone=require_backbone)
    return masked_residue_cross_entropy(
        logits,
        targets,
        loss_mask,
        label_smoothing=label_smoothing,
    )


def sampled_residue_sequence(samples: torch.Tensor) -> str:
    decode_mapping = {j: i for i, j in residue_tokens.items()}
    restype_3to1 = {v: k for k, v in restype_1to3.items()}
    samples = samples.clone()
    samples[samples > 21] = 10
    return "".join([restype_3to1[decode_mapping[i.item()]] for i in samples.flatten()])


def write_packed_sidechains(
    protein_pdb: str,
    pdb_path: str,
    sample_save_folder: str,
    t: float,
) -> str:
    protein_name = os.path.basename(pdb_path).replace(".pdb", "") + "_0"
    save_path = os.path.join(sample_save_folder, f"side_chain_t={round(t, 3)}")
    os.makedirs(save_path, exist_ok=True)
    output_path = os.path.join(save_path, protein_name + ".pdb")
    with open(output_path, "w") as f:
        f.write(protein_pdb)
    return output_path
