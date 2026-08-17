from __future__ import annotations

from typing import Any, Tuple

import torch
import torch.nn as nn

try:
    import PIPPack.data.residue_constants as rc
    from PIPPack.data.rigid_utils import Rigid, Rotation
except:
    import PIPPack.data.residue_constants as rc
    from PIPPack.data.rigid_utils import Rigid, Rotation


def make_atom14_masks(S: torch.Tensor) -> Tuple[Any, ...]:
    """Construct denser atom positions (14 dimensions instead of 37).

    Args:
        S: Input tensor.

    Returns:
        Computed result values.
    """
    restype_atom14_to_atom37 = []
    restype_atom37_to_atom14 = []
    restype_atom14_mask = []

    for rt in rc.RESTYPES:
        atom_names = rc.RESTYPE_NAME_TO_ATOM14_NAMES[rc.RESTYPE_1TO3[rt]]
        restype_atom14_to_atom37.append(
            [(rc.ATOM_ORDER[name] if name else 0) for name in atom_names]
        )
        atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
        restype_atom37_to_atom14.append(
            [
                (atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0)
                for name in rc.ATOM_TYPES
            ]
        )

        restype_atom14_mask.append([(1.0 if name else 0.0) for name in atom_names])

    # Add dummy mapping for restype 'UNK'
    restype_atom14_to_atom37.append([0] * 14)
    restype_atom37_to_atom14.append([0] * 37)
    restype_atom14_mask.append([0.0] * 14)

    restype_atom14_to_atom37 = torch.tensor(
        restype_atom14_to_atom37,
        dtype=torch.int32,
        device=S.device,
    )
    restype_atom37_to_atom14 = torch.tensor(
        restype_atom37_to_atom14,
        dtype=torch.int32,
        device=S.device,
    )
    restype_atom14_mask = torch.tensor(
        restype_atom14_mask,
        dtype=torch.float32,
        device=S.device,
    )
    protein_aatype = S.to(torch.long)

    # create the mapping for (residx, atom14) --> atom37, i.e. an array
    # with shape (num_res, 14) containing the atom37 indices for this protein
    residx_atom14_to_atom37 = restype_atom14_to_atom37[protein_aatype]
    residx_atom14_mask = restype_atom14_mask[protein_aatype]

    # create the gather indices for mapping back
    residx_atom37_to_atom14 = restype_atom37_to_atom14[protein_aatype].long()

    # create the corresponding mask
    restype_atom37_mask = torch.zeros([21, 37], dtype=torch.float32, device=S.device)
    for restype, restype_letter in enumerate(rc.RESTYPES):
        restype_name = rc.RESTYPE_1TO3[restype_letter]
        atom_names = rc.RESIDUE_ATOMS[restype_name]
        for atom_name in atom_names:
            atom_type = rc.ATOM_ORDER[atom_name]
            restype_atom37_mask[restype, atom_type] = 1

    residx_atom37_mask = restype_atom37_mask[protein_aatype]

    return (
        residx_atom37_to_atom14,
        residx_atom37_mask,
        residx_atom14_to_atom37,
        residx_atom14_mask,
    )


def atom14_to_atom37(
    atom14_data: torch.Tensor,  # (B, N, 14, 3)
    residx_atom37_to_atom14: torch.Tensor,  # (B, N, 37)
    atom37_atom_exists: torch.Tensor,  # (B, N, 37)
) -> torch.Tensor:  # (B, N, 37, 3)
    """Convert atom14 to atom37 representation.

    Args:
        atom14_data: Atom14 data value.
        residx_atom37_to_atom14: Residx atom37 to atom14 value.
        atom37_atom_exists: Atom37 atom exists value.

    Returns:
        Computed tensor values.
    """

    atom37_data = torch.gather(
        atom14_data,
        dim=2,
        index=residx_atom37_to_atom14.unsqueeze(-1).expand(-1, -1, -1, 3).long(),
    )

    atom37_data *= atom37_atom_exists[..., None].float()

    return atom37_data


def get_bb_frames(
    N: torch.Tensor, CA: torch.Tensor, C: torch.Tensor, fixed: bool = True
) -> Any:
    # N, CA, C = [*, L, 3]
    """Return bb frames.

    Args:
        N: N value.
        CA: Ca value.
        C: C value.
        fixed: Fixed value.

    Returns:
        Result of the get bb frames operation.
    """
    return Rigid.from_3_points(N, CA, C, fixed=fixed)


def torsion_angles_to_frames(
    r: Rigid,
    alpha: torch.Tensor,
    aatype: torch.Tensor,
    rrgdf: torch.Tensor,
) -> Any:
    # [*, N, 8, 4, 4]
    """Execute the torsion angles to frames operation.

    Args:
        r: Input tensor.
        alpha: Alpha value.
        aatype: Aatype value.
        rrgdf: Rrgdf value.

    Returns:
        Result of the torsion angles to frames operation.
    """
    default_4x4 = rrgdf[aatype, ...]

    # [*, N, 8] transformations, i.e.
    #   One [*, N, 8, 3, 3] rotation matrix and
    #   One [*, N, 8, 3]    translation matrix
    default_r = r.from_tensor_4x4(default_4x4)

    bb_rot = alpha.new_zeros((*((1,) * len(alpha.shape[:-1])), 2))
    bb_rot[..., 1] = 1

    # [*, N, 8, 2]
    alpha = torch.cat([bb_rot.expand(*alpha.shape[:-2], -1, -1), alpha], dim=-2)

    # [*, N, 8, 3, 3]
    # Produces rotation matrices of the form:
    # [
    #   [1, 0  , 0  ],
    #   [0, a_2,-a_1],
    #   [0, a_1, a_2]
    # ]
    # This follows the original code rather than the supplement, which uses
    # different indices.

    all_rots = alpha.new_zeros(default_r.get_rots().get_rot_mats().shape)
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = alpha[..., 1]
    all_rots[..., 1, 2] = -alpha[..., 0]
    all_rots[..., 2, 1:] = alpha

    all_rots = Rigid(Rotation(rot_mats=all_rots), None)

    all_frames = default_r.compose(all_rots)

    chi2_frame_to_frame = all_frames[..., 5]
    chi3_frame_to_frame = all_frames[..., 6]
    chi4_frame_to_frame = all_frames[..., 7]

    chi1_frame_to_bb = all_frames[..., 4]
    chi2_frame_to_bb = chi1_frame_to_bb.compose(chi2_frame_to_frame)
    chi3_frame_to_bb = chi2_frame_to_bb.compose(chi3_frame_to_frame)
    chi4_frame_to_bb = chi3_frame_to_bb.compose(chi4_frame_to_frame)

    all_frames_to_bb = Rigid.cat(
        [
            all_frames[..., :5],
            chi2_frame_to_bb.unsqueeze(-1),
            chi3_frame_to_bb.unsqueeze(-1),
            chi4_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )

    all_frames_to_global = r[..., None].compose(all_frames_to_bb)

    return all_frames_to_global


def frames_and_literature_positions_to_atom14_pos(
    r: Rigid,
    aatype: torch.Tensor,
    default_frames: torch.Tensor,
    group_idx: int,
    atom_mask: torch.Tensor,
    lit_positions: torch.Tensor,
) -> Any:
    # [*, N, 14]
    """Execute the frames and literature positions to atom14 pos operation.

    Args:
        r: Input tensor.
        aatype: Aatype value.
        default_frames: Default frames value.
        group_idx: Group idx value.
        atom_mask: Boolean mask for atom.
        lit_positions: Lit positions value.

    Returns:
        Result of the frames and literature positions to atom14 pos operation.
    """
    group_mask = group_idx[aatype, ...]

    # [*, N, 14, 8]
    group_mask = nn.functional.one_hot(
        group_mask,
        num_classes=default_frames.shape[-3],
    )

    # [*, N, 14, 8]
    t_atoms_to_global = r[..., None, :] * group_mask

    # [*, N, 14]
    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(lambda x: torch.sum(x, dim=-1))

    # [*, N, 14, 1]
    atom_mask = atom_mask[aatype, ...].unsqueeze(-1)

    # [*, N, 14, 3]
    lit_positions = lit_positions[aatype, ...]
    pred_positions = t_atoms_to_global.apply(lit_positions)
    pred_positions = pred_positions * atom_mask

    return pred_positions
