from __future__ import annotations

import dataclasses
import gzip
import os
from collections import OrderedDict
from importlib import resources
from typing import Any, Dict, List, Optional, OrderedDict, Union

import numpy as np
import prody
import torch
from Bio.Data import IUPACData
from Bio.PDB import PDBIO, MMCIFParser, PDBParser, Select
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from scipy.spatial import cKDTree
from torch import Tensor


def _read_misc_lines(filename: str) -> List[str]:
    """Read misc lines.

    Args:
        filename: Filename value.

    Returns:
        Computed result items.
    """
    resource = resources.files(__package__).joinpath("misc", filename)
    return [line.strip() for line in resource.read_text().splitlines()]


RESTYPE_1TO3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "X": "<UNK>",
}
RESTYPE_3TO1 = {v: k for k, v in RESTYPE_1TO3.items()}


# Tokenization goes from proteins to nucleic acids, to single atoms
RESIDUE_TOKENS = OrderedDict()

RESIDUE_TOKENS["<PAD>"] = len(RESIDUE_TOKENS)
RESIDUE_TOKENS["<MASK>"] = len(RESIDUE_TOKENS)


def add_flag_tokens(flag: Any) -> None:
    """Execute the add flag tokens operation.

    Args:
        flag: Flag value.
    """
    for residue in prody.flagDefinition(flag):
        if residue not in RESIDUE_TOKENS:
            RESIDUE_TOKENS[residue] = len(RESIDUE_TOKENS)


add_flag_tokens("stdaa")
RESIDUE_TOKENS["<UNK>"] = len(RESIDUE_TOKENS)

# Extra tokens for ligands
add_flag_tokens("nucleotide")
# add_flag_tokens("nonstdaa")
# add_flag_tokens("nucleic")
# add_flag_tokens("ion")
# RESIDUE_TOKENS["<GLYCAN>"] = len(RESIDUE_TOKENS)


TOKEN_TO_INDEX = {token: i for i, token in enumerate(RESIDUE_TOKENS)}
INDEX_TO_TOKEN = {i: token for i, token in enumerate(RESIDUE_TOKENS)}
NUM_RESIDUE_TOKENS = len(RESIDUE_TOKENS)

# NUM_PROTEIN_TOKENS = RESIDUE_TOKENS["XAA"] + 1


# Residue type sets — imported from centralized config
from adflip.data.residue_config import ION, NUCLEOTIDE, get_protein_residues

NUCLEOTIDE_RESIDUES = NUCLEOTIDE
ION_RESIDUES = ION


# Element tokens
ELEMENTS = list(IUPACData.atom_weights.keys())
ELEMENTS.pop(ELEMENTS.index("H"))  # Remove hydrogen
ELEMENTS.append("<ATOM_UNK>")
ELEMENTS.append("<ATOM_PAD>")
ELEMENTS = [e.upper() for e in ELEMENTS]

ELEMENT_TO_INDEX = {element: i for i, element in enumerate(ELEMENTS)}
NUM_ELEMENT_TOKENS = len(ELEMENTS)


# AF3 crystallization aids
AF3_CRYSTALLIZATION_AIDS = _read_misc_lines("af3_crystallization_aids.txt")
AF3_CRYSTALLIZATION_AIDS = set(AF3_CRYSTALLIZATION_AIDS)


# AF3 ligand exclusion list
AF3_LIGANDS_EXCLUDED = _read_misc_lines("af3_ligands_excluded.txt")
AF3_LIGANDS_EXCLUDED = set(AF3_LIGANDS_EXCLUDED)

# Glycans
AF3_GLYCANS = _read_misc_lines("af3_glycans.txt")
AF3_GLYCANS = set(AF3_GLYCANS)

_LEGACY_CONSTANT_NAMES = {
    "restype_1to3": "RESTYPE_1TO3",
    "restype_3to1": "RESTYPE_3TO1",
    "residue_tokens": "RESIDUE_TOKENS",
    "token_to_index": "TOKEN_TO_INDEX",
    "index_to_token": "INDEX_TO_TOKEN",
    "num_residue_tokens": "NUM_RESIDUE_TOKENS",
    "nucleotide_residues": "NUCLEOTIDE_RESIDUES",
    "ion_residues": "ION_RESIDUES",
    "elements": "ELEMENTS",
    "element_to_index": "ELEMENT_TO_INDEX",
    "num_element_tokens": "NUM_ELEMENT_TOKENS",
    "af3_crystallization_aids": "AF3_CRYSTALLIZATION_AIDS",
    "af3_ligands_excluded": "AF3_LIGANDS_EXCLUDED",
    "af3_glycans": "AF3_GLYCANS",
    "structure_data_fields": "STRUCTURE_DATA_FIELDS",
    "pad_constants": "PAD_CONSTANTS",
}


def __getattr__(name: str) -> Any:
    """Resolve legacy lowercase constant names.

    Args:
        name: Requested module attribute name.

    Returns:
        Value of the corresponding uppercase constant.

    Raises:
        AttributeError: If the requested name is not a legacy constant.
    """
    constant_name = _LEGACY_CONSTANT_NAMES.get(name)
    if constant_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return globals()[constant_name]


def get_token_index(token: Any) -> Any:
    """Return token index.

    Args:
        token: Token value.

    Returns:
        Result of the get token index operation.
    """
    return TOKEN_TO_INDEX.get(token, TOKEN_TO_INDEX["<UNK>"])


def get_element_index(element: Any) -> Any:
    """Return element index.

    Args:
        element: Element value.

    Returns:
        Result of the get element index operation.
    """
    return ELEMENT_TO_INDEX.get(element, ELEMENT_TO_INDEX["<ATOM_UNK>"])


@dataclasses.dataclass
class StructureData:
    """Implement the structure data component."""

    residue_token: np.ndarray
    residue_index: np.ndarray
    residue_atom_index: np.ndarray
    occupancy: np.ndarray
    bfactor: np.ndarray
    chain_id: np.ndarray
    position: np.ndarray
    element_index: np.ndarray
    # atom_name: np.ndarray
    is_ion: np.ndarray
    is_protein: np.ndarray
    is_nucleotide: np.ndarray
    is_center: np.ndarray
    is_backbone: np.ndarray
    backbone_mask: np.ndarray = dataclasses.field(
        default_factory=lambda: np.array([], dtype=bool)
    )
    not_pad_mask: np.ndarray = dataclasses.field(
        default_factory=lambda: np.array([], dtype=bool)
    )
    interact_non_protein_res: np.ndarray = np.nan
    interact_ion_res: np.ndarray = np.nan
    interact_nucleotide_res: np.ndarray = np.nan
    interact_molecule_res: np.ndarray = np.nan
    is_mask: np.ndarray = np.nan
    noisy_residue_token: np.ndarray = np.nan
    time_step: np.ndarray = np.nan

    def __len__(self) -> int:
        """Return the number of contained items.

        Returns:
            Computed integer value.
        """
        return len(self.residue_token)

    def num_residues(self) -> Any:
        """Execute the num residues operation.

        Returns:
            Result of the num residues operation.
        """
        return len(np.unique(self.residue_index))


@dataclasses.dataclass
class BatchStructureData:
    """Implement the batch structure data component."""

    residue_token: Union[np.ndarray, Tensor]
    residue_index: Union[np.ndarray, Tensor]
    residue_atom_index: Union[np.ndarray, Tensor]
    occupancy: Union[np.ndarray, Tensor]
    bfactor: Union[np.ndarray, Tensor]
    batch_index: Union[np.ndarray, Tensor]
    chain_id: Union[np.ndarray, Tensor]
    position: Union[np.ndarray, Tensor]
    element_index: Union[np.ndarray, Tensor]
    mask_chain: Union[np.ndarray, Tensor]
    # atom_name: np.ndarray
    is_ion: Union[np.ndarray, Tensor]
    is_protein: Union[np.ndarray, Tensor]
    is_nucleotide: Union[np.ndarray, Tensor]
    is_center: Union[np.ndarray, Tensor]
    is_backbone: Union[np.ndarray, Tensor]
    backbone_mask: Union[np.ndarray, Tensor]
    not_pad_mask: Union[np.ndarray, Tensor]
    interact_non_protein_res: Union[np.ndarray, Tensor]
    interact_ion_res: Union[np.ndarray, Tensor]
    interact_nucleotide_res: Union[np.ndarray, Tensor]
    interact_molecule_res: Union[np.ndarray, Tensor]
    noisy_residue_token: Union[np.ndarray, Tensor]
    time_step: Union[np.ndarray, Tensor]

    def __len__(self) -> int:
        """Return the number of contained items.

        Returns:
            Computed integer value.
        """
        return len(self.residue_token)


STRUCTURE_DATA_FIELDS = StructureData.__dataclass_fields__


def slice_structure_data(
    structure_data: StructureData,
    slice_array: np.array,
) -> StructureData:
    """Execute the slice structure data operation.

    Args:
        structure_data: Structure data value.
        slice_array: Slice array value.

    Returns:
        Result of the slice structure data operation.
    """
    new_data = {}
    for key in STRUCTURE_DATA_FIELDS:
        key_obj = getattr(structure_data, key)
        if hasattr(key_obj, "shape") and len(key_obj.shape) > 0:
            if key_obj.shape[0] == slice_array.shape[0]:
                new_data[key] = getattr(structure_data, key)[slice_array]
        else:
            new_data[key] = getattr(structure_data, key)
        if key == "residue_index":
            # new_data[key] -= np.min(structure_data.residue_index)
            new_data[key] = np.unique(new_data[key], return_inverse=True)[1]
    return StructureData(**new_data)


def init_struct_data_dict() -> Any:
    """Execute the init struct data dict operation.

    Returns:
        Result of the init struct data dict operation.
    """
    struct_data = {}
    struct_data["residue_token"] = []
    struct_data["residue_index"] = []
    struct_data["residue_atom_index"] = []
    struct_data["occupancy"] = []
    struct_data["bfactor"] = []
    struct_data["chain_id"] = []
    struct_data["position"] = []
    struct_data["element_index"] = []
    struct_data["is_ion"] = []
    struct_data["is_protein"] = []
    struct_data["is_nucleotide"] = []
    struct_data["is_center"] = []
    struct_data["is_backbone"] = []
    struct_data["backbone_mask"] = []
    struct_data["mask_chain"] = []
    struct_data["not_pad_mask"] = []
    return struct_data


def extend_struct_data_dict(
    struct_data_dict: Dict[str, List[Any]],
    extension_data_dict: Dict[str, List[Any]],
    ligand_center: bool = False,
    ion_center: bool = False,
) -> bool:
    # Check if ligand or ion
    """Execute the extend struct data dict operation.

    Args:
        struct_data_dict: Struct data dict value.
        extension_data_dict: Extension data dict value.
        ligand_center: Ligand center value.
        ion_center: Ion center value.

    Returns:
        Whether the extend struct data dict condition is satisfied.
    """
    if len(extension_data_dict["position"]) == 0:
        return False
    if not (
        extension_data_dict["is_protein"][-1]
        or extension_data_dict["is_nucleotide"][-1]
        or extension_data_dict["is_ion"][-1]
    ):  # for ligand
        positions = np.array(extension_data_dict["position"])
        center_idx = np.argmin(
            np.linalg.norm(
                positions - np.mean(positions, axis=0, keepdims=True), axis=-1
            ),
            axis=0,
        )
        if ligand_center:
            extension_data_dict["is_center"] = [
                i == center_idx for i in range(len(extension_data_dict["position"]))
            ]
            for key in struct_data_dict:
                struct_data_dict[key].extend(extension_data_dict[key])
            return True
        else:
            extension_data_dict["is_center"] = [
                False for i in range(len(extension_data_dict["position"]))
            ]
            for key in struct_data_dict:
                struct_data_dict[key].extend(extension_data_dict[key])
            return False
    elif extension_data_dict["is_ion"][-1]:
        if ion_center:
            for key in struct_data_dict:
                struct_data_dict[key].extend(extension_data_dict[key])
            return True
        else:
            extension_data_dict["is_center"] = [
                False for i in range(len(extension_data_dict["position"]))
            ]
            for key in struct_data_dict:
                struct_data_dict[key].extend(extension_data_dict[key])
            return False
    else:
        if any(extension_data_dict["is_center"]):
            for key in struct_data_dict:
                struct_data_dict[key].extend(extension_data_dict[key])
            return True
    return False


def _normalize_chain_id(chainid: str) -> Any:
    """Execute the normalize chain id operation.

    Args:
        chainid: Chainid value.

    Returns:
        Result of the normalize chain id operation.
    """
    chainid = str(chainid).strip()
    if not chainid or chainid in (".", "?"):
        return None
    return chainid


def _parse_mmcif_assemblies(path_or_name: str) -> Any:
    """Parse mmcif assemblies.

    Args:
        path_or_name: Path or name value.

    Returns:
        Result of the parse mmcif assemblies operation.
    """
    if not os.path.isfile(path_or_name):
        return None

    if path_or_name.endswith(".gz"):
        opener = lambda p: gzip.open(p, "rt", errors="ignore")
    else:
        opener = lambda p: open(p, "rt", errors="ignore")

    try:
        with opener(path_or_name) as f:
            cif = MMCIF2Dict(f)
    except Exception:
        return None

    def _to_list(val: Any, n: Optional[Any] = None) -> Any:
        """Execute the to list operation.

        Args:
            val: Val value.
            n: N value.

        Returns:
            Result of the to list operation.
        """
        if val is None:
            return [] if n is None else ["" for _ in range(n)]
        if isinstance(val, str):
            return [val]
        return list(val)

    asmb_ids = _to_list(cif.get("_pdbx_struct_assembly_gen.assembly_id"))
    if not asmb_ids:
        return None

    asmb_chains = _to_list(
        cif.get("_pdbx_struct_assembly_gen.asym_id_list"), len(asmb_ids)
    )

    return {
        "asmb_ids": asmb_ids,
        "asmb_chains": asmb_chains,
    }


def parse_structure(path_or_name: str) -> Any:
    """Parse structure.

    Args:
        path_or_name: Path or name value.

    Returns:
        Result of the parse structure operation.
    """
    if ".pdb" in path_or_name:
        parser = PDBParser(QUIET=True)
        return parser.get_structure("structure", path_or_name)
    parser = AllAltlocMMCIFParser(QUIET=True, auth_chains=False)
    if path_or_name.endswith(".gz"):
        with gzip.open(path_or_name, "rt", errors="ignore") as f:
            return parser.get_structure("structure", f)
    return parser.get_structure("structure", path_or_name)


def _normalize_duplicate_residue_altlocs(
    mmcif_dict: Dict[str, List[str]],
    auth_chains: bool = True,
    auth_residues: bool = True,
) -> None:
    # Biopython rejects point-mutated residues if one residue alternative has
    # blank atom altlocs. Normalize those blanks before StructureBuilder runs.
    """Execute the normalize duplicate residue altlocs operation.

    Args:
        mmcif_dict: Mmcif dict value.
        auth_chains: Auth chains value.
        auth_residues: Auth residues value.
    """
    alt_ids = mmcif_dict.get("_atom_site.label_alt_id")
    comp_ids = mmcif_dict.get("_atom_site.label_comp_id")
    if not alt_ids or not comp_ids:
        return

    # MMCIF2Dict values may be immutable-ish parser-owned lists; replace the
    # atom-site altloc column with a mutable copy that Biopython will read.
    alt_ids = list(alt_ids)
    mmcif_dict["_atom_site.label_alt_id"] = alt_ids

    # Use the same chain and residue-id columns that MMCIFParser will use.
    if auth_chains and "_atom_site.auth_asym_id" in mmcif_dict:
        chain_ids = mmcif_dict["_atom_site.auth_asym_id"]
    else:
        chain_ids = mmcif_dict["_atom_site.label_asym_id"]

    if auth_residues and "_atom_site.auth_seq_id" in mmcif_dict:
        seq_ids = mmcif_dict["_atom_site.auth_seq_id"]
    else:
        seq_ids = mmcif_dict["_atom_site.label_seq_id"]

    icode_ids = mmcif_dict.get("_atom_site.pdbx_PDB_ins_code", ["?"] * len(comp_ids))
    group_ids = mmcif_dict.get("_atom_site.group_PDB", ["ATOM"] * len(comp_ids))
    model_ids = mmcif_dict.get("_atom_site.pdbx_PDB_model_num", [""] * len(comp_ids))
    groups = {}

    # Group atoms by the residue identity Biopython uses before considering
    # residue name. Different comp_ids in one group indicate residue disorder.
    for i, comp_id in enumerate(comp_ids):
        hetatm_flag = " "
        if group_ids[i] == "HETATM":
            hetatm_flag = "W" if comp_id in {"HOH", "WAT"} else "H"
        icode = " " if icode_ids[i] in {".", "?"} else icode_ids[i]
        key = (model_ids[i], chain_ids[i], seq_ids[i], icode, hetatm_flag)
        groups.setdefault(key, []).append(i)

    for atom_indices in groups.values():
        explicit_by_comp = {}
        for i in atom_indices:
            alt_id = alt_ids[i]
            if alt_id not in {".", "?", " "}:
                explicit_by_comp.setdefault(comp_ids[i], []).append(alt_id)
        if len(explicit_by_comp) < 2:
            continue

        # Fill blank altlocs only inside duplicate-residue groups. Prefer the
        # altloc already used by that residue name; otherwise use any group alt.
        first_alt = next(iter(explicit_by_comp.values()))[0]
        for i in atom_indices:
            if alt_ids[i] in {".", "?", " "}:
                alt_ids[i] = explicit_by_comp.get(comp_ids[i], [first_alt])[0]


class AllAltlocMMCIFParser(MMCIFParser):
    """Implement the all altloc mmcifparser component."""

    def _build_structure(self, structure_id: Any) -> Any:
        """Build structure.

        Args:
            structure_id: Structure id value.

        Returns:
            Result of the build structure operation.
        """
        _normalize_duplicate_residue_altlocs(
            self._mmcif_dict,
            auth_chains=self.auth_chains,
            auth_residues=self.auth_residues,
        )
        return super()._build_structure(structure_id)


class _NonWaterHydrogenSelect(Select):
    """PDBIO selector that skips water and hydrogen, consistent with parse_mmcif_to_structure_data."""

    _WATER = {"HOH", "WAT", "DOD"}

    def accept_residue(self, residue: Any) -> bool:
        """Execute the accept residue operation.

        Args:
            residue: Residue value.

        Returns:
            Whether the accept residue condition is satisfied.
        """
        return residue.get_resname().strip() not in self._WATER

    def accept_atom(self, atom: Any) -> bool:
        """Execute the accept atom operation.

        Args:
            atom: Atom value.

        Returns:
            Whether the accept atom condition is satisfied.
        """
        element = (atom.element or "").strip().upper()
        if element in {"H", "D"} or atom.get_name().startswith("H"):
            return 0
        return 1


def cif_to_pdb(cif_path: str, pdb_path: str) -> None:
    """Convert mmCIF to PDB using Biopython, consistent with parse_structure.

    Uses the same MMCIFParser(auth_chains=False) so the resulting PDB
    contains exactly the same chains/residues that parse_mmcif_to_structure_data sees.

    Handles:
    - Multi-character chain IDs -> remapped to single-character PDB chain IDs
    - Water-only chains -> removed (reduces chain count for PDB format limit)
    - Long residue names (>3 chars) -> truncated to 3 chars for PDB compatibility

    Args:
        cif_path: Path for cif.
        pdb_path: Path for pdb.
    """
    structure = parse_structure(cif_path)

    # Single-char chain IDs: A-Z, 0-9, a-z (total 62)
    pdb_chain_chars = (
        [chr(i) for i in range(ord("A"), ord("Z") + 1)]
        + [chr(i) for i in range(ord("0"), ord("9") + 1)]
        + [chr(i) for i in range(ord("a"), ord("z") + 1)]
    )

    model = next(structure.get_models())
    chains = list(model.get_chains())

    # Detach all chains first to avoid ID collision during renaming
    for chain in chains:
        model.detach_child(chain.id)

    # Filter out water-only chains (they are skipped by _NonWaterHydrogenSelect anyway)
    water_resnames = {"HOH", "WAT", "DOD"}
    non_water_chains = []
    for chain in chains:
        has_non_water = any(
            r.get_resname().strip() not in water_resnames for r in chain.get_residues()
        )
        if has_non_water:
            non_water_chains.append(chain)

    if len(non_water_chains) > len(pdb_chain_chars):
        raise ValueError(
            f"Structure has {len(non_water_chains)} non-water chains, "
            f"exceeding PDB format limit of {len(pdb_chain_chars)} single-char chain IDs."
        )

    # Re-add with single-char IDs; also fix long residue names
    for i, chain in enumerate(non_water_chains):
        chain.id = pdb_chain_chars[i]
        for res in chain:
            # Truncate residue names > 3 chars (PDB format limit)
            resname = res.resname.strip()
            if len(resname) > 3:
                res.resname = resname[:3]
        model.add(chain)

    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_path, _NonWaterHydrogenSelect())


def parse_mmcif_to_structure_data(
    path_or_name: str,
    parser_chain_id: Optional[str] = None,
    ion_center: bool = False,
    ligand_center: bool = False,
) -> StructureData:
    """Parse a mmCIF file and return a StructureData object.

    Args:
        path_or_name: Path or name value.
        parser_chain_id: Parser chain id value.
        ion_center: Ion center value.
        ligand_center: Ligand center value.

    Returns:
        Result of the parse mmcif to structure data operation.
    """
    protein_residues = get_protein_residues()

    # ------------ load & pre-filter atoms ------------
    structure = parse_structure(path_or_name)
    model = next(structure.get_models())

    # ------------ init containers ------------
    struct_data = init_struct_data_dict()
    internal_residue_index = 0
    internal_chain_index = -1
    chain_id_map = {}

    water_resnames = {"HOH", "WAT", "DOD"}

    for chain in model:
        chainid = _normalize_chain_id(chain.id)
        if chainid is None:
            continue
        if parser_chain_id is not None and chainid not in parser_chain_id:
            continue

        internal_chain_index += 1
        chain_id_map[chainid] = internal_chain_index

        for residue in chain.get_unpacked_list():
            resname = residue.get_resname().strip()
            if resname in water_resnames:
                continue
            if resname in AF3_CRYSTALLIZATION_AIDS:
                continue

            temp_residue_data = init_struct_data_dict()
            residue_atom_count = 0
            backbone_atoms_present = set()

            is_ion_res = resname in ION_RESIDUES
            is_protein_res = resname in protein_residues
            is_nucleotide_res = resname in NUCLEOTIDE_RESIDUES

            for atom in residue.get_unpacked_list():
                element = (atom.element or "").strip().upper()
                atom_name = atom.get_name()
                if element in {"H", "D"} or atom_name.startswith("H"):
                    continue

                if resname in AF3_GLYCANS:
                    resname_use = "<GLYCAN>"
                else:
                    resname_use = resname

                temp_residue_data["residue_token"].append(get_token_index(resname_use))
                temp_residue_data["residue_index"].append(internal_residue_index)
                temp_residue_data["residue_atom_index"].append(residue_atom_count)
                temp_residue_data["occupancy"].append(atom.get_occupancy() or 0.0)
                temp_residue_data["bfactor"].append(atom.get_bfactor() or 0.0)
                temp_residue_data["chain_id"].append(internal_chain_index)
                temp_residue_data["position"].append(atom.get_coord())
                temp_residue_data["element_index"].append(get_element_index(element))
                temp_residue_data["is_ion"].append(is_ion_res)
                temp_residue_data["is_protein"].append(is_protein_res)
                temp_residue_data["is_nucleotide"].append(is_nucleotide_res)
                temp_residue_data["not_pad_mask"].append(True)
                temp_residue_data["is_backbone"].append(
                    is_protein_res and atom_name in ["N", "CA", "C", "O"]
                )

                if is_protein_res and atom_name in ["N", "CA", "C", "O"]:
                    backbone_atoms_present.add(atom_name)

                if is_protein_res:
                    temp_residue_data["is_center"].append(atom_name == "CA")
                elif is_nucleotide_res:
                    temp_residue_data["is_center"].append(atom_name == "C1'")
                elif is_ion_res:
                    temp_residue_data["is_center"].append(
                        ion_center and residue_atom_count == 0
                    )
                else:
                    temp_residue_data["is_center"].append(
                        ligand_center and residue_atom_count == 0
                    )

                residue_atom_count += 1

            if len(temp_residue_data["position"]) == 0:
                continue

            backbone_complete = is_protein_res and all(
                x in backbone_atoms_present for x in ["N", "CA", "C", "O"]
            )
            temp_residue_data["backbone_mask"] = [
                backbone_complete for _ in range(len(temp_residue_data["position"]))
            ]

            if extend_struct_data_dict(
                struct_data,
                temp_residue_data,
                ligand_center=ligand_center,
                ion_center=ion_center,
            ):
                internal_residue_index += 1

    for key in struct_data:
        struct_data[key] = np.asarray(struct_data[key])

    struct_data["residue_index"] -= struct_data["residue_index"].min()

    if struct_data["mask_chain"].shape[0] == 0:
        del struct_data["mask_chain"]

    data = StructureData(**struct_data)

    assembly_info = _parse_mmcif_assemblies(path_or_name)
    if assembly_info is not None:
        asmb_chain_indices = []
        for chains in assembly_info.get("asmb_chains", []):
            idxs = []
            for c in str(chains).split(","):
                c_norm = _normalize_chain_id(c)
                if c_norm is None:
                    continue
                if c_norm in chain_id_map:
                    idxs.append(chain_id_map[c_norm])
            asmb_chain_indices.append(idxs)

        data.asmb_ids = assembly_info.get("asmb_ids", [])
        data.asmb_chains = assembly_info.get("asmb_chains", [])
        data.asmb_chain_indices = asmb_chain_indices
        data.asym_id_to_chain_index = chain_id_map

    return data


def mask_structure_data(struct_data: StructureData, mask_array: np.ndarray) -> Any:
    """Mask the structure data with a boolean array.

    Args:
        struct_data: Struct data value.
        mask_array: Boolean mask for mask array.

    Returns:
        Result of the mask structure data operation.
    """
    # Masking the data consists of putting in <MASK> tokens for the protein residues
    # And removing the non-backbone atoms for the masked residues

    # The mask array is the same length as the numbe of residues
    # We need to calculate the atom_mask_array
    # This is a boolean array of the same length as the number of atoms

    masked_residues = np.nonzero(mask_array)[0]
    atom_mask_array = np.isin(struct_data.residue_index, masked_residues)
    atom_mask_array = np.logical_and(
        atom_mask_array, np.logical_not(struct_data.is_backbone)
    )
    atom_keep_array = np.logical_not(atom_mask_array)

    struct_data.residue_token[atom_mask_array] = TOKEN_TO_INDEX["<MASK>"]

    masked_data = {}
    for key in STRUCTURE_DATA_FIELDS:
        masked_data[key] = struct_data.__dict__[key][atom_keep_array]
    return StructureData(**masked_data)


def get_closest_n_residues(
    struct_data: StructureData, position: np.ndarray, n: int
) -> StructureData:
    """Get the n closest residues to a position.

    Args:
        struct_data: Struct data value.
        position: Position value.
        n: N value.

    Returns:
        Result of the get closest n residues operation.
    """
    # Going to pick residues by center atom
    center_atoms = struct_data.position[struct_data.is_center]
    kdtree = cKDTree(center_atoms)
    _, closest_residue_indices = kdtree.query(position, n)
    closest_residue_indices = np.sort(closest_residue_indices)
    # Now remove residues not selected
    keep_array = np.isin(struct_data.residue_index, closest_residue_indices)

    return slice_structure_data(struct_data, keep_array)


def get_closest_n_atoms(
    struct_data: StructureData,
    position: np.ndarray,
    n: int,
    remove_incomplete_residues: bool = True,
) -> StructureData:
    """Get the n closest atoms to a position.
    Throw out atoms that are partially picked from a residue

    Args:
        struct_data: Struct data value.
        position: Position value.
        n: N value.
        remove_incomplete_residues: Remove incomplete residues value.

    Returns:
        Result of the get closest n atoms operation.
    """
    kdtree = cKDTree(struct_data.position)
    _, closest_atom_indices = kdtree.query(position, n)
    closest_atom_indices = np.sort(closest_atom_indices)
    closest_atom_mask = np.zeros(len(struct_data.position), dtype=bool)
    closest_atom_mask[closest_atom_indices] = True
    if remove_incomplete_residues:
        residues_to_throw = np.unique(struct_data.residue_index[~closest_atom_mask])
        atoms_to_throw = np.isin(struct_data.residue_index, residues_to_throw)
        closest_atom_mask[atoms_to_throw] = False

    return slice_structure_data(struct_data, closest_atom_mask)


PAD_CONSTANTS = {
    "residue_token": TOKEN_TO_INDEX["<PAD>"],
    "residue_index": -1,
    "residue_atom_index": -1,
    "occupancy": 0.0,
    "bfactor": 0.0,
    "chain_id": -1,
    "position": 0.0,
    "element_index": ELEMENT_TO_INDEX["<ATOM_PAD>"],
    "atom_name": "",
    "is_ion": False,
    "is_protein": False,
    "is_nucleotide": False,
    "is_center": False,
    "is_backbone": False,
    "backbone_mask": False,
    "not_pad_mask": False,
    "is_mask": False,
    "interact_non_protein_res": False,
    "interact_ion_res": False,
    "interact_nucleotide_res": False,
    "interact_molecule_res": False,
    "noisy_residue_token": TOKEN_TO_INDEX["<PAD>"],
    "time_step": -1,
}


def pad_structure_data(struct_data: StructureData, pad_length: int) -> Any:
    """Pad a structure data to a certain length.

    Args:
        struct_data: Struct data value.
        pad_length: Pad length value.

    Returns:
        Result of the pad structure data operation.
    """
    pad_data = {}
    for key in STRUCTURE_DATA_FIELDS:
        if key != "position" and key != "time_step":
            pad_data[key] = np.pad(
                struct_data.__dict__[key],
                (0, pad_length - len(struct_data)),
                mode="constant",
                constant_values=PAD_CONSTANTS[key],
            )
        elif key == "time_step":
            pad_data[key] = struct_data.__dict__[key]
        else:
            pad_values = np.zeros((pad_length - len(struct_data), 3))
            pad_data[key] = np.concatenate(
                [struct_data.__dict__[key], pad_values], axis=0
            )
    if "mask_chain" in struct_data.__dict__.keys():
        pad_mask_chain = np.pad(
            struct_data.__dict__["mask_chain"],
            (0, pad_length - len(struct_data)),
            mode="constant",
            constant_values=PAD_CONSTANTS["not_pad_mask"],
        )
    data = StructureData(**pad_data)
    data.mask_chain = pad_mask_chain
    return data


def from_numpy(x_array: np.array) -> torch.Tensor:
    """Execute the from numpy operation.

    Args:
        x_array: X array value.

    Returns:
        Computed tensor values.
    """
    if x_array.dtype == np.float64:
        x_array = x_array.astype(np.float32)
    return torch.from_numpy(x_array)


def batch_structure_data_list(
    struct_data_list: List[StructureData],
    pad_length: int = None,
    to_torch: bool = False,
) -> BatchStructureData:
    """Batch a list of structure data.

    Args:
        struct_data_list: Struct data list value.
        pad_length: Pad length value.
        to_torch: To torch value.

    Returns:
        Result of the batch structure data list operation.
    """
    try:
        struct_data_list = [x for x in struct_data_list if x is not None]
        if len(struct_data_list) == 0:
            return None
        # struct_data_list = [struct_data for struct_data in struct_data_list if struct_data.is_center.sum() <= 512 and np.unique(struct_data.residue_index).shape[0] == struct_data.is_center.sum()]

        stack_fn = np.stack if not to_torch else torch.stack
        full_fn = np.full if not to_torch else torch.full
        convert_fn = (lambda x: x) if not to_torch else from_numpy

        if pad_length is None:
            pad_length = max(len(struct_data) for struct_data in struct_data_list)

        batch_data = {}
        new_struct_data_list = []
        for struct_data in struct_data_list:
            if len(struct_data) < pad_length:
                new_struct_data_list.append(pad_structure_data(struct_data, pad_length))
            else:
                new_struct_data_list.append(struct_data)

        for key in STRUCTURE_DATA_FIELDS:
            if key not in BatchStructureData.__dataclass_fields__ or key in [
                "time_step"
            ]:
                continue
            batch_data[key] = stack_fn(
                [
                    convert_fn(struct_data.__dict__[key])
                    for struct_data in new_struct_data_list
                ]
            )
        batch_data["mask_chain"] = stack_fn(
            [convert_fn(struct_data.mask_chain) for struct_data in new_struct_data_list]
        )

        batch_data["batch_index"] = stack_fn(
            [full_fn((pad_length,), i) for i in range(len(new_struct_data_list))]
        )
        batch_data["time_step"] = stack_fn(
            [convert_fn(struct_data.time_step) for struct_data in new_struct_data_list]
        )
        return BatchStructureData(**batch_data)

    except Exception as e:
        print(e)
        return None


def dump_structure_data(struct_data: StructureData, path: str) -> None:
    """Dump a structure data to a file.

    Args:
        struct_data: Struct data value.
        path: Path value.
    """

    def _to_saveable(value: Any) -> Any:
        """Execute the to saveable operation.

        Args:
            value: Value value.

        Returns:
            Result of the to saveable operation.
        """
        if isinstance(value, dict):
            return np.array([value], dtype=object)
        if isinstance(value, list):
            try:
                with np.warnings.catch_warnings():
                    np.warnings.simplefilter("ignore", np.VisibleDeprecationWarning)
                    arr = np.array(value)
                return arr
            except Exception:
                return np.array(value, dtype=object)
        return value

    data = {k: _to_saveable(v) for k, v in struct_data.__dict__.items()}
    np.savez_compressed(path, **data)


def load_structure_data(path: str) -> StructureData:
    """Load a structure data to a file.

    Args:
        path: Path value.

    Returns:
        Result of the load structure data operation.
    """
    struct_data = np.load(path, allow_pickle=True)
    field_names = set(STRUCTURE_DATA_FIELDS.keys())
    data_kwargs = {k: struct_data[k] for k in struct_data.files if k in field_names}
    # Backward compatibility: older .npz may not have backbone_mask
    if "backbone_mask" not in data_kwargs:
        if "residue_index" in data_kwargs and "is_backbone" in data_kwargs:
            residue_index = data_kwargs["residue_index"]
            is_backbone = data_kwargs["is_backbone"]
            backbone_mask = np.zeros_like(is_backbone, dtype=bool)
            for rid in np.unique(residue_index):
                sel = residue_index == rid
                # residue is complete if it has all 4 backbone atoms
                if np.sum(is_backbone[sel]) >= 4:
                    backbone_mask[sel] = True
            data_kwargs["backbone_mask"] = backbone_mask
        elif "is_backbone" in data_kwargs:
            data_kwargs["backbone_mask"] = np.zeros_like(
                data_kwargs["is_backbone"], dtype=bool
            )
    data = StructureData(**data_kwargs)
    for k in struct_data.files:
        if k not in field_names:
            val = struct_data[k]
            if isinstance(val, np.ndarray) and val.dtype.kind == "O" and val.size == 1:
                val = val.item()
            setattr(data, k, val)
    return data


def parse_or_load_mmcif(
    path_or_name: str,
    parser_chain_id: Optional[str] = None,
    ion_center: bool = False,
    ligand_center: bool = False,
) -> StructureData:
    """Parse or load mmcif.

    Args:
        path_or_name: Path or name value.
        parser_chain_id: Parser chain id value.
        ion_center: Ion center value.
        ligand_center: Ligand center value.

    Returns:
        Result of the parse or load mmcif operation.
    """
    if os.path.splitext(path_or_name)[1] == ".npz":
        return load_structure_data(path_or_name)
    else:
        return parse_mmcif_to_structure_data(
            path_or_name,
            parser_chain_id,
            ion_center=ion_center,
            ligand_center=ligand_center,
        )


def get_example_batch() -> Any:
    """Return example batch.

    Returns:
        Result of the get example batch operation.
    """
    struct_data1 = parse_mmcif_to_structure_data("5a00")
    struct_data2 = parse_mmcif_to_structure_data("7npm")
    struct_data3 = parse_mmcif_to_structure_data("dataset/test_nucleotide/1bc7.pdb")
    struct_data_list = [struct_data1, struct_data2, struct_data3]
    batch_data = batch_structure_data_list(struct_data_list)
    return batch_data


def propagate_mask_vectorized(mask: Any, index: int) -> Any:
    # Ensure inputs are tensors
    """Execute the propagate mask vectorized operation.

    Args:
        mask: Boolean mask for mask.
        index: Index value.

    Returns:
        Result of the propagate mask vectorized operation.
    """
    mask = torch.as_tensor(mask, dtype=torch.bool)
    index = torch.as_tensor(index)

    # Get unique indices and their inverse mapping
    unique_indices, inverse_indices = torch.unique(index, return_inverse=True)

    # Create a tensor to hold the "any True" status for each unique index
    any_true = torch.zeros_like(unique_indices, dtype=torch.bool)

    # Use scatter_reduce to check if any mask is True for each unique index
    any_true.scatter_reduce_(0, inverse_indices, mask, reduce="amax")

    # Expand the any_true tensor to match the original shape
    return any_true[inverse_indices]


def interact_residue(data: Any) -> Any:
    """Execute the interact residue operation.

    Args:
        data: Data value.

    Returns:
        Result of the interact residue operation.
    """
    protein_pos = np.expand_dims(data.position[data.is_protein], axis=1)
    non_protein_pos = np.expand_dims(data.position[~data.is_protein], axis=0)
    ion_pos = np.expand_dims(data.position[data.is_ion], axis=0)
    nucleotide_pos = np.expand_dims(data.position[data.is_nucleotide], axis=0)
    molecule_pos = np.expand_dims(
        data.position[~data.is_protein & ~data.is_ion & ~data.is_nucleotide], axis=0
    )

    dist_non_protein = np.sqrt((np.sum((protein_pos - non_protein_pos) ** 2, axis=-1)))
    interact_non_protein = np.any(dist_non_protein < 5, axis=1)
    dist_ion = np.sqrt((np.sum((protein_pos - ion_pos) ** 2, axis=-1)))
    interact_ion = np.any(dist_ion < 5, axis=1)
    dist_nucleotide = np.sqrt((np.sum((protein_pos - nucleotide_pos) ** 2, axis=-1)))
    interact_nucleotide = np.any(dist_nucleotide < 5, axis=1)
    dist_molecule = np.sqrt((np.sum((protein_pos - molecule_pos) ** 2, axis=-1)))
    interact_molecule = np.any(dist_molecule < 5, axis=1)

    interact_non_protein_res = propagate_mask_vectorized(
        interact_non_protein, data.residue_index[data.is_protein]
    )
    interact_ion_res = propagate_mask_vectorized(
        interact_ion, data.residue_index[data.is_protein]
    )
    interact_nucleotide_res = propagate_mask_vectorized(
        interact_nucleotide, data.residue_index[data.is_protein]
    )
    interact_molecule_res = propagate_mask_vectorized(
        interact_molecule, data.residue_index[data.is_protein]
    )

    data.interact_non_protein_res = interact_non_protein_res.numpy()
    data.interact_ion_res = interact_ion_res.numpy()
    data.interact_nucleotide_res = interact_nucleotide_res.numpy()
    data.interact_molecule_res = interact_molecule_res.numpy()

    return data


def pdb2data(
    pdb_file: str,
    device: str = "cpu",
    ligand_center: bool = False,
    ion_center: bool = False,
) -> Any:
    """Execute the pdb2data operation.

    Args:
        pdb_file: Path for pdb.
        device: Device used for tensor operations.
        ligand_center: Ligand center value.
        ion_center: Ion center value.

    Returns:
        Result of the pdb2data operation.
    """
    data = parse_mmcif_to_structure_data(
        pdb_file, ligand_center=ligand_center, ion_center=ion_center
    )
    data = interact_residue(data)
    result = {}
    for k, v in data.__dict__.items():
        if isinstance(v, np.ndarray):
            result[k] = torch.from_numpy(v).to(device)
        elif isinstance(v, (int, float)):
            result[k] = torch.Tensor([v])
        # skip non-tensor fields (assembly info lists/dicts)
    result["batch_index"] = torch.zeros_like(result["residue_index"])
    result = {
        k: v.float() if isinstance(v, torch.Tensor) and v.dtype == torch.float64 else v
        for k, v in result.items()
    }
    return result


def _switch_pdb_order_add_ligand(new_pdb_path: str, original_pdb_path: str) -> None:
    """new_pdb_path is the sidechain packing pdb
    original_pdb_path is the original pdb

    Args:
        new_pdb_path: Path for new pdb.
        original_pdb_path: Path for original pdb.
    """
    structure = prody.parsePDB(new_pdb_path)
    protein_atom = structure.select("protein")

    structure2 = prody.parsePDB(original_pdb_path)
    ligand_atom = structure2.select("not protein and not water and not hydrogen")

    merged = protein_atom.toAtomGroup() + ligand_atom.toAtomGroup()

    prody.proteins.pdbfile.writePDB(new_pdb_path, merged)


def switch_pdb_order_add_ligand(
    new_pdb_path: str, original_pdb_path: str, confidence: Optional[np.ndarray] = None
) -> None:
    """Execute the switch pdb order add ligand operation.

    Args:
        new_pdb_path: Path for new pdb.
        original_pdb_path: Path for original pdb.
        confidence: Confidence value.
    """
    new_lines = []
    temp_line = None
    ligand_lines = []

    # Read the original PDB to extract ligand information
    with open(original_pdb_path, "r") as f:
        for line in f:
            if line.startswith("HETATM"):
                ligand_lines.append(line)

    # Process the new PDB file
    current_residue = None
    current_chain = None
    confidence_index = -1
    with open(new_pdb_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("ATOM"):
                residue_number = int(
                    line[22:26]
                    .strip()
                    .rstrip("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
                    or "0"
                )
                chain_id = line[21]
                if residue_number != current_residue or chain_id != current_chain:
                    current_residue = residue_number
                    current_chain = chain_id
                    if confidence is not None:
                        confidence_index = (confidence_index + 1) % len(confidence)

                atom_name = line[12:16].strip()
                if atom_name == "CB":
                    atom_index = int(line[6:11])
                    temp_line = line[:6] + f"{atom_index + 1:5}" + line[11:]
                    if confidence is not None:
                        temp_line = (
                            temp_line[:60]
                            + f"{confidence[confidence_index]:6.2f}"
                            + temp_line[66:]
                        )
                elif atom_name == "O":
                    atom_index = int(line[6:11])
                    new_line = line[:6] + f"{atom_index - 1:5}" + line[11:]
                    if confidence is not None:
                        new_line = (
                            new_line[:60]
                            + f"{confidence[confidence_index]:6.2f}"
                            + new_line[66:]
                        )
                    new_lines.append(new_line)
                    if temp_line is not None:
                        new_lines.append(temp_line)
                    temp_line = None
                else:
                    new_line = line
                    if confidence is not None:
                        new_line = (
                            new_line[:60]
                            + f"{confidence[confidence_index]:6.2f}"
                            + new_line[66:]
                        )
                    new_lines.append(new_line)
            elif line.startswith("TER"):
                new_lines.append(line)
            elif line.startswith("END"):
                new_lines.append(line)
                new_lines.extend(ligand_lines)
                break
            else:
                new_lines.append(line)

    # Write the new PDB file
    with open(new_pdb_path, "w") as f:
        f.writelines(new_lines)


def make_merged_pdb(original_pdb_path: str, sc_packing_pdb_path: str) -> None:
    """Create merged pdb.

    Args:
        original_pdb_path: Path for original pdb.
        sc_packing_pdb_path: Path for sc packing pdb.
    """
    og_pdb = prody.parsePDB(original_pdb_path)
    sc_pdb = prody.parsePDB(sc_packing_pdb_path)

    iter_og_pdb = og_pdb.iterResidues()
    iter_sc_pdb = sc_pdb.iterResidues()
    internal_residue_num = 1

    new_residues = []

    while True:
        try:
            og_residue = next(iter_og_pdb)
        except StopIteration:
            break
        is_protein = og_residue.select("not protein") is None
        if is_protein:
            updated_residue = next(iter_sc_pdb)
        else:
            updated_residue = og_residue
        updated_atom_group = updated_residue.toAtomGroup()
        updated_atom_group.setCoords(updated_residue.getCoords())
        updated_atom_group.setResnums(
            [internal_residue_num] * len(updated_atom_group.getCoords())
        )
        new_residues.append(updated_atom_group)
        internal_residue_num += 1

    new_atom_group = sum(
        [residue for residue in new_residues[1:]], start=new_residues[0]
    )
    new_atom_group.setTitle(og_pdb.getTitle())
    prody.proteins.writePDB(sc_packing_pdb_path, new_atom_group)
