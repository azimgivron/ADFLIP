from __future__ import annotations

from importlib import resources
from typing import Any, Tuple

import numpy as np

#            0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18   19
RESTYPES = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]
RESTYPES_WITH_X = RESTYPES + ["X"]
RESTYPE_ORDER = {restype: i for i, restype in enumerate(RESTYPES)}
RESTYPE_NUM = len(RESTYPES)  # := 20.
UNK_RESTYPE_INDEX = RESTYPE_NUM

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
}
RESTYPE_3TO1 = {v: k for k, v in RESTYPE_1TO3.items()}

# Atoms positions relative to the 8 rigid groups, defined by the pre-omega, phi,
# psi and chi angles:
# 0: 'backbone group',
# 1: 'pre-omega-group', (empty)
# 2: 'phi-group', (currently empty, because it defines only hydrogens)
# 3: 'psi-group',
# 4,5,6,7: 'chi1,2,3,4-group'
# The atom positions are relative to the axis-end-atom of the corresponding
# rotation axis. The x-axis is in direction of the rotation axis, and the y-axis
# is defined such that the dihedral-angle-definiting atom (the last entry in
# chi_angles_atoms above) is in the xy-plane (with a positive y-coordinate).
# format: [atomname, group_idx, rel_position]
RIGID_GROUP_ATOM_POSITIONS = {
    "ALA": [
        ["N", 0, (-0.525, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, -0.000, -0.000)],
        ["CB", 0, (-0.529, -0.774, -1.205)],
        ["O", 3, (0.627, 1.062, 0.000)],
    ],
    "ARG": [
        ["N", 0, (-0.524, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.524, -0.778, -1.209)],
        ["O", 3, (0.626, 1.062, 0.000)],
        ["CG", 4, (0.616, 1.390, -0.000)],
        ["CD", 5, (0.564, 1.414, 0.000)],
        ["NE", 6, (0.539, 1.357, -0.000)],
        ["NH1", 7, (0.206, 2.301, 0.000)],
        ["NH2", 7, (2.078, 0.978, -0.000)],
        ["CZ", 7, (0.758, 1.093, -0.000)],
    ],
    "ASN": [
        ["N", 0, (-0.536, 1.357, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, -0.000, -0.000)],
        ["CB", 0, (-0.531, -0.787, -1.200)],
        ["O", 3, (0.625, 1.062, 0.000)],
        ["CG", 4, (0.584, 1.399, 0.000)],
        ["ND2", 5, (0.593, -1.188, 0.001)],
        ["OD1", 5, (0.633, 1.059, 0.000)],
    ],
    "ASP": [
        ["N", 0, (-0.525, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, 0.000, -0.000)],
        ["CB", 0, (-0.526, -0.778, -1.208)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.593, 1.398, -0.000)],
        ["OD1", 5, (0.610, 1.091, 0.000)],
        ["OD2", 5, (0.592, -1.101, -0.003)],
    ],
    "CYS": [
        ["N", 0, (-0.522, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.524, 0.000, 0.000)],
        ["CB", 0, (-0.519, -0.773, -1.212)],
        ["O", 3, (0.625, 1.062, -0.000)],
        ["SG", 4, (0.728, 1.653, 0.000)],
    ],
    "GLN": [
        ["N", 0, (-0.526, 1.361, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, 0.000, 0.000)],
        ["CB", 0, (-0.525, -0.779, -1.207)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.615, 1.393, 0.000)],
        ["CD", 5, (0.587, 1.399, -0.000)],
        ["NE2", 6, (0.593, -1.189, -0.001)],
        ["OE1", 6, (0.634, 1.060, 0.000)],
    ],
    "GLU": [
        ["N", 0, (-0.528, 1.361, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, -0.000, -0.000)],
        ["CB", 0, (-0.526, -0.781, -1.207)],
        ["O", 3, (0.626, 1.062, 0.000)],
        ["CG", 4, (0.615, 1.392, 0.000)],
        ["CD", 5, (0.600, 1.397, 0.000)],
        ["OE1", 6, (0.607, 1.095, -0.000)],
        ["OE2", 6, (0.589, -1.104, -0.001)],
    ],
    "GLY": [
        ["N", 0, (-0.572, 1.337, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.517, -0.000, -0.000)],
        ["O", 3, (0.626, 1.062, -0.000)],
    ],
    "HIS": [
        ["N", 0, (-0.527, 1.360, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, 0.000, 0.000)],
        ["CB", 0, (-0.525, -0.778, -1.208)],
        ["O", 3, (0.625, 1.063, 0.000)],
        ["CG", 4, (0.600, 1.370, -0.000)],
        ["CD2", 5, (0.889, -1.021, 0.003)],
        ["ND1", 5, (0.744, 1.160, -0.000)],
        ["CE1", 5, (2.030, 0.851, 0.002)],
        ["NE2", 5, (2.145, -0.466, 0.004)],
    ],
    "ILE": [
        ["N", 0, (-0.493, 1.373, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, -0.000, -0.000)],
        ["CB", 0, (-0.536, -0.793, -1.213)],
        ["O", 3, (0.627, 1.062, -0.000)],
        ["CG1", 4, (0.534, 1.437, -0.000)],
        ["CG2", 4, (0.540, -0.785, -1.199)],
        ["CD1", 5, (0.619, 1.391, 0.000)],
    ],
    "LEU": [
        ["N", 0, (-0.520, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.522, -0.773, -1.214)],
        ["O", 3, (0.625, 1.063, -0.000)],
        ["CG", 4, (0.678, 1.371, 0.000)],
        ["CD1", 5, (0.530, 1.430, -0.000)],
        ["CD2", 5, (0.535, -0.774, 1.200)],
    ],
    "LYS": [
        ["N", 0, (-0.526, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, 0.000, 0.000)],
        ["CB", 0, (-0.524, -0.778, -1.208)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.619, 1.390, 0.000)],
        ["CD", 5, (0.559, 1.417, 0.000)],
        ["CE", 6, (0.560, 1.416, 0.000)],
        ["NZ", 7, (0.554, 1.387, 0.000)],
    ],
    "MET": [
        ["N", 0, (-0.521, 1.364, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, 0.000, 0.000)],
        ["CB", 0, (-0.523, -0.776, -1.210)],
        ["O", 3, (0.625, 1.062, -0.000)],
        ["CG", 4, (0.613, 1.391, -0.000)],
        ["SD", 5, (0.703, 1.695, 0.000)],
        ["CE", 6, (0.320, 1.786, -0.000)],
    ],
    "PHE": [
        ["N", 0, (-0.518, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.524, 0.000, -0.000)],
        ["CB", 0, (-0.525, -0.776, -1.212)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.607, 1.377, 0.000)],
        ["CD1", 5, (0.709, 1.195, -0.000)],
        ["CD2", 5, (0.706, -1.196, 0.000)],
        ["CE1", 5, (2.102, 1.198, -0.000)],
        ["CE2", 5, (2.098, -1.201, -0.000)],
        ["CZ", 5, (2.794, -0.003, -0.001)],
    ],
    "PRO": [
        ["N", 0, (-0.566, 1.351, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, -0.000, 0.000)],
        ["CB", 0, (-0.546, -0.611, -1.293)],
        ["O", 3, (0.621, 1.066, 0.000)],
        ["CG", 4, (0.382, 1.445, 0.0)],
        # ['CD', 5, (0.427, 1.440, 0.0)],
        ["CD", 5, (0.477, 1.424, 0.0)],  # manually made angle 2 degrees larger
    ],
    "SER": [
        ["N", 0, (-0.529, 1.360, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.518, -0.777, -1.211)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["OG", 4, (0.503, 1.325, 0.000)],
    ],
    "THR": [
        ["N", 0, (-0.517, 1.364, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, 0.000, -0.000)],
        ["CB", 0, (-0.516, -0.793, -1.215)],
        ["O", 3, (0.626, 1.062, 0.000)],
        ["CG2", 4, (0.550, -0.718, -1.228)],
        ["OG1", 4, (0.472, 1.353, 0.000)],
    ],
    "TRP": [
        ["N", 0, (-0.521, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, 0.000)],
        ["CB", 0, (-0.523, -0.776, -1.212)],
        ["O", 3, (0.627, 1.062, 0.000)],
        ["CG", 4, (0.609, 1.370, -0.000)],
        ["CD1", 5, (0.824, 1.091, 0.000)],
        ["CD2", 5, (0.854, -1.148, -0.005)],
        ["CE2", 5, (2.186, -0.678, -0.007)],
        ["CE3", 5, (0.622, -2.530, -0.007)],
        ["NE1", 5, (2.140, 0.690, -0.004)],
        ["CH2", 5, (3.028, -2.890, -0.013)],
        ["CZ2", 5, (3.283, -1.543, -0.011)],
        ["CZ3", 5, (1.715, -3.389, -0.011)],
    ],
    "TYR": [
        ["N", 0, (-0.522, 1.362, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.524, -0.000, -0.000)],
        ["CB", 0, (-0.522, -0.776, -1.213)],
        ["O", 3, (0.627, 1.062, -0.000)],
        ["CG", 4, (0.607, 1.382, -0.000)],
        ["CD1", 5, (0.716, 1.195, -0.000)],
        ["CD2", 5, (0.713, -1.194, -0.001)],
        ["CE1", 5, (2.107, 1.200, -0.002)],
        ["CE2", 5, (2.104, -1.201, -0.003)],
        ["OH", 5, (4.168, -0.002, -0.005)],
        ["CZ", 5, (2.791, -0.001, -0.003)],
    ],
    "VAL": [
        ["N", 0, (-0.494, 1.373, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, -0.000, -0.000)],
        ["CB", 0, (-0.533, -0.795, -1.213)],
        ["O", 3, (0.627, 1.062, -0.000)],
        ["CG1", 4, (0.540, 1.429, -0.000)],
        ["CG2", 4, (0.533, -0.776, 1.203)],
    ],
    "UNK": [],
}

# A list of atoms (excluding hydrogen) for each AA type. PDB naming convention.
RESIDUE_ATOMS = {
    "ALA": ["C", "CA", "CB", "N", "O"],
    "ARG": ["C", "CA", "CB", "CG", "CD", "CZ", "N", "NE", "O", "NH1", "NH2"],
    "ASP": ["C", "CA", "CB", "CG", "N", "O", "OD1", "OD2"],
    "ASN": ["C", "CA", "CB", "CG", "N", "ND2", "O", "OD1"],
    "CYS": ["C", "CA", "CB", "N", "O", "SG"],
    "GLU": ["C", "CA", "CB", "CG", "CD", "N", "O", "OE1", "OE2"],
    "GLN": ["C", "CA", "CB", "CG", "CD", "N", "NE2", "O", "OE1"],
    "GLY": ["C", "CA", "N", "O"],
    "HIS": ["C", "CA", "CB", "CG", "CD2", "CE1", "N", "ND1", "NE2", "O"],
    "ILE": ["C", "CA", "CB", "CG1", "CG2", "CD1", "N", "O"],
    "LEU": ["C", "CA", "CB", "CG", "CD1", "CD2", "N", "O"],
    "LYS": ["C", "CA", "CB", "CG", "CD", "CE", "N", "NZ", "O"],
    "MET": ["C", "CA", "CB", "CG", "CE", "N", "O", "SD"],
    "PHE": ["C", "CA", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "N", "O"],
    "PRO": ["C", "CA", "CB", "CG", "CD", "N", "O"],
    "SER": ["C", "CA", "CB", "N", "O", "OG"],
    "THR": ["C", "CA", "CB", "CG2", "N", "O", "OG1"],
    "TRP": [
        "C",
        "CA",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
        "N",
        "NE1",
        "O",
    ],
    "TYR": ["C", "CA", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "N", "O", "OH"],
    "VAL": ["C", "CA", "CB", "CG1", "CG2", "N", "O"],
}


RESIDUE_ATOM_RENAMING_SWAPS = {
    "PHE": [["CD1", "CD2"], ["CE1", "CE2"]],
    "TYR": [["CD1", "CD2"], ["CE1", "CE2"]],
    "ARG": [["NH1", "NH2"]],
    "ASP": [["OD1", "OD2"]],
    "GLU": [["OE1", "OE2"]],
}


RESIDUE_ATOM_PSEUDO_RENAMING_SWAPS = {
    "HIS": [["ND1", "CD2"], ["NE2", "CE1"]],
    "ASN": [["OD1", "ND2"]],
    "GLN": [["OE1", "NE2"]],
}


# Van der Waals radii [Angstroem] of the atoms (from Wikipedia)
VAN_DER_WAALS_RADIUS = {
    "C": 1.7,
    "N": 1.55,
    "O": 1.52,
    "S": 1.8,
}

# Sidechain bond lengths (from Rosetta database)
SC_BOND_LENGTHS = {
    "ARG": {
        ("CB", "CG"): 1.5204,
        ("CG", "CD"): 1.4854,
        ("CD", "NE"): 1.4541,
        ("NE", "CZ"): 1.3473,
    },
    "ASN": {("CB", "CG"): 1.5035, ("CG", "OD1"): 1.2364},
    "ASP": {("CB", "CG"): 1.5228, ("CG", "OD1"): 1.2082},
    "CYS": {("CB", "SG"): 1.8088},
    "GLU": {("CB", "CG"): 1.5221, ("CG", "CD"): 1.5034, ("CD", "OE1"): 1.2076},
    "GLN": {("CB", "CG"): 1.5191, ("CG", "CD"): 1.5169, ("CD", "OE1"): 1.2342},
    "HIS": {("CB", "CG"): 1.4972, ("CG", "ND1"): 1.3792},
    "ILE": {("CB", "CG1"): 1.5309, ("CG1", "CD1"): 1.5117},
    "LEU": {("CB", "CG"): 1.5340, ("CG", "CD1"): 1.5227},
    "LYS": {
        ("CB", "CG"): 1.5229,
        ("CG", "CD"): 1.5213,
        ("CD", "CE"): 1.5216,
        ("CE", "NZ"): 1.4881,
    },
    "MET": {("CB", "CG"): 1.5222, ("CG", "SD"): 1.8038, ("SD", "CE"): 1.7904},
    "PHE": {("CB", "CG"): 1.5022, ("CG", "CD1"): 1.3870},
    "PRO": {("CB", "CG"): 1.4906, ("CG", "CD"): 1.5055},
    "SER": {("CB", "OG"): 1.4012},
    "THR": {("CB", "OG1"): 1.4335},
    "TRP": {("CB", "CG"): 1.4987, ("CG", "CD1"): 1.3627},
    "TYR": {("CB", "CG"): 1.5127, ("CG", "CD1"): 1.3872},
    "VAL": {("CB", "CG1"): 1.5214},
}


# This mapping is used when we need to store atom data in a format that requires
# fixed atom data size for every residue (e.g. a numpy array).
ATOM_TYPES = [
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
]
ATOM_ORDER = {atom_type: i for i, atom_type in enumerate(ATOM_TYPES)}
ATOM_TYPE_NUM = len(ATOM_TYPES)  # := 37.

# A compact atom encoding with 14 columns
# pylint: disable=line-too-long
# pylint: disable=bad-whitespace
RESTYPE_NAME_TO_ATOM14_NAMES = {
    "ALA": ["N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", ""],
    "ARG": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "NE",
        "CZ",
        "NH1",
        "NH2",
        "",
        "",
        "",
    ],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", "", "", "", "", "", ""],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "", "", "", "", "", ""],
    "CYS": ["N", "CA", "C", "O", "CB", "SG", "", "", "", "", "", "", "", ""],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2", "", "", "", "", ""],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2", "", "", "", "", ""],
    "GLY": ["N", "CA", "C", "O", "", "", "", "", "", "", "", "", "", ""],
    "HIS": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "ND1",
        "CD2",
        "CE1",
        "NE2",
        "",
        "",
        "",
        "",
    ],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "", "", "", "", "", ""],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "", "", "", "", "", ""],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "", "", "", "", ""],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE", "", "", "", "", "", ""],
    "PHE": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE1",
        "CE2",
        "CZ",
        "",
        "",
        "",
    ],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD", "", "", "", "", "", "", ""],
    "SER": ["N", "CA", "C", "O", "CB", "OG", "", "", "", "", "", "", "", ""],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2", "", "", "", "", "", "", ""],
    "TRP": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "NE1",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
    ],
    "TYR": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE1",
        "CE2",
        "CZ",
        "OH",
        "",
        "",
    ],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "", "", "", "", "", "", ""],
    "UNK": ["", "", "", "", "", "", "", "", "", "", "", "", "", ""],
}

CG_ATOMS = {
    "ALA": [["C", "CA", "CB", "N"], ["C", "CA", "O"]],
    "ARG": [
        ["C", "CA", "CB", "N"],
        ["C", "CA", "O"],
        ["CB", "CG", "CD"],
        ["NE", "NH1", "NH2", "CZ"],
    ],
    "ASN": [["C", "CA", "CB", "N"], ["C", "CA", "O"], ["CG", "ND2", "OD1"]],
    "ASP": [["C", "CA", "CB", "N"], ["C", "CA", "O"], ["CG", "OD1", "OD2"]],
    "CYS": [["C", "CA", "CB", "N"], ["C", "CA", "O"], ["CA", "CB", "SG"]],
    "GLN": [["C", "CA", "CB", "N"], ["C", "CA", "O"], ["CG", "CD", "OE1", "NE2"]],
    "GLU": [["C", "CA", "CB", "N"], ["C", "CA", "O"], ["CG", "CD", "OE1", "OE2"]],
    "GLY": [["C", "CA", "N"], ["C", "CA", "O"]],
    "HIS": [
        ["C", "CA", "CB", "N"],
        ["C", "CA", "O"],
        ["CG", "CD2", "CE1", "ND1", "NE2"],
    ],
    "ILE": [
        ["C", "CA", "CB", "N"],
        ["C", "CA", "O"],
        ["CB", "CG1", "CG2"],
        ["CB", "CG1", "CD1"],
    ],
    "LEU": [["C", "CA", "CB", "N"], ["C", "CA", "O"], ["CG", "CD1", "CD2"]],
    "LYS": [
        ["C", "CA", "CB", "N"],
        ["C", "CA", "O"],
        ["CB", "CG", "CD"],
        ["CD", "CE", "NZ"],
    ],
    "MET": [["C", "CA", "CB", "N"], ["C", "CA", "O"], ["CG", "CE", "SD"]],
    "PHE": [
        ["C", "CA", "CB", "N"],
        ["C", "CA", "O"],
        ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    ],
    "PRO": [["C", "CA", "CB", "N"], ["C", "CA", "O"], ["CB", "CG", "CD"]],
    "SER": [["C", "CA", "CB", "N"], ["C", "CA", "O"], ["CA", "CB", "OG"]],
    "THR": [["C", "CA", "CB", "N"], ["C", "CA", "O"], ["CB", "CG2", "OG1"]],
    "TRP": [
        ["C", "CA", "CB", "N"],
        ["C", "CA", "O"],
        ["CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2", "NE1"],
    ],
    "TYR": [
        ["C", "CA", "CB", "N"],
        ["C", "CA", "O"],
        ["CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    ],
    "VAL": [["C", "CA", "CB", "N"], ["C", "CA", "O"], ["CB", "CG1", "CG2"]],
}

# A compact atom encoding with 16 columns
# pylint: disable=line-too-long
# pylint: disable=bad-whitespace
ATOM16 = [
    "N",
    "CA",
    "C",
    "O",
    "CB",
    "CG",
    "SG",
    "CD",
    "CD2",
    "ND1",
    "OD1",
    "OD2",
    "CE1",
    "NE2",
    "OE1",
    "OE2",
]
RESTYPE_NAME_TO_ATOM16_NAMES = {
    "ASP": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "",
        "",
        "",
        "",
        "OD1",
        "OD2",
        "",
        "",
        "",
        "",
    ],
    "CYS": ["N", "CA", "C", "O", "CB", "", "SG", "", "", "", "", "", "", "", "", ""],
    "GLU": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "",
        "CD",
        "",
        "",
        "",
        "",
        "",
        "",
        "OE1",
        "OE2",
    ],
    "HIS": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "",
        "",
        "CD2",
        "ND1",
        "",
        "",
        "CE1",
        "NE2",
        "",
        "",
    ],
    "UNK": ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
}

# A compact atom encoding with 7 columns
# pylint: disable=line-too-long
# pylint: disable=bad-whitespace
ATOM7 = [
    [0, 1, 2, 3, 1, 4, 5],
    [0, 1, 2, 3, 5, 6, 7],
    [0, 1, 2, 3, 6, 7, 8],
    [0, 1, 2, 3, 6, 8, 9],
    [0, 0, 0, 0, 0, 0, 0],
]

# A compact atom encoding with 8 columns
# pylint: disable=line-too-long
# pylint: disable=bad-whitespace
ATOM8 = [
    [0, 1, 2, 3, 4, 1, 4, 5],
    [0, 1, 2, 3, 4, 5, 6, 7],
    [0, 1, 2, 3, 4, 6, 7, 8],
    [0, 1, 2, 3, 4, 6, 8, 9],
    [0, 0, 0, 0, 0, 0, 0, 0],
]


def _make_rigid_transformation_4x4(
    ex: np.ndarray, ey: np.ndarray, translation: np.ndarray
) -> Any:
    """Create a rigid 4x4 transformation matrix from two axes and transl.

    Args:
        ex: Ex value.
        ey: Ey value.
        translation: Translation value.

    Returns:
        Result of the make rigid transformation 4x4 operation.
    """
    # Normalize ex.
    ex_normalized = ex / np.linalg.norm(ex)

    # make ey perpendicular to ex
    ey_normalized = ey - np.dot(ey, ex_normalized) * ex_normalized
    ey_normalized /= np.linalg.norm(ey_normalized)

    # compute ez as cross product
    eznorm = np.cross(ex_normalized, ey_normalized)
    m = np.stack([ex_normalized, ey_normalized, eznorm, translation]).transpose()
    m = np.concatenate([m, [[0.0, 0.0, 0.0, 1.0]]], axis=0)
    return m


HBOND_DONOR_ATOMS = [
    "OG",
    "OG1",
    "NE2",
    "ND1",
    "ND2",
    "NZ",
    "NE",
    "NH1",
    "NH2",
    "NE1",
    "OH",
    "N",
]
HBOND_ACCEPTOR_ATOMS = [
    "ND1",
    "NE2",
    "OE1",
    "OE2",
    "OD1",
    "OD2",
    "OH",
    "OG",
    "OG1",
    "O",
]

HBOND_DONORS = np.zeros(ATOM_TYPE_NUM)
HBOND_ACCEPTORS = np.zeros(ATOM_TYPE_NUM)
for atom in HBOND_DONOR_ATOMS:
    HBOND_DONORS[ATOM_ORDER[atom]] = 1.0
for atom in HBOND_ACCEPTOR_ATOMS:
    HBOND_ACCEPTORS[ATOM_ORDER[atom]] = 1.0


def _get_restype_atom14_hbond_donors_and_acceptors() -> Tuple[Any, ...]:
    """Return restype atom14 hbond donors and acceptors.

    Returns:
        Computed result values.
    """
    restype_hbond_donors = []
    restype_hbond_acceptors = []
    for res_name in RESTYPES:
        res_name = RESTYPE_1TO3[res_name]

        res_hbond_donors = [
            1.0 if atom in HBOND_DONOR_ATOMS else 0.0
            for atom in RESTYPE_NAME_TO_ATOM14_NAMES[res_name]
        ]
        res_hbond_acceptors = [
            1.0 if atom in HBOND_ACCEPTOR_ATOMS else 0.0
            for atom in RESTYPE_NAME_TO_ATOM14_NAMES[res_name]
        ]

        restype_hbond_donors.append(res_hbond_donors)
        restype_hbond_acceptors.append(res_hbond_acceptors)

    # Update for unknown restype
    restype_hbond_donors.append([0] * 14)
    restype_hbond_acceptors.append([0] * 14)

    return restype_hbond_donors, restype_hbond_acceptors


RESTYPE_HBOND_DONORS_ATOM14, RESTYPE_HBOND_ACCEPTORS_ATOM14 = (
    _get_restype_atom14_hbond_donors_and_acceptors()
)

# Format: The list for each AA type contains chi1, chi2, chi3, chi4 in
# this order (or a relevant subset from chi1 onwards). ALA and GLY don't have
# chi angles so their chi angle lists are empty.
CHI_ANGLES_ATOMS = {
    "ALA": [],
    # Chi5 in arginine is always 0 +- 5 degrees, so ignore it.
    "ARG": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "NE"],
        ["CG", "CD", "NE", "CZ"],
    ],
    "ASN": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "ASP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "CYS": [["N", "CA", "CB", "SG"]],
    "GLN": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "OE1"],
    ],
    "GLU": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "OE1"],
    ],
    "GLY": [],
    "HIS": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]],
    "ILE": [["N", "CA", "CB", "CG1"], ["CA", "CB", "CG1", "CD1"]],
    "LEU": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "LYS": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "CE"],
        ["CG", "CD", "CE", "NZ"],
    ],
    "MET": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "SD"],
        ["CB", "CG", "SD", "CE"],
    ],
    "PHE": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "PRO": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"]],
    "SER": [["N", "CA", "CB", "OG"]],
    "THR": [["N", "CA", "CB", "OG1"]],
    "TRP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "TYR": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "VAL": [["N", "CA", "CB", "CG1"]],
}

# If chi angles given in fixed-length array, this matrix determines how to mask
# them for each AA type. The order is as per restype_order (see below).
CHI_ANGLES_MASK = [
    [0.0, 0.0, 0.0, 0.0],  # ALA
    [1.0, 1.0, 1.0, 1.0],  # ARG
    [1.0, 1.0, 0.0, 0.0],  # ASN
    [1.0, 1.0, 0.0, 0.0],  # ASP
    [1.0, 0.0, 0.0, 0.0],  # CYS
    [1.0, 1.0, 1.0, 0.0],  # GLN
    [1.0, 1.0, 1.0, 0.0],  # GLU
    [0.0, 0.0, 0.0, 0.0],  # GLY
    [1.0, 1.0, 0.0, 0.0],  # HIS
    [1.0, 1.0, 0.0, 0.0],  # ILE
    [1.0, 1.0, 0.0, 0.0],  # LEU
    [1.0, 1.0, 1.0, 1.0],  # LYS
    [1.0, 1.0, 1.0, 0.0],  # MET
    [1.0, 1.0, 0.0, 0.0],  # PHE
    [1.0, 1.0, 0.0, 0.0],  # PRO
    [1.0, 0.0, 0.0, 0.0],  # SER
    [1.0, 0.0, 0.0, 0.0],  # THR
    [1.0, 1.0, 0.0, 0.0],  # TRP
    [1.0, 1.0, 0.0, 0.0],  # TYR
    [1.0, 0.0, 0.0, 0.0],  # VAL
]

# The following chi angles are pi periodic: they can be rotated by a multiple
# of pi without affecting the structure.
CHI_PI_PERIODIC = [
    [0.0, 0.0, 0.0, 0.0],  # ALA
    [0.0, 0.0, 0.0, 0.0],  # ARG
    [0.0, 0.0, 0.0, 0.0],  # ASN
    [0.0, 1.0, 0.0, 0.0],  # ASP
    [0.0, 0.0, 0.0, 0.0],  # CYS
    [0.0, 0.0, 0.0, 0.0],  # GLN
    [0.0, 0.0, 1.0, 0.0],  # GLU
    [0.0, 0.0, 0.0, 0.0],  # GLY
    [0.0, 0.0, 0.0, 0.0],  # HIS
    [0.0, 0.0, 0.0, 0.0],  # ILE
    [0.0, 0.0, 0.0, 0.0],  # LEU
    [0.0, 0.0, 0.0, 0.0],  # LYS
    [0.0, 0.0, 0.0, 0.0],  # MET
    [0.0, 1.0, 0.0, 0.0],  # PHE
    [0.0, 0.0, 0.0, 0.0],  # PRO
    [0.0, 0.0, 0.0, 0.0],  # SER
    [0.0, 0.0, 0.0, 0.0],  # THR
    [0.0, 0.0, 0.0, 0.0],  # TRP
    [0.0, 1.0, 0.0, 0.0],  # TYR
    [0.0, 0.0, 0.0, 0.0],  # VAL
    [0.0, 0.0, 0.0, 0.0],  # UNK
]

# The following chi angles are pseudo pi periodic: due to experimental limitations,
# atoms are sometimes ambiguous for HIS, ASN, GLN
CHI_PSEUDO_PI_PERIODIC = [
    [0.0, 0.0, 0.0, 0.0],  # ALA
    [0.0, 0.0, 0.0, 0.0],  # ARG
    [0.0, 1.0, 0.0, 0.0],  # ASN
    [0.0, 0.0, 0.0, 0.0],  # ASP
    [0.0, 0.0, 0.0, 0.0],  # CYS
    [0.0, 0.0, 1.0, 0.0],  # GLN
    [0.0, 0.0, 0.0, 0.0],  # GLU
    [0.0, 0.0, 0.0, 0.0],  # GLY
    [0.0, 1.0, 0.0, 0.0],  # HIS
    [0.0, 0.0, 0.0, 0.0],  # ILE
    [0.0, 0.0, 0.0, 0.0],  # LEU
    [0.0, 0.0, 0.0, 0.0],  # LYS
    [0.0, 0.0, 0.0, 0.0],  # MET
    [0.0, 0.0, 0.0, 0.0],  # PHE
    [0.0, 0.0, 0.0, 0.0],  # PRO
    [0.0, 0.0, 0.0, 0.0],  # SER
    [0.0, 0.0, 0.0, 0.0],  # THR
    [0.0, 0.0, 0.0, 0.0],  # TRP
    [0.0, 0.0, 0.0, 0.0],  # TYR
    [0.0, 0.0, 0.0, 0.0],  # VAL
    [0.0, 0.0, 0.0, 0.0],  # UNK
]

# create an array with (restype, atomtype) --> rigid_group_idx
# and an array with (restype, atomtype, coord) for the atom positions
# and compute affine transformation matrices (4,4) from one rigid group to the
# previous group
RESTYPE_ATOM37_TO_RIGID_GROUP = np.zeros([21, 37], dtype=np.int64)
RESTYPE_ATOM37_MASK = np.zeros([21, 37], dtype=np.float32)
RESTYPE_ATOM37_RIGID_GROUP_POSITIONS = np.zeros([21, 37, 3], dtype=np.float32)
RESTYPE_ATOM14_TO_RIGID_GROUP = np.zeros([21, 14], dtype=np.int64)
RESTYPE_ATOM14_MASK = np.zeros([21, 14], dtype=np.float32)
RESTYPE_ATOM14_RIGID_GROUP_POSITIONS = np.zeros([21, 14, 3], dtype=np.float32)
RESTYPE_RIGID_GROUP_DEFAULT_FRAME = np.zeros([21, 8, 4, 4], dtype=np.float32)


def _make_rigid_group_constants() -> None:
    """Fill the arrays above."""
    for restype, restype_letter in enumerate(RESTYPES):
        resname = RESTYPE_1TO3[restype_letter]
        for atomname, group_idx, atom_position in RIGID_GROUP_ATOM_POSITIONS[resname]:
            atomtype = ATOM_ORDER[atomname]
            RESTYPE_ATOM37_TO_RIGID_GROUP[restype, atomtype] = group_idx
            RESTYPE_ATOM37_MASK[restype, atomtype] = 1
            RESTYPE_ATOM37_RIGID_GROUP_POSITIONS[restype, atomtype, :] = atom_position

            atom14idx = RESTYPE_NAME_TO_ATOM14_NAMES[resname].index(atomname)
            RESTYPE_ATOM14_TO_RIGID_GROUP[restype, atom14idx] = group_idx
            RESTYPE_ATOM14_MASK[restype, atom14idx] = 1
            RESTYPE_ATOM14_RIGID_GROUP_POSITIONS[restype, atom14idx, :] = atom_position

    for restype, restype_letter in enumerate(RESTYPES):
        resname = RESTYPE_1TO3[restype_letter]
        atom_positions = {
            name: np.array(pos) for name, _, pos in RIGID_GROUP_ATOM_POSITIONS[resname]
        }

        # backbone to backbone is the identity transform
        RESTYPE_RIGID_GROUP_DEFAULT_FRAME[restype, 0, :, :] = np.eye(4)

        # pre-omega-frame to backbone (currently dummy identity matrix)
        RESTYPE_RIGID_GROUP_DEFAULT_FRAME[restype, 1, :, :] = np.eye(4)

        # phi-frame to backbone
        mat = _make_rigid_transformation_4x4(
            ex=atom_positions["N"] - atom_positions["CA"],
            ey=np.array([1.0, 0.0, 0.0]),
            translation=atom_positions["N"],
        )
        RESTYPE_RIGID_GROUP_DEFAULT_FRAME[restype, 2, :, :] = mat

        # psi-frame to backbone
        mat = _make_rigid_transformation_4x4(
            ex=atom_positions["C"] - atom_positions["CA"],
            ey=atom_positions["CA"] - atom_positions["N"],
            translation=atom_positions["C"],
        )
        RESTYPE_RIGID_GROUP_DEFAULT_FRAME[restype, 3, :, :] = mat

        # chi1-frame to backbone
        if CHI_ANGLES_MASK[restype][0]:
            base_atom_names = CHI_ANGLES_ATOMS[resname][0]
            base_atom_positions = [atom_positions[name] for name in base_atom_names]
            mat = _make_rigid_transformation_4x4(
                ex=base_atom_positions[2] - base_atom_positions[1],
                ey=base_atom_positions[0] - base_atom_positions[1],
                translation=base_atom_positions[2],
            )
            RESTYPE_RIGID_GROUP_DEFAULT_FRAME[restype, 4, :, :] = mat

        # chi2-frame to chi1-frame
        # chi3-frame to chi2-frame
        # chi4-frame to chi3-frame
        # luckily all rotation axes for the next frame start at (0,0,0) of the
        # previous frame
        for chi_idx in range(1, 4):
            if CHI_ANGLES_MASK[restype][chi_idx]:
                axis_end_atom_name = CHI_ANGLES_ATOMS[resname][chi_idx][2]
                axis_end_atom_position = atom_positions[axis_end_atom_name]
                mat = _make_rigid_transformation_4x4(
                    ex=axis_end_atom_position,
                    ey=np.array([-1.0, 0.0, 0.0]),
                    translation=axis_end_atom_position,
                )
                RESTYPE_RIGID_GROUP_DEFAULT_FRAME[restype, 4 + chi_idx, :, :] = mat


_make_rigid_group_constants()


STEREO_CHEMICAL_PROPS_PATH = resources.files(__package__).joinpath(
    "stereo_chemical_props.txt"
)


def restype_bonded_atoms(self_bonds: bool = False, atom14: bool = True) -> Any:
    """Execute the restype bonded atoms operation.

    Args:
        self_bonds: Self bonds value.
        atom14: Atom14 value.

    Returns:
        Result of the restype bonded atoms operation.
    """
    stereo_chemical_props = STEREO_CHEMICAL_PROPS_PATH.read_text()
    lines_iter = iter(stereo_chemical_props.splitlines())

    # Determine bonded residues
    if atom14:
        restype_bonded_atoms = np.zeros([21, 14, 14], dtype=np.float32)
    else:
        restype_bonded_atoms = np.zeros([21, 37, 37], dtype=np.float32)

    next(lines_iter)  # Skip header line.
    for line in lines_iter:
        if line.strip() == "-":
            break
        bond, resname, _, _ = line.split()
        atom1, atom2 = bond.split("-")

        # Get residue and atom indices
        res_idx = RESTYPE_ORDER[RESTYPE_3TO1[resname]]
        if atom14:
            atom1_idx = RESTYPE_NAME_TO_ATOM14_NAMES[resname].index(atom1)
            atom2_idx = RESTYPE_NAME_TO_ATOM14_NAMES[resname].index(atom2)
        else:
            atom1_idx = ATOM_ORDER[atom1]
            atom2_idx = ATOM_ORDER[atom2]

        # Symmetrically mark each bonded atom
        restype_bonded_atoms[res_idx, atom1_idx, atom2_idx] = 1.0
        restype_bonded_atoms[res_idx, atom2_idx, atom1_idx] = 1.0

    if self_bonds:
        for restype in RESTYPES:
            res_idx = RESTYPE_ORDER[restype]
            for atom in ATOM_TYPES:
                if atom14:
                    if atom not in RESIDUE_ATOMS[RESTYPE_1TO3[restype]]:
                        continue
                    atom_idx = RESTYPE_NAME_TO_ATOM14_NAMES[restype].index(atom)
                else:
                    atom_idx = ATOM_ORDER[atom]
                restype_bonded_atoms[res_idx, atom_idx, atom_idx] = 1.0

    return restype_bonded_atoms


def _get_chi_atom_indices_and_mask(use_atom14: bool = True) -> Tuple[Any, ...]:
    """Return chi atom indices and mask.

    Args:
        use_atom14: Use atom14 value.

    Returns:
        Computed result values.
    """
    chi_atom_indices = []
    chi_mask = []
    for res_name in RESTYPES:
        res_name = RESTYPE_1TO3[res_name]
        res_chi_angles = CHI_ANGLES_ATOMS[res_name]

        # Chi mask where 1 for existing chi angle and 0 for nonexistent chi angle
        chi_mask.append([1] * len(res_chi_angles) + [0] * (4 - len(res_chi_angles)))

        # All unique atoms for chi angles
        atoms = [atom for chi in res_chi_angles for atom in chi]
        atoms = sorted(set(atoms), key=lambda x: atoms.index(x))

        # Indices of unique atoms
        if use_atom14:
            atom_indices = [
                RESTYPE_NAME_TO_ATOM14_NAMES[res_name].index(atom) for atom in atoms
            ]
        else:
            atom_indices = [ATOM_ORDER[atom] for atom in atoms]

        for _ in range(7 - len(atom_indices)):
            atom_indices.append(0)

        chi_atom_indices.append(atom_indices)

    # Update for unknown restype
    chi_atom_indices.append([0] * 7)
    chi_mask.append([0] * 4)

    return chi_atom_indices, chi_mask


CHI_ATOM_INDICES_ATOM14, CHI_MASK_ATOM14 = _get_chi_atom_indices_and_mask(
    use_atom14=True
)
CHI_ATOM_INDICES_ATOM37, CHI_MASK_ATOM37 = _get_chi_atom_indices_and_mask(
    use_atom14=False
)


def _get_restype_atom_radius_atom14() -> Any:
    """Return restype atom radius atom14.

    Returns:
        Result of the get restype atom radius atom14 operation.
    """
    restype_atom_radius = []
    for res_name in RESTYPES:
        res_name = RESTYPE_1TO3[res_name]
        atom_radius = [
            VAN_DER_WAALS_RADIUS[name[0]]
            for name in RESTYPE_NAME_TO_ATOM14_NAMES[res_name]
            if name != ""
        ]

        for _ in range(14 - len(atom_radius)):
            atom_radius.append(0)

        restype_atom_radius.append(atom_radius)

    # Update for unknown restype
    restype_atom_radius.append([0] * 14)

    return restype_atom_radius


RESTYPE_ATOM_RADIUS_ATOM14 = _get_restype_atom_radius_atom14()
