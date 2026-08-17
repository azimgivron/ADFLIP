# ADFLIP

This repository is the minimal ADFLIP support package used by
[`hessian-flow`](../hessian-flow/hessian-flow). It provides the all-atom data
types and dataset, the Zoidberg network components, the masked-flow primitives,
side-chain packing through the retained PIPPack runtime, and the structure-data
preprocessing pipeline required by that project.

The original standalone ADFLIP trainer, inference and benchmark workflows are
outside this package's scope.

## Install

Hessian Flow already declares ADFLIP as an editable local dependency. To install
ADFLIP directly instead, run:

```bash
python -m pip install -e .
```

Zoidberg uses PyTorch Geometric's `knn_graph`; install the `torch-cluster` wheel
that matches the PyTorch and accelerator versions in the consuming environment.

## Preprocess structure data

The complete preprocessing workflow downloads the configured RCSB mmCIF files,
parses them to `.npz`, exports protein-chain FASTA files, clusters train and
validation chains with MMseqs2, and builds the cluster lookup files:

```bash
./src/data_preprocess_pipeline.sh
```

The pipeline requires `mmseqs` on `PATH`. It reads split definitions from
`data/split`, writes raw and parsed structures below `dataset` by default, and
writes cluster files below `data/cluster`. Override the structure-data location
with `DATA_DIR=/path/to/data`; override MMseqs threads with `THREADS=<count>`.

The individual installed commands are:

```text
adflip-download-data
adflip-parse-dataset
adflip-export-sequences
adflip-build-mmseqs-clusters
```

## Side-chain packing

`adflip.model.sidechain_packing` exposes `load_sidechain_models` and
`pack_sidechains` as focused functions. Models and inference configuration are
passed explicitly to the packing function so Hessian Flow can own future
concurrent scheduling.

## Citation

```bibtex
@inproceedings{yi2025allatom,
  title={All-atom inverse protein folding through discrete flow matching},
  author={Kai Yi and Kiarash Jamali and Sjors HW Scheres},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025},
  url={https://openreview.net/forum?id=8tQdwSCJmA}
}
```
