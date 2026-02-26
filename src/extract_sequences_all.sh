#!/usr/bin/env bash

export PYTHONPATH=$PWD:${PYTHONPATH:-}

DATA_DIR="${DATA_DIR:-dataset}"
OUT_DIR="data/cluster/mmseqs"

for split in train valid test_metal test_small_molecule test_nucleotide; do
  parsed_dir="${DATA_DIR}/${split}_parsed"
  if [[ -d "${parsed_dir}" ]]; then
    echo "[extract] ${parsed_dir}"
    python3 data/export_parsed_sequences.py \
      --parsed_dir "${parsed_dir}" \
      --fasta_out "${OUT_DIR}/${split}_chains.fasta" \
      --metadata_out "${OUT_DIR}/${split}_chains_meta.csv" \
      --min_len 20
  else
    echo "[skip] ${parsed_dir} (not found)"
  fi
done
