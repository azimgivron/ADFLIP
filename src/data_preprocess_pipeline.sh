#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

DATA_DIR="${DATA_DIR:-dataset}"

# 1) Download mmCIF files from RCSB (uncomment as needed)
#    Train/valid: download asymmetric unit
python3 -c 'from adflip.data.download_data import main; raise SystemExit(main())' \
  --json data/split/train.json --out_dir "${DATA_DIR}/train"
python3 -c 'from adflip.data.download_data import main; raise SystemExit(main())' \
  --json data/split/valid.json --out_dir "${DATA_DIR}/valid"
#    Test: download biological assembly 1 only
python3 -c 'from adflip.data.download_data import main; raise SystemExit(main())' \
  --json data/split/test_metal.json --out_dir "${DATA_DIR}/test_metal" \
  --base_url "https://files.rcsb.org/download/{pdb_id}-assembly1.cif.gz"
python3 -c 'from adflip.data.download_data import main; raise SystemExit(main())' \
  --json data/split/test_small_molecule.json --out_dir "${DATA_DIR}/test_small_molecule" \
  --base_url "https://files.rcsb.org/download/{pdb_id}-assembly1.cif.gz"
python3 -c 'from adflip.data.download_data import main; raise SystemExit(main())' \
  --json data/split/test_nucleotide.json --out_dir "${DATA_DIR}/test_nucleotide" \
  --base_url "https://files.rcsb.org/download/{pdb_id}-assembly1.cif.gz"

# 2) Parse mmCIF files into .npz (output: ${DATA_DIR}/*_parsed/)
bash src/parser_dataset.sh

# 3) Extract sequences from parsed data (output: FASTA + metadata CSV)
bash src/extract_sequences_all.sh

# 4) MMseqs2 clustering (requires mmseqs in PATH)
bash src/cluster_train_valid_mmseqs.sh

# 5) Build cluster split files (output: data/cluster/)
python3 -c 'from adflip.data.build_mmseqs_clusters import main; raise SystemExit(main())' \
  --train_tsv data/cluster/mmseqs/mmseqs_train/train_clusters.tsv \
  --valid_tsv data/cluster/mmseqs/mmseqs_valid/valid_clusters.tsv \
  --out_dir data/cluster
