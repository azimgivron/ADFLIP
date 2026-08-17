#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

DATA_DIR="${DATA_DIR:-dataset}"

python3 -c 'from adflip.data.parser_dataset import main; raise SystemExit(main())' \
  --data_path "${DATA_DIR}/train/"
python3 -c 'from adflip.data.parser_dataset import main; raise SystemExit(main())' \
  --data_path "${DATA_DIR}/valid/"
python3 -c 'from adflip.data.parser_dataset import main; raise SystemExit(main())' \
  --data_path "${DATA_DIR}/test_metal/"
python3 -c 'from adflip.data.parser_dataset import main; raise SystemExit(main())' \
  --data_path "${DATA_DIR}/test_small_molecule/"
python3 -c 'from adflip.data.parser_dataset import main; raise SystemExit(main())' \
  --data_path "${DATA_DIR}/test_nucleotide/"
