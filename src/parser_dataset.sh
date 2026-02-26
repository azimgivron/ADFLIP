#!/usr/bin/env bash

export PYTHONPATH=$PWD:${PYTHONPATH:-}

DATA_DIR="${DATA_DIR:-dataset}"

python3 data/parser_dataset.py --data_path ${DATA_DIR}/train/
python3 data/parser_dataset.py --data_path ${DATA_DIR}/valid/
python3 data/parser_dataset.py --data_path ${DATA_DIR}/test_metal/
python3 data/parser_dataset.py --data_path ${DATA_DIR}/test_small_molecule/
python3 data/parser_dataset.py --data_path ${DATA_DIR}/test_nucleotide/
