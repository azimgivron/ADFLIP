#!/usr/bin/env bash

export PYTHONPATH=$PWD/src:${PYTHONPATH:-}

DATA_DIR="${DATA_DIR:-dataset}"

python3 -m adflip.data.parser_dataset --data_path ${DATA_DIR}/train/
python3 -m adflip.data.parser_dataset --data_path ${DATA_DIR}/valid/
python3 -m adflip.data.parser_dataset --data_path ${DATA_DIR}/test_metal/
python3 -m adflip.data.parser_dataset --data_path ${DATA_DIR}/test_small_molecule/
python3 -m adflip.data.parser_dataset --data_path ${DATA_DIR}/test_nucleotide/
