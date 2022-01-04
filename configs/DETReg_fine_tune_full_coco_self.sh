#!/usr/bin/env bash

set -x

EXP_DIR=exps/DETReg_fine_tune_full_coco
PY_ARGS=${@:1}

python -u main.py --output_dir ${EXP_DIR} --dataset coco  ${PY_ARGS}