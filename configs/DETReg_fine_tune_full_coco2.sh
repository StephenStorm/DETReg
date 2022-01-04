#!/usr/bin/env bash

set -x

EXP_DIR=exps/DETReg_fine_tune_full_coco2
PY_ARGS=${@:1}

python -u main.py --output_dir ${EXP_DIR} --dataset coco --pretrain exps/DETReg_fine_tune_full_coco/checkpoint0049.pth ${PY_ARGS} -lr 2e-5