#!/usr/bin/env bash

set -x

EXP_DIR=exps/DETReg_fine_tune_trademark_50cls
# EXP_DIR=exps/test
PY_ARGS=${@:1}

python -u main.py --output_dir ${EXP_DIR} --dataset trademark  --pretrain pretrain/imagenet5ep_official.pth --epochs 250 ${PY_ARGS}
# python -u main.py --output_dir ${EXP_DIR} --filter_pct 0.01 --dataset coco --pretrain exps/DETReg_top30_coco/checkpoint.pth --epochs 2000 ${PY_ARGS}