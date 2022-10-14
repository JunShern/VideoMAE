#!/bin/bash
set -x

# MODIFY WITH YOUR OWN PATH
REPO_PATH='PATH/TO/VideoMAE'
DATA_PATH="PATH/TO/v2v_dataset"

MODEL_PATH="${REPO_PATH}/models/kinetics400-ViTB-1600-16x5x3-pretrain.pth"
OUTPUT_DIR="${REPO_PATH}/output/"

cd ${REPO_PATH}
python run_v2v_finetuning.py \
    --data_dir ${DATA_PATH} \
    --results_path ${OUTPUT_DIR} \
    --checkpoint ${MODEL_PATH} \
    --wandb 1 \
    --disable_tqdm 1 \
    --num_gpus 1 \
    --batch_size 4 \
    --num_workers 2