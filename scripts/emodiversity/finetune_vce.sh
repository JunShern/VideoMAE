#!/bin/bash
set -x

# MODIFY WITH YOUR OWN PATH
REPO_PATH='/path/to/emodiversity/VideoMAE'

DATA_PATH="/path/to/vce_dataset/"
MODEL_PATH="${REPO_PATH}/models/kinetics400-ViTB-1600-16x5x3-pretrain.pth"
OUTPUT_DIR="${REPO_PATH}/output/my_new_run/"

cd ${REPO_PATH}
python -u run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_set VCE \
    --nb_classes 27 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 8 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --opt adamw \
    --lr 5e-4 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 50 \
    --dist_eval \
    --test_num_segment 2 \
    --test_num_crop 3 \
    # --enable_deepspeed