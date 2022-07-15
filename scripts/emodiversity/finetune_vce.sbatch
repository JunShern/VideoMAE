#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=finetune_vce
#SBATCH --output=/accounts/projects/jsteinhardt/hendrycks/emotions/slurm_%A.out

set -x

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

DATA_PATH='/accounts/projects/jsteinhardt/hendrycks/emotions/vce_for_videomae_dataset'
MODEL_PATH='/data/hendrycks/emotions/kinetics400-videomae-no-ViTB-1600-16x5x3-pretrain.pth'
OUTPUT_DIR='/data/hendrycks/emotions/tensorboard/tmp/'

JOB_NAME="finetune-vce"
PARTITION=#${PARTITION:-"video"}
# 8 for 1 node, 16 for 2 node, etc.
GPUS=${GPUS:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:2}

# batch_size can be adjusted according to the graphics card
# srun -p $PARTITION \
srun -p jsteinhardt -w balrog \
        --job-name=${JOB_NAME} \
        --gres=gpu:${GPUS_PER_NODE} \
        --ntasks=${GPUS} \
        --ntasks-per-node=${GPUS_PER_NODE} \
        --cpus-per-task=${CPUS_PER_TASK} \
        --kill-on-bad-exit=1 \
        ${SRUN_ARGS} \
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
        # --enable_deepspeed \
        ${PY_ARGS}