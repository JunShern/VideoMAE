#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
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

JOB_NAME="finetune-v2v"
# PARTITION=#${PARTITION:-"video"}
# # 8 for 1 node, 16 for 2 node, etc.
# GPUS=${GPUS:-1}
# GPUS_PER_NODE=${GPUS_PER_NODE:-4}
# CPUS_PER_TASK=${CPUS_PER_TASK:-8}
# SRUN_ARGS=${SRUN_ARGS:-""}
# PY_ARGS=${@:2}

# batch_size can be adjusted according to the graphics card
# srun -p $PARTITION \
srun -p jsteinhardt -w balrog \
        --job-name=${JOB_NAME} \
        --gres=gpu:4 \
        --ntasks=1 \
        --ntasks-per-node=1 \
        --cpus-per-task=8 \
        --kill-on-bad-exit=1 \
        ${SRUN_ARGS} \
        python run_v2v_finetuning.py \
        --data_dir /data/hendrycks/emotions/v2v_dataset \
        --results_path /data/hendrycks/emotions/v2v_finetuning \
        --checkpoint /data/hendrycks/emotions/kinetics400-videomae-no-ViTB-1600-16x5x3-pretrain.pth \
        --wandb 1 \
        --disable_tqdm 1 \
        --num_gpus 4 \
        --batch_size 32 \
        --num_workers 2