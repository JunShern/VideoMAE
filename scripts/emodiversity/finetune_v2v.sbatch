#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=finetune_vce
#SBATCH --output=slurm_%A.out

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

JOB_NAME="finetune-v2v"
GPUS=${GPUS:-1}

RUN_PATH='PATH/TO/VideoMAE/scripts/emodiversity'
cd $RUN_PATH

# batch_size can be adjusted according to the graphics card
# srun -p $PARTITION \
srun --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS} \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=8 \
    --kill-on-bad-exit=1 \
    finetune_v2v.sh