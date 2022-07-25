#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=finetune_vce
#SBATCH --output=slurm_%A.out

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

JOB_NAME="finetune-vce"
GPUS=${GPUS:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}

RUN_PATH='PATH/TO/VideoMAE/scripts/emodiversity'
cd $RUN_PATH

# batch_size can be adjusted according to the graphics card
srun --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    finetune_vce.sh