set -x

# MODIFY WITH YOUR OWN PATH
REPO_PATH='PATH/TO/VideoMAE'
MODEL_PATH='PATH/TO/TRAINED_MODEL.pt'

DATA_PATH="${REPO_PATH}/dataset/vce_for_videomae_dataset"
OUTPUT_DIR="${REPO_PATH}/output/"

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
    --enable_deepspeed \
    --eval