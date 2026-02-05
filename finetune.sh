#!/bin/bash

CKPT=./lmsys
PLAYDATA=./playground/data

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $CKPT/vicuna-7b-v1.5 \
    --version v1 \
    --data_path $PLAYDATA/llava_v1_5_mix665k.json \
    --image_folder $PLAYDATA \
    --vision_tower $CKPT/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/${MODEL_NAME}-pretrain/mm_projector.bin \
    --mm_projector_type cross_mlp \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --feature_order_reverse False \
    --image_aspect_ratio hyres \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/${MODEL_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --skip_strategy "dense2bottom" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none



