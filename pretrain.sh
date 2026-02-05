#!/bin/bash

CKPT=./lmsys
PLAYDATA=./playground/data
#MODEL_NAME=llava4test
# 同时启动两个deepspeed 命令会端口冲突
#--image_aspect_ratio hyres \
#--include localhost:4 \
#--master_port 42957 \

deepspeed --master_port 42957 \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $CKPT/vicuna-7b-v1.5 \
    --version plain \
    --data_path $PLAYDATA/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder $PLAYDATA/LLaVA-Pretrain/images \
    --vision_tower $CKPT/clip-vit-large-patch14-336 \
    --mm_projector_type cross_mlp \
    --image_aspect_ratio hyres \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --feature_order_reverse False \
    --bf16 True \
    --output_dir ./checkpoints/${MODEL_NAME}-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --skip_strategy "dense2bottom" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
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
