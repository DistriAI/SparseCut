# login to huggingface

CKPT=${1:-"None"}
conv_template=${2:-"vicuna_v1"}
vistoken_patch_size=${3:-"None"}

eval_tasks=${eval_tasks:-"textvqa,chartqa,docvqa"}
master_port=${master_port:-"42759"}
#GPUS=`nvidia-smi -L | wc -l`
#GPUS=2
#!/bin/bash

# 自动获取可见GPU数量
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "CUDA_VISIBLE_DEVICES is not set. Using all available GPUs."
    GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi

echo "Using $GPUS GPU(s)"


echo $CKPT, $conv_template

accelerate launch --num_processes=$GPUS --main_process_port=${master_port} -m lmms_eval --model llava   \
    --model_args pretrained=$CKPT,conv_template=${conv_template} \
    --tasks $eval_tasks  --batch_size 1 --log_samples --log_samples_suffix lmms_eval --output_path $CKPT/logs/
