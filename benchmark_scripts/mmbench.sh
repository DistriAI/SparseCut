#!/bin/bash

SPLIT="mmbench_dev_20230712"
#MODEL_PATH=/data/zhangjingrui/LLaVA/checkpoints/llava-v1.5-7b-residual-8
#MODEL_NAME=llava-v1.5-7b-residual-8

#MODEL_PATH=./checkpoints/llava-v1.5-7b-residual-${res_step}
#MODEL_NAME=llava-v1.5-7b-residual-${res_step}

python -m llava.eval.model_vqa_mmbench \
    --model-path $MODEL_PATH \
    --question-file ./playground/data/eval/mmbench/${SPLIT}.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/${SPLIT}/${MODEL_NAME}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment $MODEL_NAME
