#!/bin/bash

#MODEL_PATH=/data/zhangjingrui/LLaVA/checkpoints/llava-v1.5-7b-residual-4
#MODEL_NAME=llava-v1.5-7b-residual-4

#MODEL_PATH=./checkpoints/llava-v1.5-7b-residual-${res_step}
#MODEL_NAME=llava-v1.5-7b-residual-${res_step}

python -m llava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/${MODEL_NAME}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment $MODEL_NAME

cd eval_tool

python calculation.py --results_dir answers/$MODEL_NAME
