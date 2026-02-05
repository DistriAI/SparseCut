#!/bin/bash


#MODEL_PATH=./checkpoints/llava-v1.5-7b-residual-${res_step}
#MODEL_NAME=llava-v1.5-7b-residual-${res_step}

python -m llava.eval.model_vqa \
    --model-path $MODEL_PATH \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/${MODEL_NAME}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/${MODEL_NAME}.jsonl \
    --dst ./playground/data/eval/mm-vet/results/${MODEL_NAME}.json

