#!/bin/bash

#MODEL_PATH=./checkpoints/llava-v1.5-7b-residual-${res_step}
#MODEL_NAME=llava-v1.5-7b-residual-${res_step}

python -m llava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder ./playground/data/eval/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/${MODEL_NAME}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/${MODEL_NAME}.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/${MODEL_NAME}.json
