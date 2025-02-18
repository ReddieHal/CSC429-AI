#!/bin/bash

# Get the absolute path to the PrimeVul directory (parent of os_expr)
PRIMEVUL_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )"
DATA_DIR="${PRIMEVUL_DIR}/primeVULjson"

# Define data paths
TRAIN_FILE="${DATA_DIR}/primevul_train.jsonl"
VALID_FILE="${DATA_DIR}/primevul_valid.jsonl"
TEST_FILE="${DATA_DIR}/primevul_test.jsonl"

# Check if data files exist
if [ ! -f "$TRAIN_FILE" ]; then
    echo "Error: Training file not found at: $TRAIN_FILE"
    echo "Please ensure your data files are in the correct location"
    exit 1
fi

if [ ! -f "$VALID_FILE" ]; then
    echo "Error: Validation file not found at: $VALID_FILE"
    echo "Please ensure your data files are in the correct location"
    exit 1
fi

if [ ! -f "$TEST_FILE" ]; then
    echo "Error: Test file not found at: $TEST_FILE"
    echo "Please ensure your data files are in the correct location"
    exit 1
fi

PROJECT="primevul_cls"
TYPE="roberta"
MODEL="microsoft/codebert-base"
TOKENIZER="microsoft/codebert-base"
OUTPUT_DIR="${PRIMEVUL_DIR}/output/"

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "No GPU detected, running on CPU"
    NO_CUDA="--no_cuda"
else
    NO_CUDA=""
fi

echo "Using data files:"
echo "Train: $TRAIN_FILE"
echo "Valid: $VALID_FILE"
echo "Test: $TEST_FILE"

python run_ft.py \
    --project ${PROJECT} \
    --model_dir ${MODEL} \
    --output_dir=${OUTPUT_DIR} \
    --model_type=${TYPE} \
    --tokenizer_name=${TOKENIZER} \
    --model_name_or_path=${MODEL} \
    --do_train \
    --do_test \
    ${NO_CUDA} \
    --train_data_file=${TRAIN_FILE} \
    --eval_data_file=${VALID_FILE} \
    --test_data_file=${TEST_FILE} \
    --max_source_length 400 \
    --block_size 512 \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --warmup_steps 1000 \
    --max_grad_norm 1.0 \
    --num_train_epochs 10 \
    --evaluate_during_training \
    --fp16 \
    --seed 123456 \
    --overwrite_output_dir \
    --logging_steps 100

cd ..
