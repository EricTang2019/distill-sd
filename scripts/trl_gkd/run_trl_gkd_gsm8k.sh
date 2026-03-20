#!/usr/bin/env bash
set -euo pipefail

ROOT=/work5/jingwut/On-Policy-Distillation/verl
TS=${TS:-$(date +%Y%m%d_%H%M%S)}

export PYTHONPATH=/work5/jingwut/trl:${PYTHONPATH:-}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
IFS=',' read -r -a _GPUS <<< "${CUDA_VISIBLE_DEVICES}"
NPROC_PER_NODE=${NPROC_PER_NODE:-${#_GPUS[@]}}

STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen2.5-0.5B}
TEACHER_MODEL=${TEACHER_MODEL:-Qwen/Qwen3-1.7B-Base}
TRAIN_FILE=${TRAIN_FILE:-/work5/jingwut/On-Policy-Distillation/data/gsm8k_gkd/train.parquet}
EVAL_FILE=${EVAL_FILE:-/work5/jingwut/On-Policy-Distillation/data/gsm8k_gkd/test.parquet}

MAX_STEPS=${MAX_STEPS:-2000}
SAVE_STEPS=${SAVE_STEPS:-500}
EVAL_STEPS=${EVAL_STEPS:-200}
LOGGING_STEPS=${LOGGING_STEPS:-10}
LEARNING_RATE=${LEARNING_RATE:-5e-6}
TEMPERATURE=${TEMPERATURE:-1.0}
GAMMA=${GAMMA:-6}

PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-2}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE:-2}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-16}

MAX_LENGTH=${MAX_LENGTH:-768}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
LMBDA=${LMBDA:-1.0}
BETA=${BETA:-0.0}
DATASET_NUM_PROC=${DATASET_NUM_PROC:-4}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-2}

REPORT_TO=${REPORT_TO:-wandb}
WANDB_PROJECT=${WANDB_PROJECT:-trl_gkd_gsm8k_compare}
RUN_NAME=${RUN_NAME:-${TS}_trl_gkd}
OUTPUT_DIR=${OUTPUT_DIR:-/work5/jingwut/On-Policy-Distillation/verl/outputs/trl_gkd/${RUN_NAME}}
LOG=${LOG:-/work5/jingwut/On-Policy-Distillation/logs/trl_gkd_${RUN_NAME}.log}

mkdir -p "$(dirname "$LOG")" "$OUTPUT_DIR"

echo "LOG=$LOG"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "NPROC_PER_NODE=$NPROC_PER_NODE CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "PYTHON=$(command -v python)"
python -V

cd "$ROOT"
stdbuf -oL -eL python -m torch.distributed.run --standalone --nproc_per_node "$NPROC_PER_NODE" \
  scripts/trl_gkd/train_trl_gkd_gsm8k.py \
  --student_model "$STUDENT_MODEL" \
  --teacher_model "$TEACHER_MODEL" \
  --train_file "$TRAIN_FILE" \
  --eval_file "$EVAL_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --run_name "$RUN_NAME" \
  --max_steps "$MAX_STEPS" \
  --save_steps "$SAVE_STEPS" \
  --eval_steps "$EVAL_STEPS" \
  --logging_steps "$LOGGING_STEPS" \
  --learning_rate "$LEARNING_RATE" \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
  --max_length "$MAX_LENGTH" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --gamma "$GAMMA" \
  --lmbda "$LMBDA" \
  --beta "$BETA" \
  --dataset_num_proc "$DATASET_NUM_PROC" \
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
  --report_to "$REPORT_TO" \
  --project_name "$WANDB_PROJECT" \
  --gradient_checkpointing \
  2>&1 | tee "$LOG"
