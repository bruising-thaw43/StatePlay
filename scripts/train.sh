#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
BASE_MODEL="${STATEPLAY_BASE_MODEL:-${ROOT}/base_model}"
DATA_ROOT="${STATEPLAY_DATA_ROOT:-${ROOT}/data/SF3/train/clips_5s}"
METADATA="${STATEPLAY_METADATA:-${DATA_ROOT}/metadata_state_polish.csv}"
OUTPUT_DIR="${STATEPLAY_OUTPUT:-${ROOT}/outputs/StatePlay}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29520}"
RESERVE_CUDA_MEMORY_GB="${RESERVE_CUDA_MEMORY_GB:-100}"
SAVE_STEPS="${SAVE_STEPS:-500}"
export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_PROCESSES="${#GPU_IDS[@]}"
LAUNCH_ARGS=(--num_processes "${NUM_PROCESSES}" --main_process_port "${MAIN_PROCESS_PORT}")
if (( NUM_PROCESSES > 1 )); then LAUNCH_ARGS+=(--multi_gpu); fi
MODEL_PATHS="[[\"${BASE_MODEL}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors\",\"${BASE_MODEL}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors\",\"${BASE_MODEL}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors\"],\"${BASE_MODEL}/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth\",\"${BASE_MODEL}/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth\"]"
cd "${ROOT}"
"${ACCELERATE_BIN}" launch "${LAUNCH_ARGS[@]}" training/train.py \
  --dataset_base_path "${DATA_ROOT}" --dataset_metadata_path "${METADATA}" \
  --data_file_keys "video,action" --height 480 --width 832 --num_frames 101 --dataset_repeat 1 \
  --state_columns "timer,hp1,hp2,meter1,meter2" --state_norm_max "99,160,160,104,96" \
  --video_loss_weight 1.0 --state_loss_weight 1.0 --state_sampling end --model_paths "${MODEL_PATHS}" \
  --tokenizer_path "${BASE_MODEL}/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl" --learning_rate 5e-5 \
  --num_epochs 10000 --save_steps "${SAVE_STEPS}" --output_path "${OUTPUT_DIR}" \
  --use_gradient_checkpointing --trainable_models "dit" --extra_inputs "input_image" \
  --action_hold_window 10 --action_dropout_prob 0.0 --use_csv_prompt true --prompt_column prompt \
  --prompt_dropout_prob 0.1 --reserve_cuda_memory_gb "${RESERVE_CUDA_MEMORY_GB}" --dataset_num_workers 4 "$@"
