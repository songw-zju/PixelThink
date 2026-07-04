#!/usr/bin/env bash
set -xeuo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
if [ -n "${VENV_PATH:-}" ]; then
    source "${VENV_PATH}/bin/activate"
elif [ -f "$ROOT/.venv/bin/activate" ]; then
    source "$ROOT/.venv/bin/activate"
fi
export VLLM_USE_V1=1

MODEL_PATH=${MODEL_PATH:-$ROOT/models/Qwen3-VL-4B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-$ROOT/data/train.parquet}
REWARD_MGR=${REWARD_MGR:-$ROOT/pixelthink_reward_manager.py}
LENGTH_COUNTING_MODE=${LENGTH_COUNTING_MODE:-reward_word_tokens}
export PIXELTHINK_REWARD_LENGTH_MODE="${LENGTH_COUNTING_MODE}"

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
ACTOR_LR=${ACTOR_LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-5e-3}
ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.55}
ROLLOUT_N=${ROLLOUT_N:-8}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
SAVE_FREQ=${SAVE_FREQ:-50}
MAX_STEPS=${MAX_STEPS:--1}
N_GPUS=${N_GPUS:-4}

PROJECT_NAME=${PROJECT_NAME:-pixelthink_qwen3vl}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3vl_4b_pixelthink_$(date +%m%d_%H%M)}

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TRAIN_FILE}"
    data.image_key=images
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
)
MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_fused_kernels=True
)
ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)
ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
)
REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)
REWARD=(
    reward.reward_manager.source=importlib
    reward.reward_manager.name=PixelThinkRewardManager
    reward.reward_manager.module.path="${REWARD_MGR}"
)
TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${N_GPUS}
    trainer.nnodes=1
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=100000
    trainer.val_before_train=False
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.default_local_dir="${ROOT}/workdir/${EXPERIMENT_NAME}"
)
EXTRA=()
if [ "${MAX_STEPS}" -gt 0 ] 2>/dev/null; then EXTRA+=(trainer.total_training_steps=${MAX_STEPS}); fi

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" "${MODEL[@]}" "${ACTOR[@]}" "${ROLLOUT[@]}" "${REF[@]}" \
    "${REWARD[@]}" "${TRAINER[@]}" "${EXTRA[@]}" "$@"
