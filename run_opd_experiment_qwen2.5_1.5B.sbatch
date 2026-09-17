(ENV) zhijiex@vulcan3:~/verl$ cat run_opd_experiment.sbatch
#!/bin/bash
#SBATCH --job-name=opd-rl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --time=24:00:00
#SBATCH --output=/home/zhijiex/verl/logs/%x-%j.out
#SBATCH --error=/home/zhijiex/verl/logs/%x-%j.err

set -euo pipefail

# ============================================================
# Usage
# ============================================================
#
# Qwen3-0.6B:
#   sbatch run_opd_experiment.sbatch 0.6B baseline
#   sbatch run_opd_experiment.sbatch 0.6B layerwise
#   sbatch run_opd_experiment.sbatch 0.6B topk
#   sbatch run_opd_experiment.sbatch 0.6B both
#
# Qwen2.5-1.5B-Instruct:
#   sbatch run_opd_experiment.sbatch 1.5B baseline
#   sbatch run_opd_experiment.sbatch 1.5B layerwise
#   sbatch run_opd_experiment.sbatch 1.5B topk
#   sbatch run_opd_experiment.sbatch 1.5B both
#
# ============================================================

if [[ $# -ne 2 ]]; then
    echo "Usage: sbatch $0 <0.6B|1.5B> <baseline|layerwise|topk|both>"
    exit 1
fi

MODEL_SIZE="$1"
EXPERIMENT="$2"

# ============================================================
# Alliance modules
# IMPORTANT: load BEFORE activating virtualenv
# ============================================================

module load python/3.11
module load gcc
module load arrow/25.0.0
module load opencv/4.14.0

# ============================================================
# VERL environment
# ============================================================

cd /home/zhijiex/verl
source /home/zhijiex/verl/VERL/bin/activate

export HYDRA_FULL_ERROR=1

mkdir -p /home/zhijiex/verl/logs

# ============================================================
# Model-dependent settings
# ============================================================

case "${MODEL_SIZE}" in

    0.6B)
        MODEL_PATH="/home/zhijiex/models/Qwen3-0.6B"

        PPO_MICRO_BATCH_SIZE=2
        LOGPROB_MICRO_BATCH_SIZE=4
        ROLLOUT_GPU_MEMORY_UTILIZATION=0.35
        ;;

    1.5B)
        MODEL_PATH="/home/zhijiex/models/Qwen2.5-1.5B-Instruct"

        PPO_MICRO_BATCH_SIZE=2
        LOGPROB_MICRO_BATCH_SIZE=4
        ROLLOUT_GPU_MEMORY_UTILIZATION=0.35
        ;;

    *)
        echo "ERROR: unknown MODEL_SIZE=${MODEL_SIZE}"
        echo "Expected: 0.6B or 1.5B"
        exit 1
        ;;
esac

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: model directory not found:"
    echo "${MODEL_PATH}"
    exit 1
fi

# ============================================================
# Experiment switches
# ============================================================

TOPK=false
LAYERWISE=false

case "${EXPERIMENT}" in

    baseline)
        TOPK=false
        LAYERWISE=false
        ;;

    layerwise)
        TOPK=false
        LAYERWISE=true
        ;;

    topk)
        TOPK=true
        LAYERWISE=false
        ;;

    both)
        TOPK=true
        LAYERWISE=true
        ;;

    *)
        echo "ERROR: unknown EXPERIMENT=${EXPERIMENT}"
        echo "Expected: baseline, layerwise, topk, or both"
        exit 1
        ;;
esac

# ============================================================
# Dataset
# ============================================================

TRAIN_FILE="/home/zhijiex/data/gsm8k/train.parquet"
VAL_FILE="/home/zhijiex/data/gsm8k/test.parquet"

if [[ ! -f "${TRAIN_FILE}" ]]; then
    echo "ERROR: train dataset not found:"
    echo "${TRAIN_FILE}"
    exit 1
fi

if [[ ! -f "${VAL_FILE}" ]]; then
    echo "ERROR: validation dataset not found:"
    echo "${VAL_FILE}"
    exit 1
fi

# ============================================================
# Common training settings
# ============================================================

ACTOR_LR=5e-7

TRAIN_BATCH_SIZE=32
PPO_MINI_BATCH_SIZE=16

MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=1024

ROLLOUT_N=4

# 1 epoch ~= 233 steps
# 2 epochs ~= 466 steps
TOTAL_EPOCHS=2

# ============================================================
# Experimental feature settings
# ============================================================

TOPK_SVD_RATIO=0.10
TOPK_SVD_MULTIPLIER=2.0

MIDDLE_LR_MULTIPLIER=2.0
MIDDLE_LAYER_START_FRAC=0.3333333333
MIDDLE_LAYER_END_FRAC=0.6666666667

# Both Qwen3-0.6B and Qwen2.5-1.5B-Instruct have 28 transformer layers.
#
# floor(28 * 1/3) = 9
# ceil (28 * 2/3) = 19
#
# Middle layers = 9 ... 18

if [[ "${MODEL_SIZE}" == "1.5B" ]]; then
    EXPERIMENT_NAME="${EXPERIMENT}_qwen2.5_${MODEL_SIZE}_instruct"
else
    EXPERIMENT_NAME="${EXPERIMENT}_qwen3_${MODEL_SIZE}"
fi

# ============================================================
# Environment information
# ============================================================

echo
echo "============================================================"
echo "                  OPD-RL EXPERIMENT"
echo "============================================================"
echo "SLURM job ID:          ${SLURM_JOB_ID}"
echo "Hostname:              $(hostname)"
echo
echo "Model size:            ${MODEL_SIZE}"
echo "Model path:            ${MODEL_PATH}"
echo
echo "Experiment:            ${EXPERIMENT}"
echo "Top-k SVD enabled:     ${TOPK}"
echo "Layerwise LR enabled:  ${LAYERWISE}"
echo
echo "Base actor LR:         ${ACTOR_LR}"
echo "Middle LR multiplier:  ${MIDDLE_LR_MULTIPLIER}"
echo
echo "Top-k ratio:           ${TOPK_SVD_RATIO}"
echo "Top-k multiplier:      ${TOPK_SVD_MULTIPLIER}"
echo
echo "Train batch size:      ${TRAIN_BATCH_SIZE}"
echo "PPO mini batch:        ${PPO_MINI_BATCH_SIZE}"
echo "PPO micro batch/GPU:   ${PPO_MICRO_BATCH_SIZE}"
echo "Rollout n:             ${ROLLOUT_N}"
echo "vLLM GPU utilization:  ${ROLLOUT_GPU_MEMORY_UTILIZATION}"
echo "Total epochs:          ${TOTAL_EPOCHS}"
echo "Expected steps:        ~466"
echo
echo "Train dataset:         ${TRAIN_FILE}"
echo "Validation dataset:    ${VAL_FILE}"
echo "============================================================"
echo

# ============================================================
# Verify environment
# ============================================================

echo "Loaded modules:"
module list

echo
echo "Python:"
which python
python --version

echo
echo "Installed components:"

python - <<'PY'
import torch
import vllm
import verl
import pyarrow
import cv2

print()
print("Torch:          ", torch.__version__)
print("Torch CUDA:     ", torch.version.cuda)
print("CUDA available: ", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU count:      ", torch.cuda.device_count())
    print("GPU:            ", torch.cuda.get_device_name(0))

print("vLLM:           ", vllm.__version__)
print("VERL:           ", verl.__file__)
print("PyArrow:        ", pyarrow.__version__)
print("OpenCV:         ", cv2.__version__)
print()
PY

echo
echo "TensorBoard:"
python - <<'PY'
try:
    import tensorboard
    print("TensorBoard:", tensorboard.__version__)
except ImportError:
    raise SystemExit(
        "ERROR: tensorboard is not installed in the VERL environment."
    )
PY

echo
echo "nvidia-smi:"
nvidia-smi

echo
echo "============================================================"
echo "Starting training..."
echo "============================================================"
echo

# ============================================================
# Training
# ============================================================

srun env \
    -u ROCR_VISIBLE_DEVICES \
    -u HIP_VISIBLE_DEVICES \
    python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=1 \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.actor.optim.topk_svd_enabled=${TOPK} \
    actor_rollout_ref.actor.optim.topk_svd_ratio=${TOPK_SVD_RATIO} \
    actor_rollout_ref.actor.optim.topk_svd_multiplier=${TOPK_SVD_MULTIPLIER} \
    \
    actor_rollout_ref.actor.optim.layerwise_lr_enabled=${LAYERWISE} \
    actor_rollout_ref.actor.optim.middle_lr_multiplier=${MIDDLE_LR_MULTIPLIER} \
    actor_rollout_ref.actor.optim.middle_layer_start_frac=${MIDDLE_LAYER_START_FRAC} \
    actor_rollout_ref.actor.optim.middle_layer_end_frac=${MIDDLE_LAYER_END_FRAC} \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOGPROB_MICRO_BATCH_SIZE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION} \
    actor_rollout_ref.rollout.max_model_len=2048 \
    actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOGPROB_MICRO_BATCH_SIZE} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=opd_rl \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.val_before_train=True \
    trainer.test_freq=10 \
    trainer.total_epochs=${TOTAL_EPOCHS}

echo
echo "============================================================"
echo "Training completed successfully."
echo "============================================================"