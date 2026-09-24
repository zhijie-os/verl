#!/bin/bash
#SBATCH --job-name=opd-rl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# ============================================================
# Usage
# ============================================================
#
#   cd <repo root>  &&  mkdir -p logs  &&  sbatch run_opd_experiment_qwen2.5_1.5B.sbatch <0.6B|1.5B> <experiment>
#
# Paths are taken from the directory you submit from (SLURM_SUBMIT_DIR): the repo is that
# directory, logs go to <repo>/logs (create it once; Slurm needs it before the job starts), the
# virtualenv is ~/verl/VERL, models are looked up in <repo>/models then ~/models, and the GSM8K
# parquet files in <repo>/data/gsm8k then ~/data/gsm8k. Override with VENV_DIR, MODELS_DIR,
# DATA_DIR in the environment. PYTHONPATH is set to the repo, so the checkout you submit from is
# the verl that runs, whichever checkout the venv was pip-installed from.
#
# Original runs
#   baseline        lr 5e-7 everywhere
#   global2x        lr 1e-6 everywhere
#   layerwise       layers 9-18 (all modules) at 2x
#   topk            per-step top-10% SVD gradient amplification (2x wall clock, no gain)
#   both            topk + layerwise
#   adaptive        original adaptive run: top-k gradient energy, min-max -> [1, 2]
#
# Fixed depth-profile ladder (the ceiling every adaptive method must beat)
#   global1.36x     lr 6.8e-7 everywhere (same mean multiplier as layerwise)
#   mid_mlp2x       layers 9-18, MLP matrices only, at 2x
#   mid_attn2x      layers 9-18, attention only, at 2x
#   mid3x           layers 9-18 (all modules) at 3x
#   periph0.5x      middle 1x, everything else (layers 0-8, 19-27, embeddings, lm_head) at 0.5x
#   periph_frozen   middle 1x, everything else frozen (lr 0)
#   gauss2x         smooth Gaussian bump centred on the middle, peak 2x
#
# Adaptive methods (see claude/adaptive-layer-lr-next-steps.md in the project)
#   stability       5.1  direction-stability score (cos of successive per-layer displacements),
#                        prior = layerwise profile, budget-neutral (mean multiplier 1.36)
#                        [first run: cos ~0.5 in every layer -- adjacent Adam windows are momentum-
#                        correlated alike everywhere -- so it behaved like layerwise + noise]
#   stability_cum   5.1  cosine against the accumulated direction W(t-window) - W_0 instead (the
#                        paper's alignment measure), flat profiles stay flat (z floor 0.05)
#   stability_mlp   5.1  stability_cum, scoring / boosting MLP matrices only
#   stability_flat  5.1  stability_cum signal, no prior, mean multiplier 1.0 (pure reallocation)
#   utility         5.2  leave-one-block-out utility probes every 20 updates -> block multipliers
#   extrap          5.3  block-wise EffOPD extrapolation at updates 1, 2, 4, ... (4 blocks)
#                        [first run: fastest start of all runs but KL bound 0.05 let it over-drift]
#   extrap_early    5.3  same with KL bound 0.01, alphas 1,2,4 and no events after update 128
#   extrap_global   5.3  the same with a single block (global EffOPD-style control)
#   extrap_stab     5.3  block-wise extrapolation ordered by the stability score (+ 5.1)
#   fitprofile      5.4  stability signal smoothed by a fitted 3-parameter Gaussian bump
#
# Before the first GPU run of the new code:
#   python -m pytest tests/workers/test_layer_lr_experiments_on_cpu.py -x -q
#
# ============================================================

if [[ $# -ne 2 ]]; then
    echo "Usage: sbatch $0 <0.6B|1.5B> <experiment>   (see the header of this file for the list)"
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

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
VENV_DIR="${VENV_DIR:-${HOME}/verl/VERL}"

if [[ ! -f "${REPO_DIR}/run_opd_experiment_qwen2.5_1.5B.sbatch" ]]; then
    echo "ERROR: submit this script from the repo root (got REPO_DIR=${REPO_DIR})"
    exit 1
fi
if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    echo "ERROR: virtualenv not found at ${VENV_DIR} (set VENV_DIR=/path/to/venv)"
    exit 1
fi

cd "${REPO_DIR}"
source "${VENV_DIR}/bin/activate"

# The checkout you submit from is the verl that runs -- in the driver and in every Ray worker --
# regardless of which directory the venv's editable install points at. (One venv, many branches:
# ~/opd, ~/continual, ...; the job header prints the resolved verl path.)
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

export HYDRA_FULL_ERROR=1

mkdir -p "${REPO_DIR}/logs"

# ============================================================
# Model-dependent settings
# ============================================================

# first existing directory among the candidates (or empty)
first_existing_dir() {
    local d
    for d in "$@"; do
        if [[ -n "${d}" && -d "${d}" ]]; then
            echo "${d}"
            return 0
        fi
    done
    echo ""
}

MODELS_DIR="$(first_existing_dir "${MODELS_DIR:-}" "${REPO_DIR}/models" "${HOME}/opd/models" "${HOME}/models")"
DATA_DIR="$(first_existing_dir "${DATA_DIR:-}" "${REPO_DIR}/data/gsm8k" "${HOME}/opd/data/gsm8k" "${HOME}/data/gsm8k")"

if [[ -z "${MODELS_DIR}" ]]; then
    echo "ERROR: no models directory found (set MODELS_DIR=/path/to/models)"
    exit 1
fi
if [[ -z "${DATA_DIR}" ]]; then
    echo "ERROR: no gsm8k data directory found (set DATA_DIR=/path/to/gsm8k)"
    exit 1
fi

case "${MODEL_SIZE}" in

    0.6B)
        MODEL_PATH="${MODELS_DIR}/Qwen3-0.6B"

        PPO_MICRO_BATCH_SIZE=2
        LOGPROB_MICRO_BATCH_SIZE=4
        ROLLOUT_GPU_MEMORY_UTILIZATION=0.35
        ;;

    1.5B)
        MODEL_PATH="${MODELS_DIR}/Qwen2.5-1.5B-Instruct"

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

SAVE_FREQ=-1             # never save a checkpoint (466 would keep the final actor, e.g. as an OPD teacher)
TEST_FREQ=10             # validate every 10 steps
VAL_BEFORE_TRAIN=True    # validate at step 0

# Optional smoke test of the new machinery without a 5-hour run:
#   SMOKE_STEPS=12 sbatch run_opd_experiment_qwen2.5_1.5B.sbatch 1.5B extrap
# runs 12 steps, skips the step-0 validation and validates every 6 steps.
EXTRA_ARGS=()
if [[ -n "${SMOKE_STEPS:-}" ]]; then
    EXTRA_ARGS+=("trainer.total_training_steps=${SMOKE_STEPS}")
    VAL_BEFORE_TRAIN=False
    TEST_FREQ=$(( SMOKE_STEPS / 2 ))
fi

# Optional independent repeat: SEED=N moves the three seeds that matter for training
# (prompt order: data.seed, default null = PyTorch's fixed generator state; vLLM sampling:
# rollout.seed, default 42; mini-batch shuffling: actor.data_loader_seed, default 42) and logs
# the run as <experiment>_seed<N> next to the original in TensorBoard:
#   SEED=1 sbatch run_opd_experiment_qwen2.5_1.5B.sbatch 1.5B layerwise
SEED="${SEED:-}"
if [[ -n "${SEED}" ]]; then
    EXTRA_ARGS+=("data.seed=${SEED}")
    EXTRA_ARGS+=("actor_rollout_ref.rollout.seed=${SEED}")
    EXTRA_ARGS+=("actor_rollout_ref.actor.data_loader_seed=${SEED}")
fi

# ============================================================
# Experiment defaults (every experiment overrides only what it needs)
# ============================================================

# --- original top-k SVD gradient experiment
TOPK=false
TOPK_SVD_RATIO=0.10
TOPK_SVD_MULTIPLIER=2.0

# --- fixed depth profile
LAYERWISE=false
LAYERWISE_PROFILE=window            # window | gaussian | cosine
LAYERWISE_MODULE_SCOPE=all          # all | mlp | attn
MIDDLE_LR_MULTIPLIER=2.0
MIDDLE_LAYER_START_FRAC=0.3333333333
MIDDLE_LAYER_END_FRAC=0.6666666667
LAYERWISE_CENTER_FRAC=0.5
LAYERWISE_WIDTH_FRAC=0.1666666667
PERIPHERY_LR_MULTIPLIER=1.0         # layers outside the window (0 = frozen)
OTHER_LR_MULTIPLIER=1.0             # embeddings / lm_head / final norm (0 = frozen)

# --- adaptive per-layer multipliers
ADAPTIVE=false
ADAPTIVE_SCORE=topk_energy          # topk_energy | direction_stability
ADAPTIVE_SCORE_SCOPE=all            # all | mlp | attn
ADAPTIVE_APPLY_SCOPE=all            # all | mlp | attn
ADAPTIVE_MAPPING=minmax             # minmax | prior_exp
ADAPTIVE_PRIOR=none                 # none | profile
ADAPTIVE_GAMMA=0.25
ADAPTIVE_TARGET_MEAN=-1.0           # <= 0: no renormalisation
ADAPTIVE_MIN_MULT=0.5
ADAPTIVE_MAX_MULT=2.0
ADAPTIVE_FIT_PROFILE=false
ADAPTIVE_STABILITY_REF=prev_window  # prev_window | cumulative
ADAPTIVE_Z_FLOOR=0.0                # min across-layer std when standardising (0 = plain z-score)
ADAPTIVE_LAYER_LR_INTERVAL=10       # optimizer updates (2 per training step here)
ADAPTIVE_LAYER_LR_TOPK_RATIO=0.10
ADAPTIVE_LAYER_LR_EMA_BETA=0.90

# --- utility probes
UTILITY=false
UTILITY_INTERVAL=20                 # optimizer updates
UTILITY_NUM_BLOCKS=4
UTILITY_MODULE_SCOPE=all
UTILITY_MODE=leave_one_out          # leave_one_out | inject
UTILITY_EMA_BETA=0.7
UTILITY_NORMALIZE=true
UTILITY_GAMMA=0.5
PROBE_OBJECTIVE=adv_logp            # adv_logp | pos_logp

# --- block-wise extrapolation
EXTRAP=false
EXTRAP_SCHEDULE=exp                 # exp | interval
EXTRAP_INTERVAL=50
EXTRAP_FIRST_STEP=1
EXTRAP_MAX_STEP=-1
EXTRAP_NUM_BLOCKS=4
EXTRAP_MODULE_SCOPE=all
EXTRAP_ALPHAS="1,2,4,8"
EXTRAP_TOLERANCE=0.0
EXTRAP_MAX_KL=0.05
EXTRAP_ORDER=middle_out             # middle_out | stability | bottom_up | top_down
EXTRAP_FEEDBACK_LR=false

# ============================================================
# Experiment switches
# ============================================================

# helper: the 5.1 stability configuration shared by several experiments
enable_stability() {
    ADAPTIVE=true
    ADAPTIVE_SCORE=direction_stability
    ADAPTIVE_MAPPING=prior_exp
    ADAPTIVE_PRIOR=profile
    ADAPTIVE_GAMMA=0.25
    ADAPTIVE_TARGET_MEAN=1.36
    ADAPTIVE_MIN_MULT=0.5
    ADAPTIVE_MAX_MULT=3.0
    ADAPTIVE_LAYER_LR_INTERVAL=10
    ADAPTIVE_LAYER_LR_EMA_BETA=0.8
}

case "${EXPERIMENT}" in

    # ---------------- original runs ----------------
    baseline)
        ;;

    global2x)
        ACTOR_LR=1e-6
        ;;

    layerwise)
        LAYERWISE=true
        ;;

    topk)
        TOPK=true
        ;;

    both)
        TOPK=true
        LAYERWISE=true
        ;;

    adaptive)
        ADAPTIVE=true
        ;;

    # ---------------- fixed ladder ----------------
    global1.36x)
        ACTOR_LR=6.8e-7
        ;;

    mid_mlp2x)
        LAYERWISE=true
        LAYERWISE_MODULE_SCOPE=mlp
        ;;

    mid_attn2x)
        LAYERWISE=true
        LAYERWISE_MODULE_SCOPE=attn
        ;;

    mid3x)
        LAYERWISE=true
        MIDDLE_LR_MULTIPLIER=3.0
        ;;

    periph0.5x)
        LAYERWISE=true
        MIDDLE_LR_MULTIPLIER=1.0
        PERIPHERY_LR_MULTIPLIER=0.5
        OTHER_LR_MULTIPLIER=0.5
        ;;

    periph_frozen)
        LAYERWISE=true
        MIDDLE_LR_MULTIPLIER=1.0
        PERIPHERY_LR_MULTIPLIER=0.0
        OTHER_LR_MULTIPLIER=0.0
        ;;

    gauss2x)
        LAYERWISE=true
        LAYERWISE_PROFILE=gaussian
        MIDDLE_LR_MULTIPLIER=2.0
        ;;

    # ---------------- 5.1 direction stability ----------------
    stability)
        enable_stability
        ;;

    stability_cum)
        enable_stability
        ADAPTIVE_STABILITY_REF=cumulative
        ADAPTIVE_Z_FLOOR=0.05
        ;;

    stability_mlp)
        enable_stability
        ADAPTIVE_STABILITY_REF=cumulative
        ADAPTIVE_Z_FLOOR=0.05
        ADAPTIVE_SCORE_SCOPE=mlp
        ADAPTIVE_APPLY_SCOPE=mlp
        ;;

    stability_flat)
        enable_stability
        ADAPTIVE_STABILITY_REF=cumulative
        ADAPTIVE_Z_FLOOR=0.05
        ADAPTIVE_PRIOR=none
        ADAPTIVE_TARGET_MEAN=1.0
        ;;

    # ---------------- 5.2 utility probes ----------------
    utility)
        UTILITY=true
        ADAPTIVE_PRIOR=profile
        ADAPTIVE_TARGET_MEAN=1.36
        ADAPTIVE_MIN_MULT=0.5
        ADAPTIVE_MAX_MULT=3.0
        ;;

    # ---------------- 5.3 block-wise extrapolation ----------------
    extrap)
        EXTRAP=true
        ;;

    extrap_global)
        EXTRAP=true
        EXTRAP_NUM_BLOCKS=1
        ;;

    extrap_early)
        EXTRAP=true
        EXTRAP_ALPHAS="1,2,4"
        EXTRAP_MAX_KL=0.01
        EXTRAP_MAX_STEP=128
        ;;

    extrap_stab)
        EXTRAP=true
        EXTRAP_ALPHAS="1,2,4"
        EXTRAP_MAX_KL=0.01
        EXTRAP_MAX_STEP=128
        EXTRAP_ORDER=stability
        enable_stability
        ADAPTIVE_STABILITY_REF=cumulative
        ADAPTIVE_Z_FLOOR=0.05
        ;;

    # ---------------- 5.4 parametric profile ----------------
    fitprofile)
        enable_stability
        ADAPTIVE_STABILITY_REF=cumulative
        ADAPTIVE_Z_FLOOR=0.05
        ADAPTIVE_FIT_PROFILE=true
        ;;

    *)
        echo "ERROR: unknown EXPERIMENT=${EXPERIMENT} (see the header of this file for the list)"
        exit 1
        ;;
esac

# ============================================================
# Dataset
# ============================================================

TRAIN_FILE="${DATA_DIR}/train.parquet"
VAL_FILE="${DATA_DIR}/test.parquet"

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
if [[ -n "${SEED}" ]]; then
    EXPERIMENT_NAME="${EXPERIMENT_NAME}_seed${SEED}"
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
echo "Repo / venv:           ${REPO_DIR} / ${VENV_DIR}"
echo "Models / data:         ${MODELS_DIR} / ${DATA_DIR}"
echo
echo "Model size:            ${MODEL_SIZE}"
echo "Model path:            ${MODEL_PATH}"
echo
echo "Experiment:            ${EXPERIMENT}"
echo "Base actor LR:         ${ACTOR_LR}"
echo
echo "Top-k SVD enabled:     ${TOPK}  (ratio ${TOPK_SVD_RATIO}, multiplier ${TOPK_SVD_MULTIPLIER})"
echo
echo "Layerwise enabled:     ${LAYERWISE}"
echo "  profile / scope:     ${LAYERWISE_PROFILE} / ${LAYERWISE_MODULE_SCOPE}"
echo "  middle / periphery / other multipliers: ${MIDDLE_LR_MULTIPLIER} / ${PERIPHERY_LR_MULTIPLIER} / ${OTHER_LR_MULTIPLIER}"
echo
echo "Adaptive enabled:      ${ADAPTIVE}"
echo "  score / mapping / prior: ${ADAPTIVE_SCORE} / ${ADAPTIVE_MAPPING} / ${ADAPTIVE_PRIOR}"
echo "  score scope / apply scope: ${ADAPTIVE_SCORE_SCOPE} / ${ADAPTIVE_APPLY_SCOPE}"
echo "  gamma / target mean / min / max: ${ADAPTIVE_GAMMA} / ${ADAPTIVE_TARGET_MEAN} / ${ADAPTIVE_MIN_MULT} / ${ADAPTIVE_MAX_MULT}"
echo "  interval / ema beta / fit profile: ${ADAPTIVE_LAYER_LR_INTERVAL} / ${ADAPTIVE_LAYER_LR_EMA_BETA} / ${ADAPTIVE_FIT_PROFILE}"
echo
echo "Utility probes:        ${UTILITY}  (every ${UTILITY_INTERVAL} updates, ${UTILITY_NUM_BLOCKS} blocks, ${UTILITY_MODE}, objective ${PROBE_OBJECTIVE})"
echo "Block extrapolation:   ${EXTRAP}  (${EXTRAP_SCHEDULE}, alphas ${EXTRAP_ALPHAS}, ${EXTRAP_NUM_BLOCKS} blocks, order ${EXTRAP_ORDER}, max kl ${EXTRAP_MAX_KL})"
echo
echo "Train batch size:      ${TRAIN_BATCH_SIZE}"
echo "PPO mini batch:        ${PPO_MINI_BATCH_SIZE}"
echo "PPO micro batch/GPU:   ${PPO_MICRO_BATCH_SIZE}"
echo "Rollout n:             ${ROLLOUT_N}"
echo "vLLM GPU utilization:  ${ROLLOUT_GPU_MEMORY_UTILIZATION}"
echo "Total epochs:          ${TOTAL_EPOCHS}"
echo "Save freq / test freq: ${SAVE_FREQ} / ${TEST_FREQ}  (val before train: ${VAL_BEFORE_TRAIN})"
echo "Seed:                  ${SEED:-defaults (data null, rollout 42, mini-batch 42)}"
echo "Run name:              ${EXPERIMENT_NAME}"
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
    actor_rollout_ref.actor.optim.layerwise_profile=${LAYERWISE_PROFILE} \
    actor_rollout_ref.actor.optim.layerwise_module_scope=${LAYERWISE_MODULE_SCOPE} \
    actor_rollout_ref.actor.optim.middle_lr_multiplier=${MIDDLE_LR_MULTIPLIER} \
    actor_rollout_ref.actor.optim.middle_layer_start_frac=${MIDDLE_LAYER_START_FRAC} \
    actor_rollout_ref.actor.optim.middle_layer_end_frac=${MIDDLE_LAYER_END_FRAC} \
    actor_rollout_ref.actor.optim.layerwise_center_frac=${LAYERWISE_CENTER_FRAC} \
    actor_rollout_ref.actor.optim.layerwise_width_frac=${LAYERWISE_WIDTH_FRAC} \
    actor_rollout_ref.actor.optim.periphery_lr_multiplier=${PERIPHERY_LR_MULTIPLIER} \
    actor_rollout_ref.actor.optim.other_lr_multiplier=${OTHER_LR_MULTIPLIER} \
    \
    actor_rollout_ref.actor.optim.adaptive_layer_lr_enabled=${ADAPTIVE} \
    actor_rollout_ref.actor.optim.adaptive_layer_lr_score=${ADAPTIVE_SCORE} \
    actor_rollout_ref.actor.optim.adaptive_layer_lr_score_scope=${ADAPTIVE_SCORE_SCOPE} \
    actor_rollout_ref.actor.optim.adaptive_layer_lr_apply_scope=${ADAPTIVE_APPLY_SCOPE} \
    actor_rollout_ref.actor.optim.adaptive_layer_lr_mapping=${ADAPTIVE_MAPPING} \
    actor_rollout_ref.actor.optim.adaptive_layer_lr_prior=${ADAPTIVE_PRIOR} \
    actor_rollout_ref.actor.optim.adaptive_layer_lr_gamma=${ADAPTIVE_GAMMA} \
    actor_rollout_ref.actor.optim.adaptive_layer_lr_target_mean=${ADAPTIVE_TARGET_MEAN} \
    actor_rollout_ref.actor.optim.adaptive_layer_lr_min_multiplier=${ADAPTIVE_MIN_MULT} \
    actor_rollout_ref.actor.optim.adaptive_layer_lr_max_multiplier=${ADAPTIVE_MAX_MULT} \
    actor_rollout_ref.actor.optim.adaptive_layer_lr_fit_profile=${ADAPTIVE_FIT_PROFILE} \
    actor_rollout_ref.actor.optim.adaptive_layer_lr_stability_ref=${ADAPTIVE_STABILITY_REF} \
    actor_rollout_ref.actor.optim.adaptive_layer_lr_z_floor=${ADAPTIVE_Z_FLOOR} \
    actor_rollout_ref.actor.optim.adaptive_layer_lr_interval=${ADAPTIVE_LAYER_LR_INTERVAL} \
    actor_rollout_ref.actor.optim.adaptive_layer_lr_topk_ratio=${ADAPTIVE_LAYER_LR_TOPK_RATIO} \
    actor_rollout_ref.actor.optim.adaptive_layer_lr_ema_beta=${ADAPTIVE_LAYER_LR_EMA_BETA} \
    \
    actor_rollout_ref.actor.optim.utility_probe_enabled=${UTILITY} \
    actor_rollout_ref.actor.optim.utility_probe_interval=${UTILITY_INTERVAL} \
    actor_rollout_ref.actor.optim.utility_probe_num_blocks=${UTILITY_NUM_BLOCKS} \
    actor_rollout_ref.actor.optim.utility_probe_module_scope=${UTILITY_MODULE_SCOPE} \
    actor_rollout_ref.actor.optim.utility_probe_mode=${UTILITY_MODE} \
    actor_rollout_ref.actor.optim.utility_probe_ema_beta=${UTILITY_EMA_BETA} \
    actor_rollout_ref.actor.optim.utility_probe_normalize_by_norm=${UTILITY_NORMALIZE} \
    actor_rollout_ref.actor.optim.utility_probe_gamma=${UTILITY_GAMMA} \
    actor_rollout_ref.actor.optim.probe_objective=${PROBE_OBJECTIVE} \
    \
    actor_rollout_ref.actor.optim.block_extrap_enabled=${EXTRAP} \
    actor_rollout_ref.actor.optim.block_extrap_schedule=${EXTRAP_SCHEDULE} \
    actor_rollout_ref.actor.optim.block_extrap_interval=${EXTRAP_INTERVAL} \
    actor_rollout_ref.actor.optim.block_extrap_first_step=${EXTRAP_FIRST_STEP} \
    actor_rollout_ref.actor.optim.block_extrap_max_step=${EXTRAP_MAX_STEP} \
    actor_rollout_ref.actor.optim.block_extrap_num_blocks=${EXTRAP_NUM_BLOCKS} \
    actor_rollout_ref.actor.optim.block_extrap_module_scope=${EXTRAP_MODULE_SCOPE} \
    "actor_rollout_ref.actor.optim.block_extrap_alphas=\"${EXTRAP_ALPHAS}\"" \
    actor_rollout_ref.actor.optim.block_extrap_tolerance=${EXTRAP_TOLERANCE} \
    actor_rollout_ref.actor.optim.block_extrap_max_kl=${EXTRAP_MAX_KL} \
    actor_rollout_ref.actor.optim.block_extrap_order=${EXTRAP_ORDER} \
    actor_rollout_ref.actor.optim.block_extrap_feedback_lr=${EXTRAP_FEEDBACK_LR} \
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
    trainer.save_freq=${SAVE_FREQ} \
    trainer.val_before_train=${VAL_BEFORE_TRAIN} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo
echo "============================================================"
echo "Training completed successfully."
echo "============================================================"
