#!/bin/bash
#SBATCH --job-name=flash-attn-build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/home/zhijiex/verl/logs/flash-build-%j.out
#SBATCH --error=/home/zhijiex/verl/logs/flash-build-%j.err

set -euo pipefail

mkdir -p /home/zhijiex/verl/logs

# Alliance modules required by your environment
module load python/3.11
module load gcc
module load arrow/25.0.0
module load opencv/4.14.0

cd /home/zhijiex/verl
source /home/zhijiex/verl/VERL/bin/activate

echo "============================================================"
echo "Existing PyTorch"
echo "============================================================"

python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CXX11 ABI:", torch._C._GLIBCXX_USE_CXX11_ABI)
PY

# Get CUDA version expected by the installed PyTorch
TORCH_CUDA=$(python - <<'PY'
import torch
print(torch.version.cuda)
PY
)

echo "PyTorch expects CUDA ${TORCH_CUDA}"

# Temporarily leave the virtualenv while loading CUDA toolchain
deactivate

echo "Loading CUDA ${TORCH_CUDA}..."

if ! module load cuda/"${TORCH_CUDA}"; then
    echo
    echo "ERROR: cuda/${TORCH_CUDA} module was not found."
    echo "Available CUDA modules:"
    module spider cuda
    exit 1
fi

# Reactivate environment after all modules are loaded
source /home/zhijiex/verl/VERL/bin/activate

echo
echo "CUDA compiler:"
which nvcc
nvcc --version

# ------------------------------------------------------------
# Build settings
# ------------------------------------------------------------

# L40S = Ada, compute capability 8.9.
# Avoid compiling kernels for GPUs we don't use.
export TORCH_CUDA_ARCH_LIST="8.9"

# Keep compilation RAM usage controlled.
export MAX_JOBS=8

# Build temporary files on node-local storage,
# not your 40 GB home allocation.
export TMPDIR="${SLURM_TMPDIR}"
export PIP_CACHE_DIR="${SLURM_TMPDIR}/pip-cache"

mkdir -p "${TMPDIR}"
mkdir -p "${PIP_CACHE_DIR}"

echo
echo "SLURM_TMPDIR: ${SLURM_TMPDIR}"
echo "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"
echo "MAX_JOBS: ${MAX_JOBS}"

# Remove the incompatible binary we installed previously.
python -m pip uninstall -y flash-attn || true

# Lightweight build requirements.
# --no-deps prevents pip from touching Torch again.
python -m pip install \
    --no-deps \
    ninja \
    packaging \
    psutil

echo
echo "============================================================"
echo "Building FlashAttention 2.8.3 from source"
echo "============================================================"

python -m pip install \
    --no-build-isolation \
    --no-deps \
    --no-cache-dir \
    --no-binary=flash-attn \
    "flash-attn==2.8.3"

echo
echo "============================================================"
echo "Testing installation"
echo "============================================================"

python - <<'PY'
import torch
import flash_attn
from flash_attn import flash_attn_func

print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("FlashAttention:", flash_attn.__version__)
print("FlashAttention import SUCCESS")
PY

echo
echo "pip check:"
python -m pip check

echo
echo "============================================================"
echo "FlashAttention build completed"
echo "============================================================"
