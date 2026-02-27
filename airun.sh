#!/bin/bash
#SBATCH --account=ai4wy-eap
#SBATCH --partition=ai4wy
#SBATCH --job-name=stable_fm
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ---- GPU choice (AI4WY has H200 nodes with either 1 or 2 GPUs)
# Use ONE of these:
#SBATCH --gres=gpu:h200:2
##SBATCH --gres=gpu:h200:1

# ---- CPU / memory
# On 2xH200 nodes you have 144 CPUs and ~1.1 TB RAM available.
# On 1xH200 nodes you have 72 CPUs and ~572 GB RAM available.
#SBATCH --cpus-per-task=64
#SBATCH --mem=0                 # take all memory on the node (safe w/ exclusive)
#SBATCH --exclusive             # full node: best for big DDPM runs

# ---- Time (AI4WY partition shows 1-00:00:00)
#SBATCH --time=1-00:00:00

# ---- Logs
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err

set -euo pipefail

echo "JOB START  : $(date)"
echo "HOST       : $(hostname)"
echo "WORKDIR    : $(pwd)"
echo "SLURM JOB  : ${SLURM_JOB_ID}"
echo "GPUS       : ${SLURM_GPUS:-unset}"
echo "CUDA VIS   : ${CUDA_VISIBLE_DEVICES:-unset}"

# --- (Optional) modules, only if your site requires them
# module load arcc gcc slurm 2>/dev/null || true
# module load cuda 2>/dev/null || true

# --- Activate your “always use” Python env (Miniforge example)
if [ -f "$HOME/opt/miniforge3/etc/profile.d/conda.sh" ]; then
  . "$HOME/opt/miniforge3/etc/profile.d/conda.sh"
  conda activate ai4wy
fi

# --- Sanity checks (won’t crash if nvidia-smi unavailable on login image)
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('ngpu', torch.cuda.device_count())" || true

# ---- Paths on AI4WY
DATA_DIR="/project/ai4wy-eap/nmille39/PRIMVS/light_curves"
FITS_FILE="/project/ai4wy-eap/nmille39/PRIMVS/catalog/PRIMVS_P.fits"

# ---- Run
# For multi-GPU DDPM you typically want torchrun (DDP). This works for 1 or 2 GPUs.
NGPUS="${SLURM_GPUS_ON_NODE:-1}"

torchrun --standalone --nproc_per_node="${NGPUS}" train.py \
  --milestone 0 \
  --data_dir "${DATA_DIR}" \
  --fits_file "${FITS_FILE}" \
  --fits_id_column "sourceid" \
  --band Ks \
  --lambda_z 2.0 \
  --lambda_tau 1.0 \
  --loss_type l2

echo "JOB END    : $(date)"