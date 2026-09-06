#!/bin/bash
#SBATCH --job-name=symbolic-bench
#SBATCH --partition=convergence
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gpus=a100_3g.40gb:1
#SBATCH --time=48:00:00
#SBATCH --array=0-7%5
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "Node: $(hostname)"
echo "Workdir: $(pwd)"
echo "Job: ${SLURM_JOB_ID:-NA}"
echo "Array task: ${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-1}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

VENV_DIR="${VENV_DIR:-.venv}"
READY_FILE="${VENV_DIR}/.slearn_experiment_deps_ready"
LOCK_DIR="${VENV_DIR}.lock"

if [[ ! -f "${READY_FILE}" || ! -f "${VENV_DIR}/bin/activate" ]]; then
  if mkdir "${LOCK_DIR}" 2>/dev/null; then
    trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT
    echo "Preparing experiment environment in ${VENV_DIR}."
    bash scripts/install_experiment_deps.sh
    rmdir "${LOCK_DIR}" 2>/dev/null || true
    trap - EXIT
  else
    echo "Another array task is preparing ${VENV_DIR}; waiting for ${READY_FILE}."
    for _ in $(seq 1 180); do
      if [[ -f "${READY_FILE}" && -f "${VENV_DIR}/bin/activate" ]]; then
        break
      fi
      sleep 20
    done
  fi
fi

if [[ ! -f "${READY_FILE}" || ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "Experiment environment is still incomplete after waiting." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "Python: $(which python)"
python --version
python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_TF=1
export USE_TF=0

MODELS="${MODELS:-LSTM GRU minGRU minLSTM Transformer LinearAttention Performer RWKV}"
SYMBOLS="${SYMBOLS:-2 4 6 8}"
COMPLEXITIES="${COMPLEXITIES:-10 30 50 70 90}"
SEQUENCE_LENGTHS="${SEQUENCE_LENGTHS:-3500}"
LAYERS="${LAYERS:-2}"
UNITS="${UNITS:-128}"
D_MODELS="${D_MODELS:-256}"
BATCH_SIZES="${BATCH_SIZES:-128}"
LEARNING_RATES="${LEARNING_RATES:-0.0003}"
WEIGHT_DECAYS="${WEIGHT_DECAYS:-0.01}"
RUNS="${RUNS:-2}"
SEED_COUNT="${SEED_COUNT:-2}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"
PATIENCE="${PATIENCE:-10}"
STOPPING_LOSS="${STOPPING_LOSS:-0.05}"
OUTPUT_DIR="${OUTPUT_DIR:-exps/results_symbolic/slurm_${SLURM_ARRAY_JOB_ID:-manual}}"

python exps/symbolic_sequence_benchmark.py \
  --models ${MODELS} \
  --symbols ${SYMBOLS} \
  --complexities ${COMPLEXITIES} \
  --sequence-lengths ${SEQUENCE_LENGTHS} \
  --window-size 100 \
  --forecast-horizon 100 \
  --seed-count "${SEED_COUNT}" \
  --runs "${RUNS}" \
  --layers ${LAYERS} \
  --units ${UNITS} \
  --d-models ${D_MODELS} \
  --batch-sizes ${BATCH_SIZES} \
  --optimizers AdamW \
  --learning-rates ${LEARNING_RATES} \
  --weight-decays ${WEIGHT_DECAYS} \
  --max-epochs "${MAX_EPOCHS}" \
  --patience "${PATIENCE}" \
  --stopping-loss "${STOPPING_LOSS}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda \
  --task-index "${SLURM_ARRAY_TASK_ID:-0}" \
  --task-count "${SLURM_ARRAY_TASK_COUNT:-1}"

echo "Finished. Results shard: ${OUTPUT_DIR}/results_task_$(printf '%03d' "${SLURM_ARRAY_TASK_ID:-0}").csv"
