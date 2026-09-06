#!/usr/bin/env bash
set -euo pipefail

RESULTS="${1:-}"
OUTPUT_DIR="${2:-}"

if [[ -z "${RESULTS}" ]]; then
    if [[ -f "exps/results_symbolic/results_merged.csv" ]]; then
        RESULTS="exps/results_symbolic/results_merged.csv"
    elif [[ -f "exps/results_symbolic/results.csv" ]]; then
        RESULTS="exps/results_symbolic/results.csv"
    else
        latest_slurm_dir="$(find exps/results_symbolic -maxdepth 1 -type d -name 'slurm_*' 2>/dev/null | sort | tail -n 1 || true)"
        if [[ -n "${latest_slurm_dir}" && -f "${latest_slurm_dir}/results_merged.csv" ]]; then
            RESULTS="${latest_slurm_dir}/results_merged.csv"
        else
            echo "Could not find a result CSV automatically." >&2
            echo "Usage: bash scripts/run_symbolic_visualizations.sh <results.csv|results_dir> [output_dir]" >&2
            exit 1
        fi
    fi
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
    if [[ "${RESULTS}" == *"/slurm_"* ]]; then
        OUTPUT_DIR="$(dirname "${RESULTS}")/figures"
    else
        OUTPUT_DIR="exps/figures_symbolic"
    fi
fi

PYTHON_BIN="${PYTHON:-python}"
if [[ -x ".venv/bin/python" && -z "${PYTHON:-}" ]]; then
    PYTHON_BIN=".venv/bin/python"
fi

echo "Results: ${RESULTS}"
echo "Output:  ${OUTPUT_DIR}"
echo "Python:  ${PYTHON_BIN}"

"${PYTHON_BIN}" exps/visualize_symbolic_results.py \
    --results "${RESULTS}" \
    --output-dir "${OUTPUT_DIR}"
