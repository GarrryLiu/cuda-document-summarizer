#!/bin/bash
#SBATCH --job-name=legal-batch-poc
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx_a6000
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
set -euo pipefail
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
if [[ ! -x "$UV_BIN" ]]; then echo "uv executable unavailable: $UV_BIN" >&2; exit 2; fi
SOURCE_DIR="${SLURM_SUBMIT_DIR:?Submit from staged source directory}"
RESULT_DIR="$SOURCE_DIR/results-$SLURM_JOB_ID"
mkdir -p "$RESULT_DIR"
TASK_DIR=$(mktemp -d /tmp/legal-batch-XXXXXX)
export HF_HOME="$TASK_DIR/hf-cache" UV_CACHE_DIR="$TASK_DIR/uv-cache"
export UV_PYTHON_INSTALL_DIR="$TASK_DIR/python" TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$SOURCE_DIR/src"
cleanup() {
  # Only remove the private temporary directory created by this job.
  if [[ "$TASK_DIR" == /tmp/legal-batch-* ]]; then rm -rf -- "$TASK_DIR"; fi
}
trap cleanup EXIT
cd "$TASK_DIR"
"$UV_BIN" venv --python 3.12 .venv
"$UV_BIN" pip install --python .venv/bin/python 'torch==2.14.1' 'transformers==4.57.6' 'accelerate>=1,<2'
"$UV_BIN" pip freeze --python .venv/bin/python > "$RESULT_DIR/environment.txt"
PYTHONPATH="$SOURCE_DIR/src" .venv/bin/python -m unittest discover -s "$SOURCE_DIR/tests" -v
.venv/bin/python -m cuda_summary.legal_benchmark \
  --input "$SOURCE_DIR/data/local/legal-poc.jsonl" \
  --output "$RESULT_DIR/raw-results.json" --batch-size 4 --max-sequences 8 \
  --token-budgets 16384 24576 32768 49152 --max-new-tokens 512 --repeats 3
printf 'Results: %s\n' "$RESULT_DIR"
