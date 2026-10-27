#!/bin/bash
#SBATCH --job-name=qwen-summary-smoke
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=qwen-summary-%j.log
set -euo pipefail

# Submit from an existing source checkout on Brown. Results are copied back to it.
SOURCE_DIR="${SLURM_SUBMIT_DIR:?Submit using sbatch from the repository root}"
TASK_DIR=$(mktemp -d /tmp/qwen-summary-XXXXXX)
export HF_HOME="$TASK_DIR/hf-cache"
export UV_CACHE_DIR="$TASK_DIR/uv-cache"
export UV_PYTHON_INSTALL_DIR="$TASK_DIR/python"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$SOURCE_DIR/src"
cd "$TASK_DIR"
uv venv .venv
uv pip install --python .venv/bin/python 'torch>=2.6,<3' 'transformers>=4.46,<5' 'accelerate>=1,<2'
mkdir -p "$SOURCE_DIR/experiments/brown-$SLURM_JOB_ID"
RESULT_DIR="$SOURCE_DIR/experiments/brown-$SLURM_JOB_ID"
uv pip freeze --python .venv/bin/python > "$RESULT_DIR/environment.txt"
.venv/bin/python -m unittest discover -s "$SOURCE_DIR/tests" -v
.venv/bin/python -m cuda_summary.benchmark \
  --input "$SOURCE_DIR/data/example.jsonl" \
  --batch-size 2 --max-new-tokens 128 --repeats 3 \
  --output "$RESULT_DIR/gpu-smoke-results.json"
# Temporary model/dependency caches intentionally remain for explicit cleanup.
echo "Results: $RESULT_DIR"
echo "Temporary cache: $TASK_DIR"
