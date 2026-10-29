#!/bin/bash
#SBATCH --job-name=decode-kernel
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
set -euo pipefail
SOURCE_DIR="${SLURM_SUBMIT_DIR:?Submit from staged checkout}"
RESULT_DIR="$SOURCE_DIR/results-$SLURM_JOB_ID"
mkdir -p "$RESULT_DIR"
TASK_DIR=$(mktemp -d /tmp/decode-kernel-XXXXXX)
export HF_HOME="$TASK_DIR/hf" UV_CACHE_DIR="$TASK_DIR/uv" UV_PYTHON_INSTALL_DIR="$TASK_DIR/python"
export TORCH_EXTENSIONS_DIR="$TASK_DIR/extensions" MAX_JOBS=2 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$SOURCE_DIR/src"
cleanup() { if [[ "$TASK_DIR" == /tmp/decode-kernel-* ]]; then rm -rf -- "$TASK_DIR"; fi; }
trap cleanup EXIT
cd "$TASK_DIR"
UV_BIN="$HOME/.local/bin/uv"
"$UV_BIN" venv --python 3.12 .venv
"$UV_BIN" pip install --python .venv/bin/python 'torch==2.14.1' 'transformers==4.57.6' 'accelerate==1.15.0' ninja 'nvidia-cuda-nvcc==13.0.88' 'nvidia-cuda-crt==13.0.88' 'nvidia-cuda-cccl==13.0.85' 'nvidia-nvvm==13.0.88'
"$UV_BIN" pip freeze --python .venv/bin/python > "$RESULT_DIR/environment.txt"
# NVIDIA's CUDA 13 wheels share one nvidia/cu13 prefix.
export CUDA_HOME="$(.venv/bin/python -c 'import site; print(site.getsitepackages()[0]+"/nvidia/cu13")')"
export PATH="$CUDA_HOME/bin:$TASK_DIR/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
"$CUDA_HOME/bin/nvcc" --version > "$RESULT_DIR/nvcc.txt"
nvidia-smi > "$RESULT_DIR/nvidia-smi.txt"
cd "$SOURCE_DIR"
"$TASK_DIR/.venv/bin/python" -m unittest discover -s tests -v
"$TASK_DIR/.venv/bin/python" experiments/kernels/check.py "$RESULT_DIR/correctness.json"
"$TASK_DIR/.venv/bin/python" experiments/kernels/benchmark.py --output "$RESULT_DIR/benchmark.json"
"$TASK_DIR/.venv/bin/python" experiments/kernels/profile_model.py --input data/local/legal-poc.jsonl --output "$RESULT_DIR/model-profile.json"
printf 'Results: %s\n' "$RESULT_DIR"
