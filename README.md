# CUDA Document Summarizer

A foundation for GPU-accelerated technical-document summarization using
`Qwen/Qwen2.5-3B-Instruct`, PyTorch and Transformers. Produces summaries with
Objectives, Methods, Quantitative findings and Limitations sections.

**Status:** real Qwen inference has now run on a Brown Slurm NVIDIA L40.
All four benchmark configurations completed on the two-document smoke fixture.
Scheduling tests passed locally and on the cluster. See the
[Brown GPU report](docs/brown-gpu.md) for raw measured results and limitations.
The original résumé claim of 25% throughput improvement is not established;
400 real reports and 100 source-backed quality reviews have not been run.

## Run without GPU dependencies

Python 3.11+:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 experiments/padding.py
```

The experiment samples 400 synthetic token-length integers from 1K–24K; it
compares padded positions at batch sizes 1, 4, 8 and 16. It is not a document
corpus or GPU performance simulation. See [results](docs/experiments.md).

## Run actual inference on an NVIDIA CUDA machine

Create an isolated environment; install a CUDA-compatible PyTorch distribution
for your machine, then install the project GPU dependencies:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[gpu]'
python -m cuda_summary.benchmark \
  --input data/example.jsonl \
  --batch-size 2 --max-new-tokens 256 --repeats 3 \
  --output experiments/gpu-results.json
```

The command downloads model weights from Hugging Face on first execution. It uses
FP16 on the default CUDA device. Memory requirements depend on model, context,
batch size and KV cache; begin with the short sample and a small batch. No paid
GPU provisioning is included. Dependency ranges are declared, not a validated
lockfile; capture `pip freeze` for real runs.

Input JSONL requires unique nonempty `id` and `text` strings. Samples in `data/`
are explicitly fictional. Documents exceeding the model context budget are
rejected without truncation. Chunking and hierarchical summarization remain
future work. This runner is offline batch processing, not an online scheduler.

## Benchmark design

- Same input, generation cap, greedy decoding and batch size in four conditions:
  arrival/length ordering × KV cache off/on.
- Chat template token lengths drive batching; left padding and attention masks
  support decoder-only generation. Output is restored to original document order.
- Longest batch warm-up precedes each condition; CUDA synchronization surrounds
  timed batches; model load/tokenization and summary text decoding are excluded.
- Timings include GPU input transfer and generated-token transfer to CPU. Batch
  p50/p95 latency is reported separately from whole-workload throughput. It is not
  online end-to-end document latency.
- Per-repeat docs/s, generated tokens/s, total time and peak PyTorch-allocated
  memory are recorded. Peak allocation is not total device memory.
- Record model revision, library versions, GPU, CUDA, input/prompt hashes and seed.
  Supply `--revision <commit>` for reproducibility; default main can change.

The runner delegates KV caching to Transformers via `use_cache`; it does not
implement a custom CUDA kernel or custom cache. Actual outputs may differ across
hardware or batch shapes. Quality parity must be checked before claiming gains.

## Quality evaluation

Use [the review rubric](docs/quality.md) to review real generated summaries.
There is no automatic factuality claim based on word overlap. Do not treat the
synthetic input documents as 400 real benchmark documents or 100 reviewed summaries.

## References

- [Qwen model card and chat-template usage](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [Transformers generation and left padding](https://huggingface.co/docs/transformers/llm_tutorial)
- [Transformers KV cache](https://huggingface.co/docs/transformers/kv_cache)

## Legal-document scheduling PoC

The [legal scheduling study](docs/legal-scheduling-poc.md) compares arrival/fixed-4,
length/fixed-4, length/fixed-8 and token-budget batching on full ContractNLI NDAs.
It fixes model, prompt and decoding, records three repeated runs and checks output
hashes. This is offline scheduling research, not validated legal advice or a trained
legal model. Token-slot limits are a scheduling proxy, not a hard VRAM guarantee.
