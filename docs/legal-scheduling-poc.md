# Legal document GPU scheduling PoC

## Research question and architecture

Can offline input-length-aware token-slot packing improve completion time and
padding efficiency for legal-summary inference under an explicit sequence-slot
budget? This optimizes serving/batching around a fixed model; it does not train
a new legal model or modify CUDA kernels.

Pipeline:

```text
ContractNLI dev NDA full texts
 -> deterministic length-spread sample + source manifest
 -> identical legal prompt and chat-template tokenization
 -> four scheduling plans
 -> fixed Qwen FP16 + KV-cache generation on one GPU
 -> per-run timing, allocation, padding, output hashes and cap diagnostics
```

## Dataset

24 full NDAs from the ContractNLI development split, selected at character-length
quantiles from documents with 2,000–24,000 characters, then restored to the
original dataset order. No repetition, truncation, oracle output-length feature or
annotation leakage is used in the scheduler. This is a deliberately length-spread
offline sample, not a representative production arrival distribution. Documents
outside the character range are excluded before selection. Actual prompt tokens
are recorded after tokenization; context overflow is rejected.

Source and attribution: Yuta Koreeda and Christopher D. Manning, 2021,
[ContractNLI](https://stanfordnlp.github.io/contract-nli/),
[paper](https://aclanthology.org/2021.findings-emnlp.164/), CC BY 4.0.
Dataset ZIP: https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip.
ContractNLI is an NLI/evidence dataset, not a summarization reference corpus; its
labels are not used as summary scores. Original contract text and full generated
summaries remain local. Public manifest retains source URLs, IDs and hashes.

## Policies

1. Arrival, fixed 4 sequences.
2. Length sort, fixed 4.
3. Length sort, fixed 8: a concurrency control, not a new scheduler.
4. Length sort and greedy token-budget packing, at most 8 sequences.

The fourth plan enforces:

`batch_size * (round_up(max_input_tokens, 8) + max_new_tokens) <= 32768`.

This reserves padded sequence slots through the output cap. It does not account
for model weights, activations, attention workspace or allocator fragmentation;
it is not a hard VRAM cap. Fixed baselines do not enforce the slot constraint.
All conditions use the same physical GPU; compare measured peak memory as well
as throughput. A fair same-memory frontier requires further budget sweeps.

The scheduler is offline and sorts the full workload. It does not model request
arrival times, fairness, online queue waiting, continuous batching or cancellation.
No output-length predictor has been trained. All plans use the same fixed reserve.

## Experiment controls

Frozen Qwen2.5-3B-Instruct revision, FP16, same legal prompt, greedy decoding,
512 new-token cap, left padding to a multiple of 8, KV cache always on. Fixed
sampling and conditions; three repetitions in seeded shuffled order within each
round. Each strategy receives a short longest-shape warmup. Model load,
tokenization, offline planning and output decoding/hashing are excluded from
reported generation time; input/generated-token transfers are included.

Report all individual runs, median time and tokens/s, maximum PyTorch-allocated
and reserved memory, padded input positions, actual output-token totals, exact
output hash matches, cap hits and document coverage. Batch latencies are not
online document response latencies. Peak allocated memory is not total device
memory. Repeated runs on one shared node are not independent hardware trials.

If output lengths or hashes change across batch shapes, speed cannot be described
as quality-preserving without separate semantic evaluation. Exact equality is a
strong consistency check; inequality does not itself prove factual degradation.

## Reproduction

Download the official ZIP locally and extract its `dev.json` to
`data/local/contract-dev.json` (do not commit the original text).

```sh
python3 experiments/legal/prepare.py
PYTHONPATH=src python3 -m cuda_summary.legal_benchmark \
  --input data/local/legal-poc.jsonl --batch-size 4 --max-sequences 8 \
  --token-budget 32768 --max-new-tokens 512 --repeats 3
python3 experiments/legal/analyze.py experiments/legal/raw-results.json
```

Requires the project's GPU dependencies and an NVIDIA CUDA machine. Results
checkpoint after each strategy run; interrupted experiments retain partial data
but are not automatically resumed. OOM is not silently hidden by splitting batches.

## Durable school execution

The original interactive L40 session ended before results were retrieved. Its
node-local checkpoints were not accessible while all GPUs on that node were in
use, so that attempt is excluded from final comparisons. A complete new run uses
one RTX A6000 via Slurm batch job; no L40/A6000 measurements are pooled.

`scripts/brown-legal.sbatch` stages dependencies/model cache in a job-local `/tmp`
directory, checkpoints predictions directly to a private shared result directory,
and removes only its own generated temporary cache at exit. Submit from a staged
source checkout containing the local input JSONL. Set the Slurm output path to a
shared directory as well. This survives terminal/chat disconnection. Do not
publish raw contract text or predictions by accident; Git retains manifests and
sanitized aggregate metrics.

The batch script explicitly requests Python 3.12 and uses the absolute user-local
`uv` executable (overridable with `UV_BIN`). The cluster's batch PATH differs from
the interactive shell and otherwise selects Python 3.7. Bootstrap failures before
generation are excluded from timing results. Keep dependencies and model cache
out of the shared home filesystem to avoid exhausting its quota.

## Measured result (2026-10-28)

Slurm job 77268 completed successfully on one NVIDIA RTX A6000 (48 GB),
Python 3.12.3, PyTorch 2.14.1+cu130, CUDA 13.0 and Transformers 4.57.6.
The 24 prompts span 548–4,717 tokens (54,097 total). Every one of the 12 runs
processed all 24 documents; none ran out of memory or hit the generation cap.
Results below use median generation time across three randomized repetitions.
Memory is the maximum PyTorch-allocated peak across those repetitions.

| Policy | Time (s) | Docs/s | Throughput vs arrival | Allocated GiB | Padding |
| --- | ---: | ---: | ---: | ---: | ---: |
| Arrival, 4 | 58.55 | 0.410 | 1.000× | 8.05 | 34.45% |
| Length, 4 | 51.31 | 0.468 | 1.141× | 8.05 | 11.35% |
| Length, 8 | 41.78 | 0.574 | 1.401× | 10.24 | 23.71% |
| Token budget, up to 8 | 45.37 | 0.529 | 1.290× | 8.26 | 14.71% |

Token-budget packing increased observed document throughput by 29.0% relative
to arrival-4, reducing completion time by 22.5%, with 2.6% more peak allocated
memory. Fixed length-8 was faster: token-budget packing used 19.3% less peak
allocated memory but took 8.6% longer. This is a speed/memory tradeoff, not a
claim that the token-budget scheduler is universally fastest. Length-4 alone
improved throughput by 14.1% at the same measured peak allocation as arrival-4.

All three repetitions of each policy produced identical token hashes within that
policy. Across policies, only 18/24 (length-4), 16/24 (length-8), and 17/24
(token budget) matched arrival-4 exactly. Generated-token totals were respectively
4,871, 4,866 and 4,854 versus 4,877 for arrival-4. Thus measured throughput is
not evidence of semantic quality equivalence. No legal factuality or reference
summary evaluation was performed. Small output-length differences remain a
confound in the generation timings.

The public `experiments/legal/aggregate-results.json` includes all individual
measurements, scheduling plans and output hashes, but omits generated prose and
absolute school filesystem paths. `gpu-environment.txt` pins the installed
packages. Full predictions remain in ignored local raw results.

## Next experiment

Sweep token budgets (for example 16K, 24K, 32K and 48K slots) and sequence caps
on a larger held-out legal corpus, comparing the complete measured speed/memory
frontier with fixed batching. Use separate development and evaluation documents
when selecting a policy. Add human-reviewed checks for parties, obligations,
exceptions, dates, termination and governing law, with source evidence spans,
before calling an optimization quality-preserving. Only then extend to longer
contracts or judgments, online arrival traces, and continuous-batching backends.
This PoC establishes a reproducible baseline; it does not establish novelty over
existing inference engines or a general legal-domain advantage.
