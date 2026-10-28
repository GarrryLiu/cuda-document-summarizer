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
