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


## Follow-up budget sweep

The sweep requests budgets of 16,384, 24,576, 32,768 and 49,152 padded
input-plus-output token slots, with a maximum of eight sequences. It keeps
arrival-4, length-4 and length-8 controls. Identical ordered batch plans are
measured once and explicitly aliased: on these documents, 49,152 slots yields
exactly length-8. Six distinct policies each run three times in seeded shuffled
order. Alias rows must not be counted as independent repetitions.

The A6000 was fully occupied at follow-up submission, so the pending job 78668
was canceled and the entire matched comparison ran on an L40S (job 78683).
Follow-up measurements are separate from the original A6000 result. Model,
prompt, revision, FP16, output reserve and input documents are unchanged. Code
hashes and environment versions accompany the measurements. Timing covers
loading input tensors and generation, not model download/load or planning.

The sweep reuses the same 24 development NDAs to explore configuration effects;
it does not provide held-out validation. The observed frontier compares median
completion time and maximum allocated memory, without statistical significance
claims. A smaller budget is not a hard memory limit. This GPU has enough physical
memory for all tested policies; improvements do not demonstrate OOM avoidance.
Batch-shape output differences remain a timing confound.

Quality diagnostics and source-inspection limitations are described in
[legal-quality-protocol.md](legal-quality-protocol.md). Automated lexical flags
are not factuality scores; policy hash matches do not validate legal content.

### Follow-up measurements (L40S)

Job 78683 completed in 13m33s with exit code 0. All 18 measured runs processed
all 24 documents without OOM or generation-cap hits. Exact source-code and input
hashes were checked after retrieval. Each distinct policy produced identical
output token hashes across its three repetitions.

| Policy | Median time (s) | Throughput vs arrival | Peak allocated GiB | Exact matches vs arrival |
| --- | ---: | ---: | ---: | ---: |
| Arrival 4 | 47.80 | 1.000× | 8.05 | 24/24 |
| Length 4 | 40.54 | 1.179× | 8.05 | 17/24 |
| Length 8 / 48K alias | 34.47 | 1.387× | 10.24 | 17/24 |
| 16K budget | 38.28 | 1.249× | 7.51 | 18/24 |
| 24K budget | 35.46 | 1.348× | 8.22 | 17/24 |
| 32K budget | 36.37 | 1.314× | 8.26 | 19/24 |

The observed speed/memory frontier consists of 16K, 24K, and length-8 (48K is
its alias). On this development workload, 16K is faster and uses less allocated
memory than fixed length-4. The 24K condition is faster and uses slightly less
allocated memory than 32K. These are observed median comparisons, not claims of
statistical significance. The median timing advantage of 24K over 32K is small
and needs confirmation on independent workloads.

24K increased observed throughput by 34.8% over arrival-4. Relative to fixed
length-8, it took 2.9% longer and used 19.8% less peak allocated memory. Its
absolute peak is 8.22 GiB of PyTorch allocation, not a guarantee that an 8 GB
GPU can run it. A measured candidate for further testing is 24K for a balance,
16K for lower allocation, or length-8 for speed; no production default is changed.

`experiments/legal/sweep-results.json` contains individual measurements and
alias metadata; `sweep-environment.txt` records the installed packages. Do not
compare these times directly to the A6000 study to infer a scheduling gain.

### Quality findings and next step

Each distinct sweep policy has two numerically flagged documents and four
summaries over the requested 160-word limit. One numeric flag is the expanded
source date year (a lexical review flag); the other is the unsupported one-year
restriction in document 381. Codex rechecked that same unsupported claim in all
six sweep policies against the source redaction, with offsets and hashes saved
in `sweep-source-audit.json`. This narrow source inspection is not a human/legal
expert evaluation, a full-summary audit, or an error-rate estimate.

24K matches arrival-4 exactly on 17/24 outputs; seven documents have changed
lines containing legal markers and require semantic review. Numeric flags alone
do not cover negation, party identity, jurisdiction or material omissions. Thus
no tested policy is established to preserve legal-summary quality.

Next, use a disjoint evaluation set and a fixed atomic-claim rubric. Test an
abstention/evidence-linked prompt for missing or redacted durations, with human
source review, before choosing a quality-constrained scheduling configuration.
Keep prompt changes separate from the scheduling ablation: changing both at once
would confound the throughput comparison.
