# GQA sharing versus sequence parallelism

Research question: for Qwen document decoding, when does sharing a staged KV tile
across more query heads compensate for fewer independent thread blocks?

## Implementation

`decode_attention(..., group_size=1|2|4|8)` selects template specializations of the
same shared-KV kernel. All use a 32-token tile, 16-byte global loads, one warp per
query head and FP32 online softmax/accumulation. Splits expose sequence parallelism.
The existing `grouped=True` API still selects group size 4. The original per-head
kernel remains available as an independent implementation.

Thread count necessarily varies here (32 × group size), as do block count and
shared weights. This is a study of the resulting resource tradeoff, not an
isolated proof that KV bandwidth causes any measured speedup. No Nsight occupancy
or memory-traffic counters were collected.

## Protocol

1. GPU correctness checks compare identical FP16 inputs to FP32 SDPA math,
   including partial tiles, empty split tails, large logits and distinct KV heads.
2. Synthetic tuning: batch 1/4, context 512/2048/8192/16384/32768,
   group 1/2/4/8 × splits 1/4/16/32/64, plus native SDPA.
3. Actual-input validation: shortest/median/longest of the local ContractNLI NDA
   sample; capture the first decode token at layers 1/18/36 of pinned Qwen2.5-3B.
   Compare the synthetic-selected configuration with the actual Transformers SDPA
   call, both before and after paying contiguous-layout conversion cost.
4. Independent bounded generation probe: two other documents, batch 1/4, three
   randomized paired runs with up to 32 output tokens. Experimental interception
   replaces eligible decode only; prefill and unsupported semantics use SDPA.
   Record synchronized wall time, peak allocated memory and output token hashes.

Synthetic tuning selects on eager latency, not validation outcomes. The replay
validation gate requires all seven eager samples including layout conversion to
beat every SDPA sample by at least 10%. Passing this small gate is not sufficient
for a production routing policy. The generation probe intentionally tests the
candidate even if the gate fails, so a negative end-to-end result can be measured.
Production routing remains unchanged.

GPU timing uses 50 operations inside each CUDA Graph and seven randomized rounds;
eager timing uses ten calls per round. Compilation, graph capture and warmup are
excluded. Reused inputs mean warm-cache conditions. Batch 4 replicates a single
real document to avoid silently ignoring padding masks; it is not a diverse
four-document serving benchmark. Actual document lengths are reported separately
from synthetic long-context shapes. No document text or raw Q/K/V is published.

The generation probe uses the original technical-summary prompt on legal NDAs.
Its 32-token cap makes it a bounded inference probe, not an evaluation of complete
legal summaries or factual accuracy. Identical hashes establish token identity;
different hashes alone do not establish better or worse summary quality.

## Reproduction

From a staged checkout on Brown:

```bash
sbatch --constraint=rtx_a6000 scripts/brown-group-sweep.sbatch
# After successful sweep:
sbatch --export=ALL,SWEEP_RESULT=/absolute/path/to/group-sweep.json \
  scripts/brown-generation-probe.sbatch
```

Keep input at ignored `data/local/legal-poc.jsonl`. Model and tool caches live in
the job's temporary node-local directory and are removed on exit. The sweep
launcher defaults to L40S; the recorded run overrides it to A6000 because the
L40S node was in maintenance. Compare methods within this run only.

## Measured results: RTX A6000, 2026-11-04

All 144 synthetic/adversarial numerical cases passed. All 18 actual attention
replays passed FP16 tolerances (maximum absolute discrepancy vs native SDPA: 0.00390625).
Two of 18 actual replays passed the conservative eager stability gate: B1/S514,
layers 18 and 36. Their CUDA Graph GPU times still regressed. This does not
validate all layers or subsequent decode lengths for a production policy.

The following are minimum median GPU times across splits for each group. These
are descriptive best-of-sweep results, not independently validated speedups.

| Batch | Tokens | G1 µs | G2 µs | G4 µs | G8 µs | SDPA µs |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 11.69 | 10.75 | 10.38 | 10.79 | 9.46 |
| 1 | 2048 | 29.55 | 19.23 | 16.90 | 16.26 | 10.77 |
| 1 | 8192 | 104.49 | 64.90 | 46.77 | 46.57 | 28.57 |
| 1 | 16384 | 200.81 | 123.06 | 84.13 | 84.05 | 40.57 |
| 1 | 32768 | 391.94 | 241.66 | 159.97 | 159.78 | 75.84 |
| 4 | 512 | 27.18 | 18.53 | 14.23 | 13.52 | 8.99 |
| 4 | 2048 | 92.69 | 55.31 | 43.32 | 43.87 | 21.61 |
| 4 | 8192 | 335.42 | 172.44 | 133.84 | 141.78 | 60.81 |
| 4 | 16384 | 661.83 | 329.67 | 255.32 | 270.09 | 109.12 |
| 4 | 32768 | 1310.39 | 643.81 | 509.69 | 532.78 | 203.86 |

At B4/S32768, the best G4 GPU time is 509.69 µs versus 1310.39 µs
for the best G1 configuration: 61.1% less GPU time within this shared-tile kernel
family. Native SDPA takes 203.86 µs, so G4 is still 2.50× slower. This is not an
end-to-end speedup. Group size 8 offers no consistent advantage over 4.

Actual Qwen inputs have contexts of 514, 2210 and 4683 tokens. Short-context
eager medians can improve while CUDA Graph GPU times worsen, illustrating why
Python/API overhead must be separated from kernel execution. No production
dispatch rule is justified by these results.

### Bounded end-to-end generation

Each row has three paired timed runs after warmup; values are medians. All
outputs hit the 32-token cap. These are full model.generate calls, including
prefill and decode, but not tokenization or model loading.

| Document | Batch | Prompt tokens | SDPA s | Custom s | Custom time change | Identical pairs |
|---|---:|---:|---:|---:|---:|---:|
| 152 | 1 | 1401 | 0.8509 | 0.8541 | +0.38% | 3/3 |
| 152 | 4 | 1401 | 1.1422 | 1.1440 | +0.16% | 3/3 |
| 32 | 1 | 2889 | 0.9629 | 0.9607 | -0.23% | 0/3 |
| 32 | 4 | 2889 | 1.6257 | 1.6021 | -1.45% | 3/3 |

Differences range from 0.38% slower to 1.45% faster, without enough repeats or
independent jobs to establish a reliable improvement. Three of twelve pairs
changed tokens (all document 32, batch 1). No factuality/quality review was
performed, so numerical operator tolerance is not a guarantee of summary parity.
The default production implementation remains SDPA.

The second job also passed all 144 numerical cases, 12 invalid-API cases and
non-default-stream checks for every group size. Local CPU tests: 9 passed.
Slurm sweep job: 137840; generation job: 137920. Results are in
[`experiments/kernels/gqa-sweep`](../experiments/kernels/gqa-sweep).

### Next decision

Keep head sharing as a measured building block. Before extending dispatch,
profile a real longer-document workload and locate the actual bottleneck. If
attention remains worth optimizing, focus on reducing execution cost relative
to SDPA, rather than extending a group-size sweep that already shows diminishing
returns. A useful follow-up should measure instruction/memory behavior and
complete summaries on held-out documents with source-grounded quality review.
