# Initial local experiment

Date: 2026-10-27. Host: Apple Silicon, macOS; Python 3.14.7.

9 unit tests passed. Coverage includes stable length sorting, tail batches,
no missing/duplicated indices, padding accounting, input validation, and randomized
comparisons over 59 workload sizes and three batch sizes. These checks do not
validate model generation or CUDA behavior.

`PYTHONPATH=src python3 experiments/padding.py` uses seed 42 and 400 uniformly
sampled integer lengths between 1,024 and 24,576. Raw lengths and results are in
`experiments/padding-results.json`. Padding allocation is
`sum(batch_size * max(input_length_in_batch))` over all batches. It measures only
input padding positions, excluding decode length, attention complexity and cache
memory. Sorting uses an entire offline workload, so online waiting latency is
not modeled.

| Batch size | Padded-position reduction vs arrival |
|---|---:|
| 1 | 0.00% |
| 4 | 34.60% |
| 8 | 41.48% |
| 16 | 43.48% |

These savings are specific to the sampled length distribution. They do not
establish a 25% throughput improvement or any measured speedup.

## CUDA execution status

The actual benchmark command was attempted on the local Mac and exited with
status 2: GPU dependencies unavailable. No model was downloaded and no GPU results
were written. CUDA availability is also checked after dependencies load.

## Next real experiment

1. Collect a licensed corpus of 400 technical reports with unique IDs; record
   source URLs/licenses and hashes. Measure lengths with the actual chat template.
2. Run a short smoke test on CUDA and inspect all four conditions for valid
   summaries, identical document coverage and context compliance.
3. Run batch sizes 1, 4 and 8 on the same corpus, fixed model revision and output
   cap. Select sizes that fit the chosen GPU. An OOM currently fails the run;
   automatic batch splitting is not implemented because it would change fairness.
4. Repeat at least three times; alternate/randomize condition order externally
   for longer studies to reduce thermal/order bias (current runner uses fixed
   order). Analyze uncertainty, output lengths and input length buckets.
5. Compare arrival/cache-on vs length/cache-on to isolate batching; compare
   cache-off vs cache-on within each strategy to isolate KV cache.
6. Manually review the same 100 document IDs across conditions using the rubric.
   Report factuality and coverage alongside throughput; do not report a quality
   improvement from scheduling alone.

## Subsequent real GPU execution

The initial local status above records the Mac attempt. Real execution has since
completed on Brown's NVIDIA L40: both a short-document smoke test and an
eight-document mixed-length synthetic GPU experiment. See
[the Brown report](brown-gpu.md) for measurements, actual token lengths, model
revision, environment, raw results and quality caveats. The cache-on batching
comparison measured 4.95% higher median throughput for length ordering on that
small fixture; it does not reproduce a 25% general improvement.
