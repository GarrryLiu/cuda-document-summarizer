# Long legal-document inference profile

The short NDA replay study could not establish where long-document summarization
spends its time. This study profiles the native Qwen SDPA pipeline on intact CUAD
contracts with a contract-specific summary prompt. It does not replace kernels.

## Inputs and reproduction

Source: [CUAD, The Atticus Project](https://github.com/The-Atticus-Project/cuad),
Hendrycks, Burns, Chen and Ball, *CUAD: An Expert-Annotated NLP Dataset for Legal
Contract Review*, NeurIPS 2021. Use the official `data.zip` and `test.json` split.
The preparation script verifies the archive SHA256 before conversion; raw
contracts stay in ignored local data. Titles and text hashes identify samples.

```bash
curl -fL https://raw.githubusercontent.com/The-Atticus-Project/cuad/main/data.zip \
  -o data/local/cuad-data.zip
python experiments/kernels/prepare_cuad.py \
  data/local/cuad-data.zip data/local/cuad-test.jsonl
sbatch scripts/brown-legal-profile.sbatch
```

The model/tokenizer revision is pinned. Choose one distinct complete contract
nearest each target of 4096/8192/16384/24576 chat-template tokens, within 20% of
the target and within the 32768-token model limit including decode. No text is
truncated, repeated to manufacture length, or concatenated across contracts.
Batch 4 repeats the same contract to keep this profile free of padding effects;
it is not a diverse-document serving benchmark.

## Measurement

For each contract and batch size, warm up prefill and 32 decode forward passes.
Measure three subsequent runs with CUDA synchronization around each phase,
separately recording peak PyTorch allocation. Inputs already reside on the GPU;
tokenization, model loading, output CPU serialization and profiler overhead are
excluded from those latency samples. This is not user-facing request latency.

Collect prefill and decode CPU/CUDA profiles in a separate run. Aggregate only
CUDA-device events for GPU activity totals. Classify kernel names into attention,
matrix, copy/cache, reduction/norm, elementwise and other, retaining every raw
name so the heuristic classification can be audited. ATen attribution is kept
separately and must never be added to CUDA activity totals. The GPU activity sum
is neither elapsed wall time nor a hardware occupancy/bandwidth measurement.

Decode always executes 32 additional forward passes, including after EOS. Token
hashes cover that continuation; the first token from prefill is not part of the
hash. This is a bounded performance profile, not complete legal summaries or a
factuality evaluation. The contract-specific prompt also differs from the prior
technical-summary prompt, so timings should not be presented as a controlled
improvement over earlier experiments.

## Follow-up: cache allocation ablation

`cache_probe.py` compares the fixed Transformers release's native `DynamicCache`
and `StaticCache` on the selected ~16K/~24K contracts at batch 1/4. This is an
experiment with existing cache implementations, not a newly authored CUDA
kernel. It follows the observed `aten::cat` cost; `cat` can also include other
concatenations, so its full attributed time is not assumed to be KV-cache work.

Both policies receive the same explicit GPU cache positions and execute the
same 32 decode forward passes. Prefill includes initial cache allocation. Each
policy is warmed up, then timed in three randomized paired rounds. Results
include prefill/decode/combined latency, peak allocated memory, first-step logit
differences and continuation token identity. The largest batch also receives a
separate decode profile for each policy, allowing changes in concatenation cost
and attention backend to be checked. StaticCache can change masks/backend work,
so removing one copy operation does not guarantee a net improvement.

```bash
sbatch --export=ALL,PROFILE_RESULT=/absolute/path/to/legal-profile.json \
  scripts/brown-cache-probe.sbatch
```

The contracts are reused from bottleneck discovery; this is a mechanism probe,
not held-out validation. The production cache policy remains unchanged until
there is a repeatable end-to-end benefit and a quality check on complete outputs.

## Results: RTX A6000

Slurm profile job 138167. Times are medians of three unprofiled runs.
Percentages are CUDA activity shares from a separate profile, not wall-time shares.

| Actual prompt tokens | Batch | Prefill s | 32-step decode s | Prefill attention | Decode attention | Decode copy/cache |
|---:|---:|---:|---:|---:|---:|---:|
| 4112 | 1 | 0.304 | 0.724 | 8.7% | 4.5% | 7.1% |
| 4112 | 4 | 1.141 | 0.913 | 8.7% | 9.0% | 14.7% |
| 8190 | 1 | 0.629 | 0.909 | 15.4% | 8.2% | 9.8% |
| 8190 | 4 | 2.484 | 0.922 | 15.3% | 13.2% | 22.3% |
| 16621 | 1 | 1.465 | 0.914 | 26.8% | 10.8% | 14.9% |
| 16621 | 4 | 6.048 | 0.916 | 26.9% | 17.6% | 32.5% |
| 22899 | 1 | 2.292 | 0.909 | 32.9% | 13.0% | 18.1% |
| 22899 | 4 | 9.474 | 0.930 | 35.4% | 19.8% | 37.5% |

At 22,899 tokens / batch 4, prefill is 9.474 s and 32-step decode is 0.930 s.
Thus prefill dominates this bounded workload. Longer complete summaries may
shift the balance, so this is not a general statement about every serving task.

Prefill CUDA activity is 47.1% matrix operations, 35.4% attention and 13.1%
elementwise work. Decode copy/cache rises to 37.5%, while matrix operations
account for 36.1% and attention 19.8%. `aten::cat` is attributed 326.6 ms over
the decode profile. That provides a concrete reason to test preallocation,
while the prefill-dominated wall time limits its possible total impact.

### Next custom-kernel candidate

Prioritize a fused **RMSNorm** experiment for long-contract prefill. Qwen has
73 normalization calls per forward (two per layer plus final norm). The profile
shows 73 `aten::pow` and 73 `aten::mean` calls; their prefill GPU attribution is
160.3 ms and 77.7 ms at B4/22,899 tokens. Cast/copy and multiply operations add
possible fusion opportunities, but their totals also include other model work
and cannot all be credited to RMSNorm. This is a candidate, not a claimed gain.

The experiment should use the pinned model's actual RMSNorm input rows and
weights (FP16, hidden width 2048; row count = batch times tokens), match its
FP32 accumulation and FP16 rounding order, compare against
both eager PyTorch and a compiled/fused baseline, then test prefill wall time
and output quality. Include short and long workloads (batch times tokens),
non-default-stream tests, numerical edge cases and fallback for unsupported
shapes. Keep SDPA intact during this experiment to isolate the change.

A fused SiLU times gate kernel is the secondary candidate: 36 SiLU calls
contribute 213.4 ms in the largest prefill profile. Its associated multiply time
still needs isolated attribution; the model-wide `aten::mul` total is not that cost.

Implementation references: [pinned Qwen2 RMSNorm/MLP source](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen2/modeling_qwen2.py) and [pinned cache implementation](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/cache_utils.py).

## Cache probe results

All four conditions completed with three paired runs each. All twelve token
continuations were identical, but first-step logits differed (maximum absolute
differences 0.078125–0.13671875); this does not establish quality parity for
complete summaries. Static allocation was included in prefill.

| Prompt tokens | Batch | Dynamic total s | Static total s | Static / dynamic | Dynamic peak GiB | Static peak GiB |
|---:|---:|---:|---:|---:|---:|---:|
| 16621 | 1 | 2.196 | 4.919 | 2.24x | 7.72 | 8.11 |
| 16621 | 4 | 6.994 | 15.826 | 2.26x | 13.26 | 14.06 |
| 22899 | 1 | 3.088 | 7.818 | 2.53x | 8.42 | 9.64 |
| 22899 | 4 | 10.642 | 26.090 | 2.45x | 16.06 | 19.44 |

Native StaticCache takes 2.24–2.53 times the total wall time of DynamicCache in
this pinned setup. It is therefore not enabled in the product. This result
concerns this eager model/backend configuration, not every static-cache design.

At 22,899 tokens / batch 4, the separate decode profile explains the regression:

- Dynamic: Flash SDPA, 176.85 ms attention attribution and 326.80 ms `aten::cat`.
- Static: efficient-attention backend, 1,451.33 ms backend attribution, plus
  2,601.78 ms `aten::clone` with 2,304 calls. These are 2 copies per layer per step.
- Static decode wall time is 4.520 s versus 0.930 s for dynamic; total model
  time is 26.090 s versus 10.642 s. Peak allocation also increases.

The pinned [SDPA integration source](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/integrations/sdpa_attention.py)
uses native GQA only when the CUDA attention mask is absent. With a mask it
calls `repeat_kv`, whose expand/reshape materializes duplicated KV heads.
Together with the observed clone counts and backend switch, this supports the
explanation that avoiding cache concatenation introduced more expensive copies
and attention execution. It is not evidence that preallocation itself is slow.
CPU parent attribution is nested: reshape/clone/copy totals must not be summed.

If cache work is resumed, test a capacity-backed cache that exposes only its
valid prefix and preserves the native GQA/Flash path, with correctness tests for
causality and cache positions. That is a separate implementation, not a feature
completed by this ablation. For the present prefill-dominated workload, fused
RMSNorm remains the next custom-kernel candidate.

Slurm cache job: 138287. Raw profiles, timing rounds, output hashes and environment
versions are in [`experiments/kernels/legal-profile`](../experiments/kernels/legal-profile).
The CPU test suite passed all 9 tests. Both GPU jobs completed successfully.
