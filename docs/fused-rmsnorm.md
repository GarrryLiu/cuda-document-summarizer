# Fused RMSNorm for long-contract prefill

The legal-contract profile showed repeated conversion, square, mean and scale
operations in Qwen normalization. This experiment fuses that chain in C++/CUDA
and tests whether it reduces prefill time for intact contracts, alongside a
compiler-generated fusion baseline. It does not change attention or KV policy.

## Operator

`src/cuda_summary/kernels/rmsnorm.cu` implements contiguous CUDA FP16 inputs of
shape `[rows,2048]` or `[batch,tokens,2048]`, with FP16 weights. One block handles
one row; 256 threads retain eight values each in registers, use warp/shared
reductions for FP32 variance, and write only the final normalized output.
No intermediate full-size FP32 tensors are materialized by the custom operator.

The pinned Qwen implementation normalizes in FP32, casts the normalized values
to FP16, and only then multiplies by FP16 weights. The kernel explicitly retains
that intermediate rounding. Reduction order and reciprocal square root can
still introduce small differences; numerical tolerance is not token identity.
FMA contraction is disabled. Epsilon must be finite and in `(0,1]`.

The extension validates dtype, device, dimensions, contiguity and inference-only
usage; empty rows return an empty tensor without launching. A device guard and
the current CUDA stream are used. Scalar half loads accept odd-half storage
offsets. There is no backward implementation and no default model patch.

## Validation protocol

- Sixty comparisons against the pinned Qwen expression: six shapes, five input
  patterns (random/zero/tiny/large/mixed), two epsilons. Maximum finite FP16 values
  exercise FP32 variance. Tolerance: absolute 0.008, relative 0.002; report actual
  errors rather than claiming bitwise equality.
- Invalid argument/gradient cases, non-default CUDA stream, empty inputs and
  contiguous unaligned storage.
- Actual prefill inputs at norm indices 0, 36 and 72 (of 73 norms), on intact CUAD
  contracts near 8K and 24K tokens, batch 1/4. Compare eager, torch.compile default
  Inductor with dynamic rows/full graph, and hand-written CUDA.
- Microbenchmark: 20 calls per CUDA Graph, seven randomized rounds, warm-cache
  fixed inputs. Compilation/capture/warmup excluded. All three outputs checked.
- Model experiment: replace all eligible **prefill** RMSNorm calls only. Decode
  uses the original modules. Three paired rounds with randomized method order,
  warmup each method; record synchronized prefill/decode/total time, peak allocated
  memory, first-logit differences and output token hashes. Hooks are restored
  after each experimental call.

The 32 additional decode forward passes run even after EOS and do not constitute
complete summaries. Batch4 repeats one contract, and inputs overlap the earlier
bottleneck study; no held-out quality/generalization claim is made. GPU activity
microbenchmarks are not end-to-end speedups. CPU tokenization, model loading,
compilation and output CPU serialization are outside timed model regions.

## Reproduction

This study builds on the profile harness in PR #5. Prepare ignored CUAD inputs
with `prepare_cuad.py`, then submit:

```bash
sbatch scripts/brown-rmsnorm.sbatch
```

The Slurm launcher uses A6000, pinned PyTorch/Transformers/CUDA versions, and
node-local temporary model/compiler caches (including Triton/Inductor). The
source and result JSONs retain hashes. Original pipeline defaults remain intact.

## Measured results: RTX A6000

GPU job 138338. All 60 numerical tests, 13 invalid API cases, non-default-stream,
empty and unaligned-storage checks passed. Maximum adversarial-test absolute
error was 0.0078125. All 12 actual-input comparisons passed the stated combined
absolute/relative tolerance; actual model norms reached max absolute errors
0.125 for both CUDA and compiled, including large final-norm activations.

### Actual-input operator timings

Each entry is the median of the three sampled norm modules
(indices 0/36/72), each measured in seven CUDA Graph rounds.

| Tokens | Batch | Eager µs | Compiled µs | CUDA µs | Eager / CUDA |
|---:|---:|---:|---:|---:|---:|
| 8190 | 1 | 895.80 | 104.50 | 100.30 | 8.93x |
| 8190 | 4 | 3529.42 | 414.62 | 395.11 | 8.93x |
| 22899 | 1 | 2471.42 | 290.25 | 276.63 | 8.93x |
| 22899 | 4 | 9849.70 | 1154.66 | 1101.57 | 8.94x |

Across all twelve captures, CUDA reduces operator time by 3.77–4.76% relative
to the compiled baseline, and is 8.91–8.95 times as fast as eager. This is an
operator result, not a 9x model speedup.

### Model timings

Medians of three randomized paired rounds; seconds. Total is median of each
run's prefill plus decode, not necessarily the sum of the separate medians.

| Tokens | Batch | Eager prefill | Compiled prefill | CUDA prefill | CUDA prefill reduction vs eager | Eager total | Compiled total | CUDA total | CUDA total reduction vs eager |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8190 | 1 | 0.6271 | 0.5757 | 0.5753 | 8.27% | 1.3481 | 1.3009 | 1.3055 | 3.16% |
| 8190 | 4 | 2.5019 | 2.3029 | 2.2988 | 8.12% | 3.2404 | 3.0372 | 3.0320 | 6.43% |
| 22899 | 1 | 2.3153 | 2.1773 | 2.1701 | 6.27% | 3.0439 | 2.9114 | 2.9011 | 4.69% |
| 22899 | 4 | 9.6853 | 9.1602 | 9.1312 | 5.72% | 10.6137 | 10.0900 | 10.0626 | 5.19% |

The custom kernel lowers prefill time by 5.72–8.27% and bounded total time by
3.16–6.43% versus eager in this run. Compiled fusion delivers most of that gain;
the custom kernel's extra prefill gain over compiled is only 0.07–0.33%, and
total time is not consistently better than compiled. Do not claim a meaningful
end-to-end advantage over the compiler from these three-round measurements.

All twelve CUDA-versus-eager token continuations match, as do all twelve
compiled-versus-eager continuations. Logits are not identical (CUDA max absolute
first-logit discrepancies 0.0586–0.0801 across conditions). This is limited to
32 decode steps and does not establish full legal-summary factuality or general
quality parity. Peak model allocation is effectively unchanged: other model
operations still determine the high-water mark.

Replacement counters verify 292 prefill norm calls (73 norms times warmup plus
three trials) and 9,344 native decode fallbacks per method/condition. Attention,
cache policy and decode norms were not replaced. Only this A6000, two intact
contracts and the tested batch sizes are covered; other shapes/GPUs need their
own validation. No Nsight hardware bandwidth or occupancy counters are claimed.

The implementation and experimental integration are complete. Production defaults
remain native; the kernel can be invoked explicitly through
`cuda_summary.rmsnorm.fused_rmsnorm`. Before broad enablement, test held-out
contracts with complete summaries and retain the compiled baseline in comparisons.

Raw rounds, output hashes, source hashes, GPU tests and environment versions:
[`experiments/kernels/rmsnorm-results`](../experiments/kernels/rmsnorm-results).
