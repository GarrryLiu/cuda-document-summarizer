# CUDA decode-attention kernel experiment

This branch adds actual CUDA C++ code in
`src/cuda_summary/kernels/decode_attention.cu`, independently of the legal
scheduling branch. It does not replace Transformers attention in production.

## Scope and math

Inference-only contiguous FP16 Q `[B,Hq,128]`, K/V `[B,Hkv,S,128]`, with
`Hq % Hkv == 0`. Each query attends to every supplied cached key. The query is
the last token: this is not top-left causal attention with a one-row mask.
There is no padding mask, variable-length batch, dropout, backward, paged KV,
rotary embedding or KV append operation. Inputs must be finite. Caller must
supply only valid keys. Bounds: B 1–32, Hq 1–128, S 1–131072, splits 1–128.

A CTA processes one query head and one contiguous sequence partition. Eight
warps cooperate on QK dot products for 64-key tiles. V loads are coalesced across
128 dimensions; Q and tile weights live in shared memory. Online max/denominator
updates and FP32 accumulators avoid materializing a full attention matrix.
Single-partition mode is a fused educational baseline. Multi-partition mode
produces FP32 partial outputs and log-sum-exp values, then merges with stable
LSE weights in a second CUDA kernel. Empty tail partitions receive zero weight.
No atomics, approximate fast-math flag or tensor-core MMA is used.

This is an initial implementation of existing tiled/online-softmax and split-KV
ideas, not an algorithmic novelty claim or an optimized FlashAttention replacement.
The first variant rereads KV per query head. The optimized grouped variant
stages one 32-key K/V tile for four query heads, using 16-byte vector loads
and warp-local softmax reductions. It requires a GQA ratio divisible by four
and 16-byte-aligned K/V storage; unsupported inputs are rejected.
Both variants remain SIMT implementations without tensor-core MMA.

## Build and correctness

Install the project GPU dependencies, `ninja`, and a compatible CUDA compiler.
The Python wrapper lazily invokes PyTorch's C++ extension loader. Merely
importing the package does not compile or require a CUDA device. The `.cu` file
is included as package data. Compiler output is kept in the configured
`TORCH_EXTENSIONS_DIR`; the script uses private job-local storage.

```sh
PYTHONPATH=src python experiments/kernels/check.py /tmp/correctness.json
PYTHONPATH=src python experiments/kernels/benchmark.py --output /tmp/benchmark.json
PYTHONPATH=src python experiments/kernels/profile_model.py \
  --input data/local/legal-poc.jsonl --output /tmp/model-profile.json
```

Correctness compares identical FP16 input values against FP32 SDPA math, using
absolute tolerance .003 and relative tolerance .02. Cases cover MHA/GQA,
non-divisible sequence lengths, splits exceeding sequence length, zero queries,
large logits, distinct KV-head values, a nondefault CUDA stream and rejected API
inputs. The microbenchmark also checks captured outputs against default SDPA.
These checks do not replace memory/race checking with Compute Sanitizer.

## Measurements

Microbenchmarks use Qwen2.5-3B's Hq=16, Hkv=2, D=128, batch 1/4, sequence
512/2048/8192/16384/32768, and split counts 1/4/16/32. Grouped-KV variants use 16/32/64 splits. Baselines are default
PyTorch SDPA and explicitly forced FlashAttention; unavailable forced backends
are reported, not silently substituted. CUDA Graph replay excludes Python launch
and allocation overhead, capturing 100 operation calls inside one graph replay per sample, with five
shuffled rounds. Capturing the whole loop avoids measuring Python replay
cadence for short kernels.
Python API wall time is separately timed with device synchronization. Compilation,
graph capture and warmup are excluded. Inputs are repeatedly reused: these are
warm-cache results, not cold-HBM bandwidth measurements. No fastest split policy
is automatically selected from the measurements.

The Qwen model profile uses the existing summarizer prompt on one full legal
NDA, with a separate prefill and 32 cached decode steps. It retains the original
Transformers attention implementation. Operator device time is profiling evidence,
not end-to-end wall time, hardware utilization counters or a measured application
speedup for the custom kernel. Prefill is outside this custom kernel's scope.

## References

- [Qwen2.5-3B-Instruct model configuration](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [PyTorch SDPA and GQA semantics](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention)
- [PyTorch C++ extension loader](https://docs.pytorch.org/docs/stable/cpp_extension.html)
- [CUDA stream interop](https://docs.pytorch.org/cppdocs/api/cuda/streams.html)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [Flash-Decoding split-sequence attention](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)

## Execution provenance

Job 85685 compiled the initial per-query-head implementation, passed 32 numerical
cases and profiled the unmodified Qwen pipeline. The initial microbenchmark
replayed a one-operation graph in a Python loop; its timings are retained as
historical diagnostics, not the final comparison.

Job 85688 tested grouped-KV reuse. Job 85689 validates the final committed
alignment-safe implementation and reruns all original/custom/SDPA controls using
the 100-operation graph protocol. All measurements are on L40S, but results from
different jobs/protocols are not pooled. The final benchmark stores hashes of its
CUDA source and Python benchmark script. The original model profile uses the
pinned model revision and records its actual attention backend and head layout.

The cluster does not expose `ncu`, `nsys` or `compute-sanitizer` on PATH in this
setup. PyTorch/CUPTI profiling, numerical/API checks and CUDA launch error checks
were used. Hardware counters, racecheck and memcheck are not claimed.

## Final measured results (2026-10-29)

The final committed kernel compiled with nvcc 13.0.88 and PyTorch 2.14.1+cu130
on an L40S (142 SMs). Job 85689 completed successfully (1m37s, exit code 0).
60 numerical cases passed (maximum absolute error 0.0008564), with nine rejected
invalid API cases and both variants tested on a nondefault stream. Benchmarks
passed captured-output checks for all ten shapes. Source and benchmark script
hashes match the committed files. The nine existing CPU tests also pass.

The table compares the SAME split count (32) to isolate the grouped implementation
change; grouping also changes tile size, block size, reduction placement and load
width, so it is not an isolated attribution to KV traffic alone. Times are median
microseconds per operation from five 100-operation CUDA graph samples.

| Batch | Context | Per-head split32 | Grouped split32 | Default SDPA | Split32 / grouped32 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 512 | 8.09 | 8.07 | 7.12 | 1.00× |
| 1 | 2048 | 10.22 | 13.74 | 8.88 | 0.74× |
| 1 | 8192 | 26.92 | 42.56 | 16.12 | 0.63× |
| 1 | 16384 | 49.00 | 80.86 | 29.37 | 0.61× |
| 1 | 32768 | 95.57 | 156.87 | 34.24 | 0.61× |
| 4 | 512 | 18.90 | 10.14 | 7.23 | 1.86× |
| 4 | 2048 | 25.46 | 19.76 | 9.74 | 1.29× |
| 4 | 8192 | 82.25 | 63.15 | 21.45 | 1.30× |
| 4 | 16384 | 157.88 | 120.82 | 31.54 | 1.31× |
| 4 | 32768 | 417.06 | 252.71 | 188.34 | 1.65× |

At batch 4/context 32768, grouped-32 reduces custom split-32 time from 417.06 to
252.71 us (39.4% less time, 1.65× throughput). Default SDPA remains faster at
188.34 us. At batch 1, grouped-32 is often slower than per-head split-32. Grouping
reduces the block count fourfold; for B=1,Hq=16,splits=32 this is only 128 CTAs
on a 142-SM GPU. Reduced parallelism is one plausible contributor, not a confirmed
hardware-counter diagnosis. Grouping and split count must be tuned jointly.

The best measured custom variant beats default SDPA only for B=1,S=512
(split16: 6.23 versus 7.12 us, about 1.14×). This is post-hoc selection from
several variants on the same GPU, not a validated general dispatch policy. In
the other nine shapes, even the post-hoc best custom variant is slower than
SDPA. The forced FlashAttention control was available for all shapes. No custom
kernel is enabled automatically in summarization.

### Actual model profile

The original summarizer prompt on one full NDA gives 3431 tokens. Prefill and
32 decode steps were profiled separately with Transformers SDPA. Counting only
CUDA device activities avoids counting both CPU operator attribution and its
GPU kernel a second time. Flash attention accounts for 8.62% of recorded
prefill device activity and 3.71% of decode activity; matrix kernels account for
67.34% and 79.09% respectively. These are single-workload profiler fractions,
not wall-time fractions or a universal bottleneck claim. They do not support a
large generation speedup by replacing attention at this context length.

### Next kernel experiment

For a kernel research project, investigate the GQA reuse/parallelism tradeoff:
query heads per CTA × split count × sequence length, then examine registers,
occupancy and memory traffic with hardware counters when available. Keep SDPA
as a strong baseline and preserve negative results. For overall application
acceleration on the profiled short-context workload, matrix-vector operations
and launch/elementwise fusion are more relevant candidates to profile next.
Long-context scaling must be profiled separately before assuming the same split.
