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
It rereads shared KV for each query head and does not yet exploit GQA KV reuse
across those heads.

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
512/2048/8192/16384/32768, and split counts 1/4/16/32. Baselines are default
PyTorch SDPA and explicitly forced FlashAttention; unavailable forced backends
are reported, not silently substituted. CUDA Graph replay excludes Python launch
and allocation overhead, using 100 replays per sample and five shuffled rounds.
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
