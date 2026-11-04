# Held-out contract summarization and opt-in RMSNorm

This phase checks whether prefill normalization fusion still helps when the
product generates a longer contract summary, and whether FP16 differences alter
that summary. It retains native SDPA attention, dynamic KV cache and native
single-token decode normalization. No weights are trained or fine-tuned.

## Use the backend

The default is `native`. The ordinary benchmark accepts:

```bash
python -m cuda_summary.benchmark --input your-documents.jsonl \
  --model Qwen/Qwen2.5-3B-Instruct \
  --revision aa8e72537993ba99e69dfaafa59ed015b17504d1 \
  --rmsnorm-backend cuda --output experiments/gpu-results.json
```

Use `--rmsnorm-backend compiled` for the Inductor baseline. This changes only
normalization; the ordinary benchmark still uses its technical-report prompt
and compares both caching and batching strategies. The legal study below uses
a contract-specific prompt and only dynamic caching.

For a separately loaded Qwen model:

```python
import torch
from cuda_summary.rmsnorm import rmsnorm_backend

model.eval()
with rmsnorm_backend(model, 'cuda') as dispatch:
    with torch.inference_mode():
        result = model.generate(**inputs, do_sample=False, max_new_tokens=512)
print(dispatch)
```

Only the exact Transformers `Qwen2RMSNorm` class is patched. Eligible inputs are
contiguous CUDA FP16 `[batch, tokens, 2048]`, with more than one token, at most
1,048,576 rows and matching contiguous FP16 weights. Evaluation mode and disabled
gradients are required. Unsupported shapes/dtypes/devices, training, gradients
and single-token decode use the existing forward. Existing instance overrides
are preserved. Counters disclose replacements and fallbacks. Exiting the context
restores the original dispatch, including on exceptions. Use an exclusively owned
model: temporarily patching forwards is not safe for concurrent requests on the
same instance. CUDA build/runtime failures propagate; they are not silently
reported as a successful optimized run. Validation is on Transformers 4.57.6.

## Frozen protocol

- Exclude **all four** document IDs in the earlier legal profile, including the
  two RMSNorm tuning inputs. This is a holdout from project tuning, not a claim
  about the model's pretraining corpus.
- Four prompt-length bins: [2048,6144), [6144,12288), [12288,20480),
  [20480,28672). Select the first two contracts in each bin ordered by
  SHA256(`holdout-v1:` + document ID). Persist IDs, lengths and source hashes
  before measurements. Never select by measured performance or output quality.
- Full intact CUAD contracts, batch one, no truncation; the same four-section
  legal prompt and model revision as the previous study.
- Greedy `generate`, EOS or 512-token cap, explicit dynamic cache. A cap hit is
  **not** considered a completed summary. Record both flags and token counts.
- Warm each method per document with two tokens (including prefill and decode).
  Three rounds with seeded randomized method order. Compilation/tokenization
  excluded; synchronized total generation wall time includes CPU orchestration.
  CUDA events around the first model forward record prefill separately.
- Record token hashes and paired equality, per-run total time, prefill time,
  generated-token throughput, peak allocated GPU memory and dispatch counts.
  Changed-length generations cannot be interpreted as equal-work speedups.
- Private raw summary packet supports side-by-side source review. Exact token
  equality demonstrates output preservation only for those measured cases;
  neither equality nor automatic numeric overlap proves legal correctness.

## Reproduce

Prepare CUAD using `experiments/kernels/prepare_cuad.py`, then run from a staged
checkout on the configured Brown GPU partition:

```bash
sbatch scripts/brown-rmsnorm-holdout.sbatch
```

The launcher runs integration tests and the existing adversarial kernel checks
before the holdout study. It pins PyTorch 2.14.1, Transformers 4.57.6 and CUDA 13
build wheels, and uses node-local model/compiler caches. Original source contract
text and raw generated summaries are not committed to the repository.

## Cap follow-up

Cap hits justify a separate output-completeness check, without changing the
fixed 512-token performance experiment. `--cap-followup-of <completed-report>`
selects only original contracts with a cap hit in any method or round. It verifies
input, model revision and prompt, and records the referenced report's SHA256.
Use a higher cap and one round:

```bash
sbatch scripts/brown-rmsnorm-holdout.sbatch \
  --cap-followup-of experiments/kernels/rmsnorm-holdout-results/rmsnorm-holdout.json \
  --max-new-tokens 1024 --repeats 1
```

A single follow-up round is for output inspection, not a stable speed estimate.
The selector remains independent of which backend is faster or produces a
preferred answer. Persistent cap hits, if any, remain explicitly incomplete.
