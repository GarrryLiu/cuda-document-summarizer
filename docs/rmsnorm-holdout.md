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

The CUDA backend builds a local extension and requires a compatible CUDA toolkit
(`nvcc`), C++ compiler and Ninja; the Slurm launcher provisions these dependencies.
The ordinary `gpu` Python extra alone does not install that compiler toolchain.

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

## Source review and enablement decision

Review distinguishes changes introduced by a backend from mistakes already
present in the native summary. The observed changed cases include:

- Columbia/Fleet amendment: the source redacts the raw-material release lead
  time. Native/CUDA output 30 days; compiled outputs 21 days. Neither number is
  supported. Compiled also substitutes the ordinary expiry date for a separately
  redacted post-termination endpoint.
- Hemispherx/SCIEN supply agreement: CUDA adds territory QA approval under the
  supplier's obligations although the corresponding clause assigns it to SCIEN.
  Both native and CUDA also reverse the supply relationship in the purpose
  paragraph. Other changes mix improved coverage with missing conditions.

These are agent-assisted checks against specific source clauses, not expert
legal review or a complete factuality score. Findings and source hashes are
published with the measured results; full source text and generated summaries
remain local. A changed number is not automatically wrong, and an unchanged
number is not automatically correct: compare it with the actual clause.

**Keep native as the default.** The optional backends are research controls.
Floating-point tolerance checks and short token matches do not establish
preservation of contractual obligations over a longer autoregressive summary.
The model itself also needs evidence-grounded treatment of redactions,
exceptions and responsible parties before its outputs can be trusted for this
use case. This phase does not change the prompt or model to hide those errors.

## Measured held-out results

GPU: NVIDIA RTX A6000. 8 intact contracts, batch one, 3 randomized rounds, cap 512 tokens.

Times below are per-document medians. Positive reductions mean faster.

| Prompt tokens | Native total s | Compiled total s | CUDA total s | CUDA total reduction vs native | CUDA prefill reduction vs native | CUDA/native identical rounds |
|---:|---:|---:|---:|---:|---:|---:|
| 4748 | 10.673 | 10.723 | 10.716 | -0.40% | 8.71% | 3/3 |
| 3683 | 12.725 | 12.663 | 12.985 | -2.04% | 9.16% | 3/3 |
| 9901 | 13.058 | 13.086 | 13.090 | -0.25% | 7.92% | 3/3 |
| 11099 | 13.197 | 13.273 | 13.274 | -0.58% | 7.64% | 3/3 |
| 13402 | 13.436 | 13.454 | 13.463 | -0.19% | 7.27% | 3/3 |
| 16133 | 13.722 | 13.710 | 13.723 | -0.01% | 6.86% | 0/3 |
| 26506 | 15.074 | 15.077 | 15.546 | -3.13% | 5.80% | 0/3 |
| 22059 | 14.634 | 14.662 | 14.624 | 0.07% | 5.97% | 3/3 |

### Output and workload checks

- native: 3/24 generations ended with EOS; 21 hit the cap; 8/8 documents had identical tokens across repeated runs.
- compiled: 3/24 generations ended with EOS; 21 hit the cap; 8/8 documents had identical tokens across repeated runs.
- cuda: 3/24 generations ended with EOS; 21 hit the cap; 8/8 documents had identical tokens across repeated runs.
- compiled vs native: 21/24 token-identical pairs; sum of per-document median generation times changes by +0.12%.
- cuda vs native: 18/24 token-identical pairs; sum of per-document median generation times changes by +0.85%.

All methods generated the same token count in each paired round, although token identities can differ. Prefill uses the same source tokens; token throughput is recorded separately. EOS, token identity and timing do not certify legal factuality.

Main GPU job **138539**, completed successfully in 19m37s. Source commit:
`5afce9e24f17d8ffdce78caf38149136edf3d819`. All 14 integration/core tests
passed on GPU, followed by the 60-case operator check and its invalid-input,
stream, empty-input and alignment checks. GitHub core-tests also passed.

The 5.80–9.16% prefill reduction does **not** translate to a stable reduction in
whole-summary wall time. CUDA total times range from 0.07% faster to 3.13%
slower than native across document medians, and the sum is 0.85% slower.
Three rounds are not enough to attribute small changes to a particular hardware
or Python overhead mechanism. Compiled total time is effectively tied with native
(+0.12% in the sum); custom CUDA has no demonstrated end-to-end advantage here.
Peak allocated model memory is effectively unchanged.

Each optimized backend records 2,336 replaced prefill calls and 877,241 native
decode fallbacks including warmups. CUDA differs on Hemispherx and AFSALA;
compiled differs on Columbia/Fleet. Each backend is repeat-stable on all eight
contracts, so these observed differences recur in every round of the affected
contract. Seven of eight contracts hit the cap in every backend, motivating the
separate higher-cap output check.

For any contracts still capped at 1024, one final output-only follow-up raises
the ceiling to 2048, using the completed 1024 report as `--cap-followup-of`.
This is the final ceiling for this phase; any remaining cap hits are reported,
not silently treated as finished summaries. Follow-up timings are not pooled
with the fixed three-round 512-token comparison.

## Higher-cap completion checks

GPU jobs 138587 (11m07s) and 138690 (9m15s), both completed successfully. The
1024 cap rechecks seven contracts; the final 2048 cap rechecks three contracts
that still hit a cap in at least one backend. Both use source commit
`a32f3a2` and one round per method. Including the main study, there are **102**
measured generations, plus separate warmups. All three jobs passed the same
14 integration/core tests and 60-case kernel check.

| Contract | Last cap | Native tokens | Compiled tokens | CUDA tokens | All methods EOS? | CUDA/native same tokens? |
|---|---:|---:|---:|---:|---|---|
| NOVOINTEGRATEDSCIENCES,INC | 512 | 427 | 427 | 427 | yes | yes |
| TUNIUCORP | 1024 | 623 | 623 | 623 | yes | yes |
| Columbia Laboratories, (Bermuda) Ltd. | 1024 | 714 | 617 | 714 | yes | yes |
| USASYNTHETICFUELCORP | 1024 | 553 | 553 | 553 | yes | yes |
| SUCAMPOPHARMACEUTICALS,INC | 2048 | 2048 | 2048 | 2048 | no | yes |
| HEMISPHERX | 2048 | 1299 | 1299 | 998 | yes | no |
| AFSALABANCORPINC | 1024 | 840 | 840 | 840 | yes | no |
| TRICITYBANKSHARESCORP | 2048 | 1336 | 1336 | 1331 | yes | no |

**Seven of eight contracts reach EOS in all backends.** Sucampo is still
incomplete at 2048 tokens, so it is not a completed-summary quality result.
Among the seven completed contracts, CUDA matches native tokens on four and
compiled matches on six. The three CUDA differences are not three factuality
failures: AFSALA changes a supported recipient term, Tri City changes only the
closing meta-summary sentence, and Hemispherx has substantive duty/right
attribution errors. Compiled changes Columbia/Fleet, including an unsupported
redacted deadline and dropped conditions. These are case findings, not an
estimated population error rate.

Hemispherx CUDA emits 998 tokens versus 1299 for native/compiled. The shorter
generation must not be reported as an equal-work inference speedup. The QA
assignment issue persists after completion, and CUDA additionally attributes
an insolvency termination right to the insolvent party instead of the other
party (source VI.B.3). Native also retains significant errors and omissions.

The original 512-token prefixes are preserved in the follow-up decoded texts.
Source-grounded review findings and exact raw rounds are available in
[the results folder](../experiments/kernels/rmsnorm-holdout-results), including
[source-review.json](../experiments/kernels/rmsnorm-holdout-results/source-review.json).
Raw contracts and generated summaries remain in ignored local data.

Next work should separately address (1) full-generation/decode overhead,
(2) numerical output preservation, and (3) bounded, evidence-grounded legal
summaries that retain parties, conditions and redactions. Increasing the cap
alone does not fix the long-enumeration failure. This phase keeps the backend
opt-in and does not claim production quality parity or whole-summary speedup.
