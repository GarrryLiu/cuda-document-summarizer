# Output length and evidence prompt pilot

## Design

Twenty real paper abstracts are fetched from arXiv using
`experiments/quality/fetch_corpus.py`. This is an abstract-only pilot, not a
full-report benchmark. The public manifest records source URLs and text hashes;
source text is retained under ignored `data/local/`. Inspect licensing before
redistributing the original text. Full generated predictions are retained locally;
the public report excludes generated prose and exact source quotations.

Compare the original four-heading prompt against a JSON prompt requesting exact
source evidence. Each prompt is tested at output caps 128, 256 and 512. Model
revision, FP16, arrival-order batch size 4, greedy decoding and KV caching are held
fixed. Conditions are shuffled with seed 42; each receives a short 32-token
warm-up. This pilot uses one run per condition, so timings are descriptive, not
stable performance estimates. A prompt-format change is intentionally confounded
with the evidence instruction: a later ablation must separate format from content.

## Diagnostics and interpretation

- A cap hit requires exhausting the output allowance without EOS. This detects
  token-limit termination, not every semantically incomplete answer.
- Plain-text section checks count headings. Structured output requires four typed
  fields and claim/evidence objects. An empty findings list can be schema-valid.
- Numeric overlap compares lexical numeric strings in the source and summary.
  It is not semantic fact coverage: numbers can be in the wrong context, omitted
  units may differ, and repeated numeric values collapse into a set.
- Evidence strings are excluded from summary numeric coverage, so copying source
  quotes does not automatically improve the score.
- For structured output, malformed JSON/schema is unscorable. Report both valid-
  output-only coverage and an explicitly named operational metric assigning zero
  to invalid outputs; do not hide parse failures in the denominator.
- Exact evidence checks only match source substrings. They do not verify that the
  quote supports the associated claim. Novel numeric mentions are review flags,
  not automatic hallucination labels.

Manual source-backed factuality review remains required. Do not treat these
proxy metrics or chapter completeness as validated summary quality.

## Reproduce

```sh
python3 experiments/quality/fetch_corpus.py
PYTHONPATH=src python3 -m cuda_summary.quality_benchmark \
  --input data/local/abstracts.jsonl --batch-size 4 \
  --output experiments/quality/raw-results.json
python3 experiments/quality/analyze.py experiments/quality/raw-results.json
```

The real run requires NVIDIA CUDA and the GPU dependencies. Inputs exceeding the
context window are rejected. Every batch checkpoints output so partial results
survive interruption; the runner does not currently resume completed batches.

## Source correction and review caveats

During source-title review, three hand-entered identifiers were found to point
to unintended papers. They were corrected to FlashAttention-2 (2307.08691),
PagedAttention (2309.06180) and Chinchilla (2203.15556), verified against official
arXiv titles. The full six-condition experiment was rerun on the corrected
20-abstract corpus; only the corrected run is used in the final report. The
initial exploratory output remains local and is excluded from final comparisons.
The fetcher now records official titles and verifies the corrected identifiers.

The lexical numeric diagnostic deliberately remains a simple proxy: it misses
numbers attached to alphabetic units/model sizes such as `65B` and `48GB`, and
can count model-version numerals as numeric mentions. It is not suitable for
claiming factual accuracy or a precise quantitative-fact recall rate. A later
experiment should freeze a manually labeled key-fact inventory before generation.

## Corrected-corpus results on Brown L40

2026-10-27; Qwen2.5-3B-Instruct revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`, FP16, batch 4, KV cache on.
Python 3.12.3, torch 2.14.1+cu130, Transformers 4.57.6. Full environment is in
`experiments/quality/gpu-environment.txt`. All 120 outputs completed; the input
hash was verified against the final local corpus. Fourteen tests passed both
locally and on the school node.

| Prompt | Output cap | Cap hits /20 | Four sections or valid schema /20 | Generation seconds |
|---|---:|---:|---:|---:|
| Baseline | 128 | 20 | 0 | 23.58 |
| Baseline | 256 | 12 | 11 | 46.56 |
| Baseline | 512 | 0 | 20 | 59.41 |
| Evidence JSON | 128 | 20 | 0 | 23.60 |
| Evidence JSON | 256 | 9 | 11 | 46.52 |
| Evidence JSON | 512 | 1 | 18 | 67.37 |

Baseline section counts are heading-presence diagnostics, not proof of complete
content. Structured counts require a parseable typed schema. Timings are one
exploratory run per condition, exclude warm-up/load/tokenization/text decoding,
and include CPU/GPU transfers. Do not use them as statistically established
speedups. Peak allocated memory was about 6.08 GiB for baseline and 6.13 GiB for
JSON; this is PyTorch allocation, not total device memory.

Sixteen abstracts have numbers recognized by the lexical proxy. Their baseline
macro numeric overlap was 40.0%, 83.5% and 86.0% at caps 128, 256 and 512. These
are **not factuality scores or manually labeled fact coverage**. JSON's valid-
output-only overlap was 39.6% (8 scorable documents) at 256 and 42.1% (14 scorable)
at 512. Invalid-zero operational overlap over all 16 numeric-source documents
was 19.8% and 36.8%, respectively; 128 had no valid output. Raw denominators and
per-document diagnostic values are in `experiments/quality/aggregate-results.json`.

Exact source evidence matching found 7/22 quoted passages at JSON-256 and 12/45
at JSON-512. The model frequently paraphrased while claiming to quote. An exact
match itself would still not establish entailment. Novel numeric flags also
occurred in plain summaries; no automatic hallucination rate is inferred.

## Qualitative checks and decision

Four 512-token outputs from BERT and QLoRA were spot-checked against source
abstracts; notes and prediction/source hashes are in `spot-review.json`. No
independent second reviewer or full 120-output human annotation was completed.
The BERT baseline retained its principal benchmark values. QLoRA's baseline
misstated the relationship between models and experiments and incorrectly claimed
limitations were absent. Structured outputs frequently used rewritten evidence;
the QLoRA JSON summary also lost numerical claims and omitted reported caveats.

Decision: use **baseline + 512** as the next candidate configuration for this
short-abstract workload. It eliminated observed cap hits, but does not pass a
factuality acceptance gate. Keep the evidence-JSON prompt experimental; it is
not a demonstrated improvement. Existing benchmark defaults and `main` remain
unchanged in this Draft PR. To try the candidate explicitly:

```sh
PYTHONPATH=src python3 -m cuda_summary.benchmark \
  --input data/local/abstracts.jsonl --max-new-tokens 512 \
  --batch-size 4 --revision aa8e72537993ba99e69dfaafa59ed015b17504d1
```

The next optimization should separate fact extraction from final rendering, use
source-span IDs or validated verbatim evidence, explicitly check reported caveats,
and evaluate against a manually labeled fact inventory. Freeze a held-out corpus
before tuning further. JSON-constrained decoding should be evaluated separately
from evidence prompting. Full-report chunking and repeated performance trials
remain future work. Temporary node files were cleaned after retrieval and the
GPU allocation released.
