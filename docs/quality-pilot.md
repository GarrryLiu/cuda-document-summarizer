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
