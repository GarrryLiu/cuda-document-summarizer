# Evidence-based summary review

Before evaluation, select and freeze a stratified sample of real documents across
length buckets. Record document ID, source hash and model configuration. Compare
the same IDs across benchmark conditions, blind reviewers to configuration, and
resolve disagreements explicitly.

For each summary:

- Mark whether each of the four requested sections is present. Section compliance
  is formatting, not factuality.
- Decompose claims and link each to a source passage. Label supported,
  contradicted or unverifiable. Check numbers, units, baselines and uncertainty.
- List key source facts before reviewing the summary; mark which were covered.
  Report coverage against that list, not a list derived from generated output.
- Record missing limitations, invented findings and instruction-following failures.

Suggested CSV columns:

```text
document_id,configuration,reviewer,source_hash,claim_count,supported_claims,contradicted_claims,unverifiable_claims,key_facts,covered_key_facts,sections_present,notes
```

Do not populate scores before source-backed review. Calculate supported-claim
fraction and key-fact coverage with explicit denominator handling for summaries
with no claims or documents with no reference facts. Store evidence annotations
alongside aggregate statistics. No quality reviews have been completed yet.
