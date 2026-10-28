# Legal-summary quality checks

The budget sweep reuses the 24-document development pilot. It is a development
experiment, not held-out validation or evidence of a legal-domain advantage.
All policies retain the same model, prompt and generation cap. No legal
annotations enter the scheduling policy or model prompt.

## Run the checks

```sh
PYTHONPATH=src python experiments/legal/quality.py experiments/legal/raw-results.json
```

This writes public diagnostic counts to `experiments/legal/quality-diagnostics.json`
and a private review packet to `data/local/legal-review-packet.json`.
The packet includes the complete source, every policy's first-repeat summary,
and ContractNLI hypotheses, original-document labels and evidence offsets/text.
Original documents and summaries are not committed to Git.

## Automated triage

The checker compares normalized durations and literal numeric anchors, and counts
summaries exceeding the requested 160-word limit (including headings). It handles
English duration numbers zero to ninety-nine, parenthesized numerals, business
versus calendar days, and first through fifth anniversaries. Date century and
locale ambiguity, larger spelled-out numbers, ranges, numeric formatting,
OCR and paraphrases can still produce false flags. A word-budget miss is an
instruction-following issue, not necessarily a factual error.

Changed summary lines mentioning negation, obligations/permissions, exceptions,
termination, survival, governing law/courts or durations are also queued against
arrival-4. Counts are public; the actual before/after lines stay in the private
packet. This detects sensitive wording changes, not semantic errors. Its counts
are separate from numeric/duration flagged-document counts.

Mismatch totals count unique normalized duration/number types per document,
not their number of repeated occurrences. These are review flags, not supported/unsupported classifications. A matching
number can be attached to the wrong party, exception or clause. The checker does
not assess negation, obligation versus permission, missing material facts,
party identity, governing law or exception preservation. Zero flags never means
a summary is accurate. Tests explicitly demonstrate the swapped-clause blind spot.

## Evidence review rubric

For each atomic summary claim, record its policy/document ID and source evidence
as character offsets, then assign supported, contradicted, unsupported-in-provided-
source, or unclear. Record omissions of material facts separately. Inspect party,
action, modality/negation, exception/condition, duration and jurisdiction rather
than relying on shared keywords. A redacted source duration must remain unknown.

ContractNLI labels evaluate a hypothesis against the ORIGINAL contract. They are
navigation aids for relevant evidence, not summary labels. NotMentioned is not a
contradiction. A short summary is not required to preserve all 17 NLI hypotheses;
coverage requirements must be defined for the intended user task before scoring.

## Pilot source inspection

`pilot-source-audit.json` records five initially flagged numeric/duration claims
in the old A6000 arrival-4 outputs. Codex inspected the cited source locations;
this is not an independent human or legal-expert evaluation. Four lexical flags
were explained by date/ordinal/anniversary normalization. For document 381, the
source redacts the anniversary defining the restriction period, while the summary
supplies one year. This is unsupported by the provided source. The findings cover
only those five named claims, not entire summaries or population-level accuracy.
The checker was refined on these examples, so they are development cases.

Before claiming quality-preserving optimization, freeze the rubric and checker,
use a larger disjoint evaluation set, review atomic claims and omissions across
policies (ideally blinded to policy), adjudicate disagreements and report both
quality uncertainty and the measured speed/memory frontier. Repeated identical
hashes are numerical reproducibility evidence, not factuality evidence.
