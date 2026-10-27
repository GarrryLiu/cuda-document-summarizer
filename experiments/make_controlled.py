"""Make fictional mixed-length documents for a real-GPU scheduling exercise.

Repeated text deliberately varies prompt lengths; this is not a quality corpus.
"""
import json
import random
from pathlib import Path

rng = random.Random(42)
repeats = [8, 12, 16, 24, 32, 48, 64, 128]
rng.shuffle(repeats)
base = ("Synthetic observation: this test uses repeated technical prose to vary input length. "
        "The test compares padded batching policies and does not claim real scientific findings. ")
docs = [{"id": f"controlled-{i}", "text": "Fictional benchmark report. Objective: exercise GPU inference. " +
         base * n + "Method: repeat a fixed paragraph. Findings: not reported. "
         "Limitations: artificial repetition; no real document quality evaluation."} for i, n in enumerate(repeats)]
path = Path("data/controlled.jsonl")
path.write_text("\n".join(json.dumps(doc) for doc in docs) + "\n")
print(f"Wrote {len(docs)} synthetic documents to {path}")
