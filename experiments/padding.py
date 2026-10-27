"""Token-length scheduling experiment; does not load a tokenizer or GPU."""
import argparse
import json
import random
from pathlib import Path
from cuda_summary.batching import batches, padding_stats

p = argparse.ArgumentParser()
p.add_argument("--output", default="experiments/padding-results.json")
args = p.parse_args()
rng = random.Random(42)
lengths = [rng.randint(1024, 24576) for _ in range(400)]
results = []
for size in (1, 4, 8, 16):
    row = {"batch_size": size}
    for strategy in ("arrival", "length"):
        row[strategy] = padding_stats(lengths, batches(lengths, size, strategy))
    row["padded_position_reduction"] = 1 - row["length"]["padded_positions"] / row["arrival"]["padded_positions"]
    assert row["length"]["padded_positions"] <= row["arrival"]["padded_positions"]
    results.append(row)
report = {"kind": "synthetic_length_scheduling", "seed": 42, "documents": 400,
          "length_range": [1024, 24576], "lengths": lengths, "results": results,
          "limitation": "Lengths are sampled integers, not measured document tokens. Padding savings are not CUDA throughput, KV-cache benefit or summary quality."}
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
for row in results:
    print(f"batch={row['batch_size']}: padded-position reduction {row['padded_position_reduction']:.2%}")
