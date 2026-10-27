"""Render a small descriptive table from a real benchmark result."""
import argparse
import json
import statistics
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("input")
a = p.parse_args()
r = json.loads(Path(a.input).read_text())
assert r["kind"] == "real_cuda_inference"
print(f"GPU: {r['gpu']}; torch: {r['torch']}; transformers: {r['transformers']}")
print("| Ordering | Cache | Median docs/s | Median tokens/s | Max allocated GiB |")
print("|---|---|---:|---:|---:|")
for c in r["results"]:
    runs = c["runs"]
    print(f"| {c['strategy']} | {c['use_cache']} | "
          f"{statistics.median(x['documents_per_second'] for x in runs):.3f} | "
          f"{statistics.median(x['generated_tokens_per_second'] for x in runs):.1f} | "
          f"{max(x['peak_allocated_bytes'] for x in runs) / 2**30:.2f} |")
