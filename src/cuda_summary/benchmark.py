"""CUDA-only real model runner. No silent CPU fallback or synthetic timings."""
import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path
from .batching import batches, padding_stats, prompt, read_documents


def percentile(values, q):
    return sorted(values)[round((len(values) - 1) * q)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="experiments/gpu-results.json")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--revision", default="main")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    if min(args.batch_size, args.max_new_tokens, args.repeats) < 1:
        p.error("batch-size, max-new-tokens and repeats must be positive")
    docs = read_documents(args.input)
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        p.exit(2, "GPU dependencies unavailable. Install the gpu extra on an NVIDIA CUDA host.\n")
    if not torch.cuda.is_available():
        p.exit(2, "CUDA unavailable; no GPU benchmark was run.\n")
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
              torch_dtype=torch.float16).to("cuda").eval()
    encoded = [tokenizer.apply_chat_template(prompt(d["text"]), tokenize=True, add_generation_prompt=True) for d in docs]
    lengths = [len(ids) for ids in encoded]
    context = model.config.max_position_embeddings
    if max(lengths) + args.max_new_tokens > context:
        p.exit(2, f"Input plus generation exceeds context limit {context}; chunk documents explicitly. No silent truncation.\n")

    def generate(indices, use_cache):
        inputs = tokenizer.pad([{"input_ids": encoded[i], "attention_mask": [1] * lengths[i]} for i in indices], return_tensors="pt").to("cuda")
        with torch.inference_mode():
            outputs = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
                                     use_cache=use_cache, pad_token_id=tokenizer.pad_token_id)
        generated = outputs[:, inputs["input_ids"].shape[1]:].cpu().tolist()
        trimmed = []
        eos = model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        for row in generated:
            stop = next((j + 1 for j, token in enumerate(row) if token in eos), len(row))
            trimmed.append(row[:stop])
        return trimmed

    results = []
    for strategy in ("arrival", "length"):
        plan = batches(lengths, args.batch_size, strategy)
        for use_cache in (False, True):
            # Warm up full longest batch for each configuration, outside measurements.
            longest = max(plan, key=lambda b: max(lengths[i] for i in b))
            generate(longest, use_cache); torch.cuda.synchronize()
            runs = []
            summary_ids = {}
            for repeat in range(args.repeats):
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize(); start = time.perf_counter()
                latencies = []; output_tokens = 0
                for batch in plan:
                    torch.cuda.synchronize(); batch_start = time.perf_counter()
                    generated = generate(batch, use_cache)
                    torch.cuda.synchronize(); latencies.append(time.perf_counter() - batch_start)
                    output_tokens += sum(map(len, generated))
                    if repeat == 0:
                        summary_ids.update({docs[i]["id"]: ids for i, ids in zip(batch, generated)})
                elapsed = time.perf_counter() - start
                runs.append({"seconds": elapsed, "documents_per_second": len(docs) / elapsed,
                             "generated_tokens_per_second": output_tokens / elapsed,
                             "generated_tokens": output_tokens,
                             "batch_latency_p50_seconds": statistics.median(latencies),
                             "batch_latency_p95_seconds": percentile(latencies, .95),
                             "peak_allocated_bytes": torch.cuda.max_memory_allocated()})
            summaries = {key: tokenizer.decode(ids, skip_special_tokens=True) for key, ids in summary_ids.items()}
            results.append({"strategy": strategy, "use_cache": use_cache,
                            "padding": padding_stats(lengths, plan), "runs": runs,
                            "summaries": [{"id": d["id"], "summary": summaries[d["id"]]} for d in docs]})
    report = {"kind": "real_cuda_inference", "model": args.model,
              "resolved_revision": getattr(model.config, "_commit_hash", None),
              "arguments": vars(args), "python": platform.python_version(),
              "torch": torch.__version__, "transformers": transformers.__version__,
              "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
              "dtype": "float16", "seed": 42,
              "input_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
              "prompt_sha256": hashlib.sha256(json.dumps(prompt(""), sort_keys=True).encode()).hexdigest(),
              "results": results}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {output}")

if __name__ == "__main__":
    main()
