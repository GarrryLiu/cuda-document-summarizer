"""Held-out full-generation comparison; raw summaries stay in a private packet."""
import argparse
import gc
import hashlib
import json
import random
import statistics
import time
from pathlib import Path
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from cuda_summary.batching import read_documents
from cuda_summary.rmsnorm import rmsnorm_backend, extension
from profile_legal import MODEL, REVISION, SYSTEM


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--profile', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--private-output', required=True)
    p.add_argument('--max-new-tokens', type=int, default=512)
    p.add_argument('--repeats', type=int, default=3)
    a = p.parse_args()
    if a.max_new_tokens < 1 or a.repeats < 1:
        p.error('positive generation cap and repeats required')
    profile = json.loads(Path(a.profile).read_text())
    assert profile['status'] == 'complete'
    assert profile['input_sha256'] == hashlib.sha256(Path(a.input).read_bytes()).hexdigest()
    excluded = {r['document_id'] for r in profile['runs']}
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    pool = []
    for doc in read_documents(a.input):
        if doc['id'] in excluded:
            continue
        ids = tok.apply_chat_template([{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': doc['text']}], add_generation_prompt=True)
        if len(ids) + a.max_new_tokens <= 32768:
            pool.append((doc, ids))
    selected = []
    for lo, hi in ((2048, 6144), (6144, 12288), (12288, 20480), (20480, 28672)):
        candidates = sorted((x for x in pool if lo <= len(x[1]) < hi),
                            key=lambda x: sha('holdout-v1:' + x[0]['id']))
        if len(candidates) < 2:
            raise ValueError(f'Not enough contracts in [{lo},{hi})')
        selected.extend(candidates[:2])
    report = {'status': 'running', 'model': MODEL, 'revision': REVISION,
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
        'input_sha256': profile['input_sha256'], 'prompt_sha256': sha(SYSTEM),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'integration_sha256': hashlib.sha256(Path('src/cuda_summary/rmsnorm.py').read_bytes()).hexdigest(),
        'excluded_ids': sorted(excluded), 'max_new_tokens': a.max_new_tokens,
        'repeats': a.repeats, 'batch': 1, 'selection': 'First two SHA256(holdout-v1:id) in each fixed length bin; excludes all four prior profiling contracts. No outcome-based selection.',
        'limitations': 'Held out from this project tuning only, not model pretraining. B1, greedy FP16 SDPA dynamic cache. EOS or cap, no truncation. No expert factuality labels. Timings exclude compilation and tokenization; include generate CPU orchestration. First-forward CUDA event measures prefill only. Raw summaries in private packet.',
        'manifest': [{'id': d['id'], 'text_sha256': sha(d['text']), 'prompt_tokens': len(ids)} for d, ids in selected],
        'results': []}
    packet = {'documents': []}
    def save():
        Path(a.output).write_text(json.dumps(report, indent=2) + '\n')
        Path(a.private_output).write_text(json.dumps(packet, indent=2) + '\n')
    save()  # Persist selection before any measurements.
    torch.manual_seed(42)
    extension()
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
        torch_dtype=torch.float16, attn_implementation='sdpa').cuda().eval()
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    with torch.inference_mode():
        # One context per mode keeps compiler lifetime outside timed calls.
        # Compiled callable is cached below by reusing the installed forward set.
        forwards = {}
        counters = {}
        norms = [m for m in model.modules() if m.__class__.__name__ == 'Qwen2RMSNorm']
        for mode in ('native', 'compiled', 'cuda'):
            with rmsnorm_backend(model, mode) as stats:
                forwards[mode] = [m.forward for m in norms]
                counters[mode] = stats
        def install(mode):
            for m, forward in zip(norms, forwards[mode]):
                m.forward = forward
        for doc, ids in selected:
            inputs = torch.tensor([ids], device='cuda')
            mask = torch.ones_like(inputs)
            def generate(mode, cap):
                install(mode)
                events = []
                def before(module, args):
                    if not events:
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record(); events.extend([start, end])
                first = [True]
                def after(module, args, output):
                    if first[0]:
                        events[1].record(); first[0] = False
                pre = model.register_forward_pre_hook(before)
                post = model.register_forward_hook(after)
                try:
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    output = model.generate(input_ids=inputs, attention_mask=mask,
                        do_sample=False, max_new_tokens=cap, use_cache=True,
                        pad_token_id=pad, cache_implementation='dynamic')
                    torch.cuda.synchronize(); elapsed = time.perf_counter() - started
                    prefill_ms = events[0].elapsed_time(events[1])
                    peak = torch.cuda.max_memory_allocated()
                finally:
                    pre.remove(); post.remove()
                tokens = output[0, len(ids):].tolist()
                ended = bool(tokens and tokens[-1] in eos)
                return {'seconds': elapsed, 'prefill_gpu_ms': prefill_ms,
                        'generated_tokens': len(tokens), 'tokens_per_second': len(tokens)/elapsed,
                        'peak_bytes': peak, 'ended_with_eos': ended, 'cap_hit': not ended and len(tokens) == cap,
                        'output_sha256': sha(json.dumps(tokens))}, tokens
            # Compile shape-specific specializations and warm decode outside timing.
            for mode in forwards:
                generate(mode, 2)
            row = {'id': doc['id'], 'prompt_tokens': len(ids), 'runs': {m: [] for m in forwards},
                   'paired_token_equal': {'compiled': [], 'cuda': []}}
            private = {'id': doc['id'], 'summaries': {m: [] for m in forwards}}
            rng = random.Random(42)
            for repeat in range(a.repeats):
                order = list(forwards); rng.shuffle(order); tokens = {}
                for mode in order:
                    measurement, tokens[mode] = generate(mode, a.max_new_tokens)
                    row['runs'][mode].append(measurement)
                    private['summaries'][mode].append(tok.decode(tokens[mode], skip_special_tokens=True))
                    print(doc['id'], repeat, mode, measurement['seconds'], measurement['generated_tokens'], flush=True)
                for mode in ('compiled', 'cuda'):
                    row['paired_token_equal'][mode].append(tokens[mode] == tokens['native'])
            row['median'] = {m: {k: statistics.median(x[k] for x in samples)
                for k in ('seconds', 'prefill_gpu_ms', 'generated_tokens', 'tokens_per_second', 'peak_bytes')}
                for m, samples in row['runs'].items()}
            report['results'].append(row); packet['documents'].append(private); save()
            del inputs, mask; gc.collect(); torch.cuda.empty_cache()
        report['dispatch_including_warmup'] = counters
        install('native')
        for module in norms:
            del module.forward
    report['status'] = 'complete'; save()


if __name__ == '__main__':
    main()
