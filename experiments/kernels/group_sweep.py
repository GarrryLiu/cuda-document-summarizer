"""Controlled grouped-kernel sweep and held-out, actual-model attention replay.

Raw Q/K/V and document text remain private. Synthetic tuning is separated from
real-document evaluation. CUDA Graph timings measure warm-cache GPU work;
wall timings include contiguous-layout conversion and the Python wrapper.
"""
import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F
from cuda_summary.decode_kernel import decode_attention, extension

ORIGINAL_SDPA = F.scaled_dot_product_attention


def measure(methods, reference, rounds=7):
    graphs = {}; errors = {}; timings = {name: [] for name in methods}
    for name, fn in methods.items():
        for _ in range(3): result = fn()
        torch.cuda.synchronize()
        torch.testing.assert_close(result.float(), reference.float(), atol=.004, rtol=.02)
        errors[name] = (result.float()-reference.float()).abs().max().item()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(50): result = fn()
        graph.replay(); torch.cuda.synchronize()
        torch.testing.assert_close(result.float(), reference.float(), atol=.004, rtol=.02)
        graphs[name] = (graph, result)
    rng = random.Random(42)
    for _ in range(rounds):
        order = list(methods); rng.shuffle(order)
        for name in order:
            start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
            start.record(); graphs[name][0].replay(); end.record(); end.synchronize()
            timings[name].append(start.elapsed_time(end)*1000/50)
    # Separately time realistic eager calls, not Python replay cadence.
    wall = {name: [] for name in methods}
    for _ in range(rounds):
        order = list(methods); rng.shuffle(order)
        for name in order:
            torch.cuda.synchronize(); start = time.perf_counter()
            for _ in range(10): methods[name]()
            torch.cuda.synchronize()
            wall[name].append((time.perf_counter()-start)*1e6/10)
    memory = {}
    for name, fn in methods.items():
        torch.cuda.synchronize(); before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats(); result = fn(); torch.cuda.synchronize()
        memory[name] = torch.cuda.max_memory_allocated()-before
        del result
    return {'gpu_us': {k: statistics.median(v) for k,v in timings.items()},
            'gpu_rounds_us': timings, 'eager_us': {k: statistics.median(v) for k,v in wall.items()},
            'eager_rounds_us': wall, 'incremental_peak_bytes': memory, 'max_abs_error': errors}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--input', required=True); p.add_argument('--output', required=True)
    args = p.parse_args(); torch.manual_seed(42); extension()
    report = {'status': 'running', 'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'cuda': torch.version.cuda, 'synthetic': [], 'actual': [],
              'source_sha256': hashlib.sha256(Path('src/cuda_summary/kernels/decode_attention.cu').read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'input_sha256': hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
              'method': '7 randomized rounds; 50 calls per CUDA Graph; eager 10 calls per round; identical FP16 inputs; tile32/vector16B/warp reductions held fixed; threads=32*group (coupled resource tradeoff)',
              'limitations': 'Warm-cache replay, not cold HBM or hardware occupancy counters. Batch4 replicates the same real document; no diverse-document padding. No long-document claim beyond actual token lengths. Tuning uses synthetic inputs only.'}
    def save(): Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    choices = {}
    with torch.inference_mode():
        for batch in (1,4):
            for length in (512,2048,8192,16384,32768):
                q = torch.randn(batch,16,128,device='cuda',dtype=torch.float16)
                k = torch.randn(batch,2,length,128,device='cuda',dtype=torch.float16); v = torch.randn_like(k)
                methods = {'sdpa': lambda: ORIGINAL_SDPA(q.unsqueeze(2),k,v,enable_gqa=True).squeeze(2)}
                for group in (1,2,4,8):
                    for splits in (1,4,16,32,64):
                        methods[f'g{group}_s{splits}'] = lambda g=group,s=splits: decode_attention(q,k,v,s,group_size=g)
                row = {'batch': batch, 'context': length, **measure(methods, methods['sdpa']())}
                candidates = [n for n in methods if n != 'sdpa']
                best = min(candidates, key=lambda n:row['eager_us'][n])
                choices[batch,length] = best
                row['selected_by_synthetic_eager'] = best
                report['synthetic'].append(row); save(); print('synthetic',batch,length,best,flush=True)
        del q,k,v,methods
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from cuda_summary.batching import prompt, read_documents
        revision = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
        tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct',revision=revision)
        model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct',revision=revision,
                    torch_dtype=torch.float16,attn_implementation='sdpa').cuda().eval()
        report['model_revision'] = revision
        docs = read_documents(args.input)
        docs = sorted(docs,key=lambda d:len(tok.encode(d['text'])))
        selected = [docs[0], docs[len(docs)//2], docs[-1]]
        for doc in selected:
            encoded = tok(tok.apply_chat_template(prompt(doc['text']),tokenize=False,add_generation_prompt=True),return_tensors='pt').to('cuda')
            for batch in (1,4):
                inputs = {name:t.repeat(batch,1) for name,t in encoded.items()}
                captured = []; calls = 0
                def capture(q,k,v,*pos,**kw):
                    nonlocal calls
                    if q.shape[-2] == 1:
                        # Preserve actual strides; keep first/middle/last layer only.
                        if calls in (0,17,35):
                            captured.append((calls,q.clone(),k.clone(),v.clone(),pos,kw.copy()))
                        calls += 1
                    return ORIGINAL_SDPA(q,k,v,*pos,**kw)
                with patch.object(F,'scaled_dot_product_attention',capture):
                    out = model(**inputs,use_cache=True,logits_to_keep=1)
                    token = out.logits[:,-1:].argmax(-1)
                    out = model(input_ids=token,past_key_values=out.past_key_values,use_cache=True,logits_to_keep=1)
                del out
                if len(captured) != 3: raise RuntimeError(f'Expected 36 SDPA layer calls; captured {len(captured)} from {calls}')
                for layer,q,k,v,pos,kw in captured:
                    length = k.shape[-2]; nearest = min((512,2048,8192,16384,32768),key=lambda x:abs(x-length))
                    choice = choices[batch,nearest]; group,splits = map(int,choice.replace('g','').replace('s','').split('_'))
                    # Transformers can repeat KV heads before SDPA. Check exact equality before deduplication.
                    kvheads = model.config.num_key_value_heads; ratio = k.shape[1]//kvheads
                    if ratio>1:
                        for head in range(kvheads):
                            for offset in range(1,ratio):
                                assert torch.equal(k[:,head*ratio],k[:,head*ratio+offset])
                                assert torch.equal(v[:,head*ratio],v[:,head*ratio+offset])
                    if pos or kw.get('attn_mask') is not None or kw.get('is_causal',False) or kw.get('dropout_p',0):
                        raise RuntimeError('Captured decode semantics unsupported; do not silently drop masks')
                    def convert(): return q.squeeze(2).contiguous(),k[:,::ratio].contiguous(),v[:,::ratio].contiguous()
                    cq,ck,cv = convert()
                    def custom_layout():
                        aq,ak,av=convert(); return decode_attention(aq,ak,av,splits,group_size=group).unsqueeze(2)
                    methods = {'sdpa_actual': lambda: ORIGINAL_SDPA(q,k,v,**kw),
                               'custom_prepacked': lambda: decode_attention(cq,ck,cv,splits,group_size=group).unsqueeze(2),
                               'custom_with_layout': custom_layout}
                    ref = methods['sdpa_actual']()
                    row = {'document_id':doc['id'],'batch':batch,'context':length,'layer':layer,
                           'query_stride':list(q.stride()),'key_stride':list(k.stride()),'captured_kv_heads':k.shape[1],
                           'selected_config':choice,'selection_context':nearest,**measure(methods,ref)}
                    # Conservative evidence gate: custom must win every eager round, including conversion, by 10%.
                    row['candidate_gate_pass'] = max(row['eager_rounds_us']['custom_with_layout']) < .9*min(row['eager_rounds_us']['sdpa_actual'])
                    report['actual'].append(row); save(); print('actual',doc['id'],batch,layer,row['candidate_gate_pass'],flush=True)
                del captured,q,k,v,cq,ck,cv,methods,ref
        report['status']='complete'
        report['dispatch_enabled']=False
        report['dispatch_note']='Actual replay is validation, not a trained production routing table. End-to-end integration requires a separate gate and paired generation runs.'
        save()

if __name__=='__main__': main()
