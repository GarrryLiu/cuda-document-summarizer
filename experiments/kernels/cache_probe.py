"""Paired native DynamicCache/StaticCache experiment on intact long contracts."""
import argparse
import gc
import hashlib
import json
import random
import statistics
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,DynamicCache,StaticCache
from cuda_summary.batching import read_documents
from profile_legal import MODEL,REVISION,SYSTEM,profile


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--profile',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    previous=json.loads(Path(a.profile).read_text());assert previous['status']=='complete'
    assert previous['gpu']==torch.cuda.get_device_name()
    assert previous['input_sha256']==hashlib.sha256(Path(a.input).read_bytes()).hexdigest()
    docs={d['id']:d for d in read_documents(a.input)}
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,torch_dtype=torch.float16,attn_implementation='sdpa').cuda().eval()
    report={'status':'running','gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'revision':REVISION,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'input_sha256':previous['input_sha256'],'profile_sha256':hashlib.sha256(Path(a.profile).read_bytes()).hexdigest(),'runs':[],
        'limitations':'Native Transformers cache policies, no custom CUDA change. Three paired runs, randomized order, warmup each policy. Allocation included in prefill. Exactly 32 decode forwards; inputs already on GPU. No complete-summary quality evaluation. Batch4 repeats one contract. Uses the same contracts as bottleneck discovery, not held-out validation. Profiling is separate from timing. Default production policy unchanged.'}
    save=lambda:Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    with torch.inference_mode():
        for previous_row in previous['runs']:
            if previous_row['target_tokens'] not in (16384,24576):continue
            doc=docs[previous_row['document_id']];batch=previous_row['batch']
            ids=tok.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':doc['text']}],add_generation_prompt=True)
            length=len(ids);inputs=torch.tensor([ids]*batch,device='cuda')
            prefill_positions=torch.arange(length,device='cuda');decode_positions=[torch.tensor([length+i],device='cuda') for i in range(32)]
            def prefill(mode):
                cache=DynamicCache(config=model.config) if mode=='dynamic' else StaticCache(config=model.config,max_cache_len=length+32)
                return model(input_ids=inputs,past_key_values=cache,cache_position=prefill_positions,use_cache=True,logits_to_keep=1)
            def decode(state):
                cache=state.past_key_values;token=state.logits[:,-1:].argmax(-1);tokens=[]
                for pos in decode_positions:
                    state=model(input_ids=token,past_key_values=cache,cache_position=pos,use_cache=True,logits_to_keep=1)
                    cache=state.past_key_values;token=state.logits[:,-1:].argmax(-1);tokens.append(token)
                return torch.cat(tokens,dim=1)
            def timed(mode):
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                state=prefill(mode);torch.cuda.synchronize();prefill_s=time.perf_counter()-start
                first_logits=state.logits;start=time.perf_counter();tokens=decode(state);torch.cuda.synchronize();decode_s=time.perf_counter()-start
                peak=torch.cuda.max_memory_allocated();cpu_tokens=tokens.cpu().tolist();cpu_logits=first_logits.float().cpu()
                return {'prefill_s':prefill_s,'decode_s':decode_s,'total_s':prefill_s+decode_s,'peak_bytes':peak,
                        'output_sha256':hashlib.sha256(json.dumps(cpu_tokens).encode()).hexdigest()},cpu_tokens,cpu_logits
            row={'document_id':doc['id'],'batch':batch,'prompt_tokens':length,'timings':{'dynamic':[],'static':[]},'paired_output_equal':[],'prefill_max_logit_error':[]}
            try:
                for mode in ('dynamic','static'): timed(mode)
                rng=random.Random(42)
                for _ in range(3):
                    order=['dynamic','static'];rng.shuffle(order);outputs={};logits={}
                    for mode in order:
                        measurement,outputs[mode],logits[mode]=timed(mode);row['timings'][mode].append(measurement)
                    row['paired_output_equal'].append(outputs['dynamic']==outputs['static'])
                    row['prefill_max_logit_error'].append((logits['dynamic']-logits['static']).abs().max().item())
                row['median']={mode:{key:statistics.median(x[key] for x in samples) for key in ('prefill_s','decode_s','total_s','peak_bytes')} for mode,samples in row['timings'].items()}
                if previous_row['target_tokens']==24576 and batch==4:
                    row['decode_profiles']={}
                    for mode in ('dynamic','static'):
                        state=prefill(mode);tokens,summary=profile(lambda:decode(state));row['decode_profiles'][mode]=summary;del state,tokens
                row['status']='complete'
            except torch.cuda.OutOfMemoryError:
                row['status']='out_of_memory';gc.collect();torch.cuda.empty_cache()
            report['runs'].append(row);save();print('cache probe',length,batch,row['status'],flush=True)
            gc.collect();torch.cuda.empty_cache()
    report['status']='complete';save()

if __name__=='__main__':main()
