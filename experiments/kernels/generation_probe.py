"""Paired full generation probe; experimental replacement, never a default route."""
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
from transformers import AutoModelForCausalLM, AutoTokenizer
from cuda_summary.batching import prompt, read_documents
from cuda_summary.decode_kernel import decode_attention, extension


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--sweep',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    sweep=json.loads(Path(a.sweep).read_text());assert sweep['status']=='complete'
    assert sweep['gpu']==torch.cuda.get_device_name(), 'Do not reuse tuning from a different GPU'
    source_sha=hashlib.sha256(Path('src/cuda_summary/kernels/decode_attention.cu').read_bytes()).hexdigest()
    assert sweep['source_sha256']==source_sha, 'Kernel changed since tuning'
    extension();original=F.scaled_dot_product_attention
    revision='aa8e72537993ba99e69dfaafa59ed015b17504d1'
    tok=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct',revision=revision)
    model=AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct',revision=revision,torch_dtype=torch.float16,attn_implementation='sdpa').cuda().eval()
    docs=sorted(read_documents(a.input),key=lambda d:len(tok.encode(d['text'])))
    # New documents, separate from the shortest/median/longest replay validation.
    docs=[docs[len(docs)//4],docs[3*len(docs)//4]]
    report={'status':'running','gpu':torch.cuda.get_device_name(),'model_revision':revision,'runs':[],
            'source_sha256':source_sha,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'production_dispatch_enabled':False,'limitations':'Experimental SDPA interception is process-local and single-threaded; default pipeline unchanged. Technical-summary prompt on legal NDAs. Token hashes compare outputs, not factuality. 32-token bounded generation is not a complete-summary evaluation.'}
    with torch.inference_mode():
        for doc in docs:
            encoded=tok(tok.apply_chat_template(prompt(doc['text']),tokenize=False,add_generation_prompt=True),return_tensors='pt').to('cuda')
            for batch in (1,4):
                inputs={name:t.repeat(batch,1) for name,t in encoded.items()};length=inputs['input_ids'].shape[1]
                row=min((r for r in sweep['synthetic'] if r['batch']==batch),key=lambda r:abs(r['context']-length))
                choice=row['selected_by_synthetic_eager'];group,splits=map(int,choice.replace('g','').replace('s','').split('_'))
                calls={'custom':0,'fallback':0}
                def replacement(q,k,v,*pos,**kw):
                    # Pinned Qwen architecture: Transformers repeats each KV head contiguously.
                    # Prefill/masked/other semantics always use the original implementation.
                    supported=(not pos and q.shape[-2:]==(1,128) and q.dtype==torch.float16 and
                        q.shape[1]==16 and k.shape[1] in (2,16) and k.shape[-2]<=131072 and
                        kw.get('attn_mask') is None and not kw.get('is_causal',False) and not kw.get('dropout_p',0) and
                        kw.get('scale',128**-.5) in (None,128**-.5) and not q.requires_grad)
                    if not supported:
                        calls['fallback']+=1;return original(q,k,v,*pos,**kw)
                    ratio=k.shape[1]//2
                    calls['custom']+=1
                    return decode_attention(q.squeeze(2).contiguous(),k[:,::ratio].contiguous(),v[:,::ratio].contiguous(),splits,group_size=group).unsqueeze(2)
                def generate(mode):
                    with patch.object(F,'scaled_dot_product_attention',replacement if mode=='custom' else original):
                        return model.generate(**inputs,max_new_tokens=32,do_sample=False,pad_token_id=tok.eos_token_id)
                for mode in ('sdpa','custom'): generate(mode)
                timings={'sdpa':[],'custom':[]};hashes={'sdpa':[],'custom':[]};counts={'sdpa':[],'custom':[]};peaks={'sdpa':[],'custom':[]}
                pairs=[];rng=random.Random(42)
                for repeat in range(3):
                    order=['sdpa','custom'];rng.shuffle(order);outputs={}
                    for mode in order:
                        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                        output=generate(mode);torch.cuda.synchronize();timings[mode].append(time.perf_counter()-start)
                        peaks[mode].append(torch.cuda.max_memory_allocated());tokens=output[:,length:].cpu().tolist()
                        hashes[mode].append(hashlib.sha256(json.dumps(tokens).encode()).hexdigest());counts[mode].append(len(tokens[0]));outputs[mode]=tokens
                        del output
                    pairs.append(outputs['sdpa']==outputs['custom'])
                report['runs'].append({'document_id':doc['id'],'batch':batch,'prompt_tokens':length,'config':choice,
                    'seconds':timings,'median_seconds':{k:statistics.median(v) for k,v in timings.items()},
                    'peak_allocated_bytes':peaks,'output_token_sha256':hashes,'output_tokens':counts,
                    'paired_outputs_identical':pairs,'calls_including_warmup':calls})
                Path(a.output).write_text(json.dumps(report,indent=2)+'\n');print('generation',doc['id'],batch,flush=True)
    report['status']='complete';Path(a.output).write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
