"""Real CUDA quality ablation: two prompts x three output caps, cache always on."""
import argparse
import hashlib
import json
import platform
import random
import time
from pathlib import Path
from .batching import batches,read_documents
from .quality import messages,diagnostics


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True)
    p.add_argument('--output',default='experiments/quality/gpu-quality-results.json')
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--revision',default='aa8e72537993ba99e69dfaafa59ed015b17504d1')
    p.add_argument('--model',default='Qwen/Qwen2.5-3B-Instruct')
    args=p.parse_args()
    if args.batch_size<1: p.error('batch size must be positive')
    docs=read_documents(args.input)
    import torch
    import transformers
    from transformers import AutoModelForCausalLM,AutoTokenizer
    if not torch.cuda.is_available(): p.exit(2,'CUDA unavailable; experiment not run.\n')
    torch.manual_seed(42)
    tok=AutoTokenizer.from_pretrained(args.model,revision=args.revision,padding_side='left')
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(args.model,revision=args.revision,torch_dtype=torch.float16).to('cuda').eval()
    eos=model.generation_config.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos])
    report={'kind':'cuda_quality_ablation','status':'running','model':args.model,
            'resolved_revision':getattr(model.config,'_commit_hash',None),'arguments':vars(args),
            'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,
            'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'seed':42,'repeats':1,
            'input_sha256':hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
            'scope':'20 real paper abstracts, not full reports; one exploratory run per condition; proxy diagnostics only',
            'results':[]}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    def checkpoint():
        temp=out.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(out)
    conditions=[(v,n) for v in ('baseline','evidence_json') for n in (128,256,512)]
    random.Random(42).shuffle(conditions)
    report['condition_order']=conditions
    for variant,cap in conditions:
        encoded=[tok.apply_chat_template(messages(d['text'],variant),tokenize=True,add_generation_prompt=True) for d in docs]
        if max(map(len,encoded))+cap>model.config.max_position_embeddings: raise ValueError('context exceeded')
        def generate(indices,limit):
            inputs=tok.pad([{'input_ids':encoded[i],'attention_mask':[1]*len(encoded[i])} for i in indices],return_tensors='pt').to('cuda')
            with torch.inference_mode():
                generated=model.generate(**inputs,use_cache=True,do_sample=False,max_new_tokens=limit,pad_token_id=tok.pad_token_id)
            rows=generated[:,inputs['input_ids'].shape[1]:].cpu().tolist()
            trimmed=[]
            for row in rows:
                end=next((i+1 for i,t in enumerate(row) if t in eos),len(row))
                trimmed.append(row[:end])
            return trimmed
        plan=batches(list(map(len,encoded)),args.batch_size,'arrival')
        generate(max(plan,key=lambda b:max(len(encoded[i]) for i in b)),32)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        condition={'prompt':variant,'max_new_tokens':cap,
                   'prompt_sha256':hashlib.sha256(json.dumps(messages('',variant),sort_keys=True).encode()).hexdigest(),
                   'use_cache':True,'strategy':'arrival','documents':[],'batch_timings':[]}
        report['results'].append(condition);checkpoint()
        for indices in plan:
            torch.cuda.synchronize();start=time.perf_counter()
            generated=generate(indices,cap);torch.cuda.synchronize();elapsed=time.perf_counter()-start
            condition['batch_timings'].append({'ids':[docs[i]['id'] for i in indices],'seconds':elapsed})
            for i,ids in zip(indices,generated):
                summary=tok.decode(ids,skip_special_tokens=True)
                condition['documents'].append({'id':docs[i]['id'],'summary':summary,'input_tokens':len(encoded[i]),
                    'generated_tokens':len(ids),'ended_with_eos':bool(ids and ids[-1] in eos),
                    'hit_output_cap':len(ids)>=cap and not (ids and ids[-1] in eos),
                    'diagnostics':diagnostics(docs[i]['text'],summary,variant)})
            condition['peak_allocated_bytes']=torch.cuda.max_memory_allocated();checkpoint()
            print(f'{variant} cap={cap}: {len(condition["documents"])}/{len(docs)}',flush=True)
    report['status']='complete';checkpoint()

if __name__=='__main__': main()
