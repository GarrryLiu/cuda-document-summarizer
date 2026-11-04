"""Real RMSNorm inputs and paired prefill integration, with compiled baseline."""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import random
import statistics
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from cuda_summary.batching import read_documents
from cuda_summary.rmsnorm import eager_rmsnorm,fused_rmsnorm,extension
from profile_legal import MODEL,REVISION,SYSTEM


def microbench(x,w,eps,compiled):
    x=x.reshape(-1,2048)
    methods={'eager':lambda:eager_rmsnorm(x,w,eps),
             'compiled':lambda:compiled(x,w,eps),
             'cuda':lambda:fused_rmsnorm(x,w,eps)}
    expected=methods['eager']();graphs={};errors={};timings={k:[] for k in methods}
    for name,fn in methods.items():
        for _ in range(3):actual=fn()
        torch.cuda.synchronize();torch.testing.assert_close(actual,expected,atol=.008,rtol=.002)
        errors[name]=(actual.float()-expected.float()).abs().max().item()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(20):actual=fn()
        graph.replay();torch.cuda.synchronize()
        torch.testing.assert_close(actual,expected,atol=.008,rtol=.002)
        graphs[name]=(graph,actual)
    rng=random.Random(42)
    for _ in range(7):
        order=list(methods);rng.shuffle(order)
        for name in order:
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record();graphs[name][0].replay();end.record();end.synchronize()
            timings[name].append(start.elapsed_time(end)*1000/20)
    return {'rows':x.shape[0],'gpu_us':{k:statistics.median(v) for k,v in timings.items()},
            'gpu_rounds_us':timings,'max_abs_error':errors}


@contextmanager
def replace_prefill(norms,fn,counters):
    originals=[]
    try:
        for module in norms:
            old=module.forward;originals.append((module,old))
            def forward(x,old=old,weight=module.weight,eps=module.variance_epsilon):
                supported=(x.ndim==3 and x.shape[-2]>1 and x.shape[-1]==2048 and x.is_cuda and
                           x.dtype==torch.float16 and x.is_contiguous() and not torch.is_grad_enabled())
                if not supported:
                    counters['fallback']+=1;return old(x)
                counters['replaced']+=1
                return fn(x.reshape(-1,2048),weight,eps).view_as(x)
            module.forward=forward
        yield
    finally:
        for module,old in originals:module.forward=old


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--profile',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    reference=json.loads(Path(a.profile).read_text());assert reference['status']=='complete'
    assert reference['input_sha256']==hashlib.sha256(Path(a.input).read_bytes()).hexdigest()
    docs={d['id']:d for d in read_documents(a.input)}
    extension()
    # dynamic rows, fixed width; default Inductor mode (no nested CUDA Graph mode).
    compiled=torch.compile(eager_rmsnorm,dynamic=True,fullgraph=True)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,torch_dtype=torch.float16,attn_implementation='sdpa').cuda().eval()
    norms=[m for m in model.modules() if m.__class__.__name__=='Qwen2RMSNorm']
    assert len(norms)==73 and model.config.hidden_size==2048
    report={'status':'running','gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,
        'model':MODEL,'revision':REVISION,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path('src/cuda_summary/kernels/rmsnorm.cu').read_bytes()).hexdigest(),
        'input_sha256':reference['input_sha256'],'microbenchmarks':[],'generation':[],
        'method':'Actual norm inputs at module indices 0/36/72. Micro: 20 calls per CUDA Graph, 7 shuffled rounds. Model: eager vs compiled vs CUDA prefill-only replacement, warmup each, 3 randomized paired rounds. Compilation excluded.',
        'limitations':'Complete intact CUAD contracts but bounded 32 decode forward passes, no factuality evaluation. Batch4 repeats one contract. Same documents as previous profile, not held-out generalization. CUDA activity timings are warm-cache operator timings. Default pipeline unchanged.'}
    save=lambda:Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    with torch.inference_mode():
        for config in reference['runs']:
            if config['target_tokens'] not in (8192,24576):continue
            doc=docs[config['document_id']];batch=config['batch']
            ids=tok.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':doc['text']}],add_generation_prompt=True)
            inputs=torch.tensor([ids]*batch,device='cuda');length=len(ids)
            prefill_pos=torch.arange(length,device='cuda');decode_pos=[torch.tensor([length+i],device='cuda') for i in range(32)]
            def prefill():return model(input_ids=inputs,cache_position=prefill_pos,use_cache=True,logits_to_keep=1)
            captures=[];hooks=[]
            for index in (0,36,72):
                def capture(module,args,index=index):captures.append((index,args[0].detach(),module.weight,module.variance_epsilon))
                hooks.append(norms[index].register_forward_pre_hook(capture))
            try:state=prefill();del state
            finally:
                for hook in hooks:hook.remove()
            for index,x,w,eps in captures:
                result=microbench(x,w,eps,compiled)
                report['microbenchmarks'].append({'document_id':doc['id'],'batch':batch,'prompt_tokens':length,'module_index':index,**result})
                save();print('norm micro',length,batch,index,flush=True)
            del captures,x,w;gc.collect();torch.cuda.empty_cache()
            def decode(state):
                cache=state.past_key_values;token=state.logits[:,-1:].argmax(-1);tokens=[]
                for pos in decode_pos:
                    state=model(input_ids=token,past_key_values=cache,cache_position=pos,use_cache=True,logits_to_keep=1)
                    cache=state.past_key_values;token=state.logits[:,-1:].argmax(-1);tokens.append(token)
                return torch.cat(tokens,dim=1)
            counters={mode:{'replaced':0,'fallback':0} for mode in ('eager','compiled','cuda')}
            def timed(mode):
                @contextmanager
                def unchanged():yield
                context=unchanged() if mode=='eager' else replace_prefill(norms,compiled if mode=='compiled' else fused_rmsnorm,counters[mode])
                with context:
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                    state=prefill();torch.cuda.synchronize();prefill_s=time.perf_counter()-start
                    first=state.logits;start=time.perf_counter();tokens=decode(state);torch.cuda.synchronize();decode_s=time.perf_counter()-start
                    peak=torch.cuda.max_memory_allocated();cpu_tokens=tokens.cpu().tolist();cpu_first=first.float().cpu()
                return {'prefill_s':prefill_s,'decode_s':decode_s,'total_s':prefill_s+decode_s,'peak_bytes':peak,
                        'output_sha256':hashlib.sha256(json.dumps(cpu_tokens).encode()).hexdigest()},cpu_tokens,cpu_first
            for mode in counters:timed(mode)
            row={'document_id':doc['id'],'batch':batch,'prompt_tokens':length,'timings':{k:[] for k in counters},
                 'paired_output_equal':{'compiled':[],'cuda':[]},'first_logit_error':{'compiled':[],'cuda':[]}}
            rng=random.Random(42)
            for _ in range(3):
                order=list(counters);rng.shuffle(order);outputs={};logits={}
                for mode in order:
                    measurement,outputs[mode],logits[mode]=timed(mode);row['timings'][mode].append(measurement)
                for mode in ('compiled','cuda'):
                    row['paired_output_equal'][mode].append(outputs[mode]==outputs['eager'])
                    row['first_logit_error'][mode].append((logits[mode]-logits['eager']).abs().max().item())
            row['median']={mode:{key:statistics.median(x[key] for x in samples) for key in ('prefill_s','decode_s','total_s','peak_bytes')} for mode,samples in row['timings'].items()}
            row['calls_including_warmup']=counters;report['generation'].append(row);save();print('norm generation',length,batch,flush=True)
            del inputs;gc.collect();torch.cuda.empty_cache()
    report['status']='complete';save()

if __name__=='__main__':main()
