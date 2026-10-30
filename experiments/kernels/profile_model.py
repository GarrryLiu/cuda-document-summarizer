"""Profile actual Qwen prefill/decode, without replacing its attention kernels."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from cuda_summary.batching import prompt,read_documents


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    model_id='Qwen/Qwen2.5-3B-Instruct';revision='aa8e72537993ba99e69dfaafa59ed015b17504d1'
    tok=AutoTokenizer.from_pretrained(model_id,revision=revision)
    model=AutoModelForCausalLM.from_pretrained(model_id,revision=revision,torch_dtype=torch.float16).cuda().eval()
    doc=read_documents(a.input)[0]
    inputs=tok(tok.apply_chat_template(prompt(doc['text']),tokenize=False,add_generation_prompt=True),return_tensors='pt').to('cuda')
    def profile(fn):
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            result=fn()
            torch.cuda.synchronize()
        events=prof.key_averages()
        rows=[{'operator':x.key,'device_type':str(x.device_type),'calls':x.count,'self_device_us':x.self_device_time_total,'total_device_us':x.device_time_total} for x in events]
        return result,sorted(rows,key=lambda x:x['self_device_us'],reverse=True)
    with torch.inference_mode():
        warm=model(**inputs,use_cache=True,logits_to_keep=1);del warm;torch.cuda.synchronize()
        result,prefill=profile(lambda:model(**inputs,use_cache=True,logits_to_keep=1))
        cache=result.past_key_values;token=result.logits[:,-1:].argmax(-1)
        def decode():
            nonlocal cache,token
            for _ in range(32):
                result=model(input_ids=token,past_key_values=cache,use_cache=True,logits_to_keep=1)
                cache=result.past_key_values;token=result.logits[:,-1:].argmax(-1)
        _,decode_rows=profile(decode)
    r={'status':'complete','model':model_id,'revision':revision,'gpu':torch.cuda.get_device_name(),
       'torch':torch.__version__,'cuda':torch.version.cuda,'attention_implementation':model.config._attn_implementation,
       'input_sha256':hashlib.sha256(Path(a.input).read_bytes()).hexdigest(),'document_id':doc['id'],
       'prompt_tokens':inputs['input_ids'].shape[1],'decode_steps':32,
       'num_query_heads':model.config.num_attention_heads,'num_kv_heads':model.config.num_key_value_heads,
       'head_dim':model.config.hidden_size//model.config.num_attention_heads,
       'prefill':prefill,'decode':decode_rows,
       'limitations':'Operator self-device time is not end-to-end wall time or Nsight hardware counters. Original Transformers SDPA path; custom kernel is not integrated into generation.'}
    Path(a.output).write_text(json.dumps(r,indent=2)+'\n');print('Model profiling completed',flush=True)


if __name__=='__main__':main()
