"""Warm-cache kernel timing, including default and forced-flash SDPA controls."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import time
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from cuda_summary.decode_kernel import decode_attention,extension


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    torch.manual_seed(42);extension()
    rows=[];rng=random.Random(42)
    report={'status':'running','gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
      'cuda':torch.version.cuda,'seed':42,'head_configuration':[16,2,128],
      'sm_count':torch.cuda.get_device_properties(0).multi_processor_count,
      'benchmark_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'source_sha256':hashlib.sha256(Path(__file__).parents[2].joinpath('src/cuda_summary/kernels/decode_attention.cu').read_bytes()).hexdigest(),
      'timing':'CUDA Graph with 100 operation calls per single replay; warm-cache fixed inputs; 5 randomized rounds. Capture and compilation excluded. Python API latency measured separately.',
      'runs':rows}
    out=Path(a.output)
    for B in (1,4):
      for S in (512,2048,8192,16384,32768):
        q=torch.randn(B,16,128,device='cuda',dtype=torch.float16)
        k=torch.randn(B,2,S,128,device='cuda',dtype=torch.float16);v=torch.randn_like(k)
        def sdpa():return F.scaled_dot_product_attention(q.unsqueeze(2),k,v,enable_gqa=True).squeeze(2)
        def flash():
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):return sdpa()
        methods={'sdpa_auto':sdpa,'sdpa_flash':flash,
                 'fused_1':lambda:decode_attention(q,k,v,1),
                 'split_4':lambda:decode_attention(q,k,v,4),
                 'split_16':lambda:decode_attention(q,k,v,16),
                 'split_32':lambda:decode_attention(q,k,v,32),
                 'grouped_16':lambda:decode_attention(q,k,v,16,grouped=True),
                 'grouped_32':lambda:decode_attention(q,k,v,32,grouped=True),
                 'grouped_64':lambda:decode_attention(q,k,v,64,grouped=True)}
        graphs={};failures={}
        expected=sdpa()
        for name,fn in methods.items():
          try:
            for _ in range(5):result=fn()
            torch.cuda.synchronize()
            torch.testing.assert_close(result,expected,atol=.004,rtol=.02)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(100):result=fn()
            graph.replay();torch.cuda.synchronize()
            torch.testing.assert_close(result,expected,atol=.004,rtol=.02)
            graphs[name]=(graph,result)
          except RuntimeError as error:
            if name!='sdpa_flash':raise
            failures[name]=str(error)
        timings={name:[] for name in graphs}
        for repeat in range(5):
          order=list(graphs);rng.shuffle(order)
          for name in order:
            graph,_=graphs[name];start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record();end.synchronize();timings[name].append(start.elapsed_time(end)*1000/100)
        api={}
        for name in graphs:
          torch.cuda.synchronize();start=time.perf_counter()
          for _ in range(20):methods[name]()
          torch.cuda.synchronize();api[name]=(time.perf_counter()-start)*1e6/20
        row={'batch':B,'context':S,'microseconds':{name:statistics.median(t) for name,t in timings.items()},
             'round_microseconds':timings,'python_api_microseconds':api,'unavailable':failures}
        rows.append(row);out.write_text(json.dumps(report,indent=2)+'\n');print(row,flush=True)
    report['status']='complete';out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
