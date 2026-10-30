"""Real-GPU correctness/adversarial API/stream tests; never fake GPU results."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from cuda_summary.decode_kernel import decode_attention


def reference(q,k,v):
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(q.float().unsqueeze(2),k.float(),v.float(),enable_gqa=True).squeeze(2)


def run(output):
    torch.manual_seed(42);results=[]
    for B,Hq,Hkv,S,kind in [(1,16,2,1,'random'),(2,16,2,17,'random'),
            (1,16,2,513,'random'),(1,16,2,4096,'random'),(1,16,2,16384,'random'),
            (1,4,4,129,'zero_query'),(1,16,2,257,'large_logits'),(2,8,2,65,'distinct_kv_heads')]:
        q=torch.randn(B,Hq,128,device='cuda',dtype=torch.float16)
        k=torch.randn(B,Hkv,S,128,device='cuda',dtype=torch.float16);v=torch.randn_like(k)
        if kind=='zero_query':q.zero_()
        if kind=='large_logits':q.mul_(8);k.mul_(8)
        if kind=='distinct_kv_heads':
            for h in range(Hkv):v[:,h].fill_(h+1)
        expected=reference(q,k,v)
        for grouped in (False,True):
            if grouped and (Hq//Hkv)%4:continue
            for splits in (1,2,8,32):
                actual=decode_attention(q,k,v,splits,grouped=grouped)
                torch.testing.assert_close(actual.float(),expected,atol=.003,rtol=.02)
                results.append({'shape':[B,Hq,Hkv,S,128],'case':kind,'splits':splits,'grouped':grouped,
                                'max_abs_error':(actual.float()-expected).abs().max().item()})
    q=torch.randn(1,16,128,device='cuda',dtype=torch.float16)
    k=torch.randn(1,2,513,128,device='cuda',dtype=torch.float16);v=torch.randn_like(k)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    for grouped in (False,True):
        with torch.cuda.stream(stream):
            actual=decode_attention(q,k,v,8,grouped=grouped)
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(actual.float(),reference(q,k,v),atol=.003,rtol=.02)
    invalid=[(q.float(),k,v,1),(q,k,v,0),(q,k,v,129),
             (torch.randn(1,16,256,device="cuda",dtype=torch.float16)[...,::2],k,v,1),(q,k[:,:,:,:64].contiguous(),v[:,:,:,:64].contiguous(),1),
             (q.cpu(),k.cpu(),v.cpu(),1),(q.detach().clone().requires_grad_(),k,v,1)]
    for args in invalid:
        try:decode_attention(*args)
        except RuntimeError:pass
        else:raise AssertionError('invalid inputs accepted')
    unaligned=torch.empty(k.numel()+1,device='cuda',dtype=torch.float16)[1:].view_as(k)
    try:decode_attention(q,unaligned,v,8,grouped=True)
    except RuntimeError as error:assert 'aligned' in str(error)
    else:raise AssertionError('unaligned vector load accepted')
    try:decode_attention(q[:,:2].contiguous(),k,v,1,grouped=True)
    except RuntimeError as error:assert 'divisible by four' in str(error)
    else:raise AssertionError('unsupported grouped head ratio accepted')
    report={'status':'passed','cases':results,'nondefault_stream':'passed','invalid_api_cases':len(invalid)+2,
            'tolerance':{'atol':.003,'rtol':.02},'reference':'FP32 PyTorch SDPA math from the identical FP16 input values'}
    Path(output).write_text(json.dumps(report,indent=2)+'\n');print('Correctness passed:',len(results),'cases',flush=True)


if __name__=='__main__':
    import sys
    run(sys.argv[1])
