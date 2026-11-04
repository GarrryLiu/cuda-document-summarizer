"""GPU correctness, rounding and API boundary checks for the fused RMSNorm."""
import json
import sys
from pathlib import Path
import torch
from cuda_summary.rmsnorm import eager_rmsnorm,fused_rmsnorm


def main(output):
    torch.manual_seed(42);cases=[]
    for shape in ((1,2048),(7,2048),(257,2048),(2,17,2048),(4112,2048),(22899,2048)):
        for kind in ('random','zero','tiny','large','mixed'):
            x=torch.randn(*shape,device='cuda',dtype=torch.float16)
            w=torch.randn(2048,device='cuda',dtype=torch.float16)
            if kind=='zero':x.zero_()
            if kind=='tiny':x.mul_(1e-5)
            if kind=='large':x.fill_(65504);x[...,::2].neg_()
            if kind=='mixed':x[...,0]=65504;x[...,1]=1e-7
            for eps in (1e-6,1e-5):
                expected=eager_rmsnorm(x,w,eps);actual=fused_rmsnorm(x,w,eps)
                torch.testing.assert_close(actual,expected,atol=.008,rtol=.002)
                assert torch.isfinite(actual).all()
                cases.append({'shape':list(shape),'case':kind,'eps':eps,'max_abs_error':(actual.float()-expected.float()).abs().max().item()})
    x=torch.randn(7,2048,device='cuda',dtype=torch.float16);w=torch.randn(2048,device='cuda',dtype=torch.float16)
    empty=x[:0];assert fused_rmsnorm(empty,w).shape==empty.shape
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream): actual=fused_rmsnorm(x,w)
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(actual,eager_rmsnorm(x,w),atol=.008,rtol=.002)
    # Scalar half loads also accept contiguous storage with an odd half offset.
    unaligned=torch.empty(x.numel()+1,device='cuda',dtype=torch.float16)[1:].view_as(x);unaligned.copy_(x)
    torch.testing.assert_close(fused_rmsnorm(unaligned,w),eager_rmsnorm(x,w),atol=.008,rtol=.002)
    invalid=[(x.cpu(),w.cpu(),1e-6),(x.float(),w,1e-6),(x,w.float(),1e-6),
             (x[:,:1024],w,1e-6),(x,w[:1024],1e-6),
             (torch.randn(7,4096,device='cuda',dtype=torch.float16)[:,::2],w,1e-6),
             (x,torch.randn(4096,device='cuda',dtype=torch.float16)[::2],1e-6),
             (x,w,0),(x,w,float('nan')),(x,w,-1),(x,w,float('inf')),
             (x.clone().requires_grad_(),w,1e-6),(x,w.clone().requires_grad_(),1e-6)]
    for args in invalid:
        try:fused_rmsnorm(*args)
        except RuntimeError:pass
        else:raise AssertionError('Invalid API arguments accepted')
    with torch.inference_mode():fused_rmsnorm(x,w.clone().requires_grad_())
    Path(output).write_text(json.dumps({'status':'passed','cases':cases,'invalid_api_cases':len(invalid),
        'nondefault_stream':'passed','empty_and_unaligned':'passed','atol':.008,'rtol':.002,
        'reference':'Pinned Qwen FP32 normalize, round to FP16, multiply FP16 weights'},indent=2)+'\n')
    print('RMSNorm correctness passed',len(cases),flush=True)

if __name__=='__main__':main(sys.argv[1])
