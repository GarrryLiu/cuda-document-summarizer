"""Experimental inference RMSNorm; model defaults are not modified on import."""
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import load
    return load(name='cuda_summary_rmsnorm_v1',
                sources=[str(Path(__file__).with_name('kernels')/'rmsnorm.cu')],
                extra_cuda_cflags=['-O3','-lineinfo','--fmad=false'],verbose=True)


def fused_rmsnorm(x,weight,eps=1e-6):
    """Contiguous CUDA FP16 [rows,2048] or [batch,tokens,2048], inference only."""
    return extension().rmsnorm(x,weight,eps)


def eager_rmsnorm(x,weight,eps=1e-6):
    """Exact ordering in Transformers 4.57.6 Qwen2RMSNorm (including half cast)."""
    import torch
    fp32=x.float()
    return weight*(fp32*torch.rsqrt(fp32.pow(2).mean(-1,keepdim=True)+eps)).to(x.dtype)
