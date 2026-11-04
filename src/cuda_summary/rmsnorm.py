"""Experimental inference RMSNorm; model defaults are not modified on import."""
from contextlib import contextmanager
from functools import lru_cache
import math
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


@contextmanager
def rmsnorm_backend(model, backend='native'):
    """Opt-in Qwen2 prefill backend; restore forwards even when inference fails.

    Use with an exclusively owned model (not concurrent requests). Native remains
    the default. Unsupported inputs and decode call the original forward. Build
    or runtime errors on supported inputs propagate instead of hiding failures.
    Only the exact Transformers Qwen2RMSNorm implementation is eligible.
    """
    if backend not in ('native', 'compiled', 'cuda'):
        raise ValueError(f'Unknown RMSNorm backend: {backend}')
    stats = {'backend': backend, 'modules': 0, 'replaced': 0, 'fallback': 0}
    if backend == 'native':
        yield stats
        return
    import torch
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    fn = fused_rmsnorm if backend == 'cuda' else torch.compile(
        eager_rmsnorm, dynamic=True, fullgraph=True)
    originals = []
    try:
        for module in model.modules():
            if type(module) is not Qwen2RMSNorm or 'forward' in module.__dict__:
                continue
            old = module.forward
            originals.append(module)
            stats['modules'] += 1

            def forward(x, module=module, old=old):
                weight = module.weight
                eps = module.variance_epsilon
                supported = (
                    not module.training and not torch.is_grad_enabled()
                    and x.ndim == 3 and x.shape[-2] > 1 and x.shape[-1] == 2048
                    and x.numel() // 2048 <= 1048576
                    and x.is_cuda and x.dtype == torch.float16 and x.is_contiguous()
                    and weight.device == x.device and weight.dtype == x.dtype
                    and tuple(weight.shape) == (2048,) and weight.is_contiguous()
                    and math.isfinite(eps) and 0 < eps <= 1)
                if not supported:
                    stats['fallback'] += 1
                    return old(x)
                stats['replaced'] += 1
                return fn(x.reshape(-1, 2048), weight, eps).view_as(x)

            module.forward = forward
        yield stats
    finally:
        for module in originals:
            del module.forward
