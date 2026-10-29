"""Lazy JIT CUDA C++ extension. Import is CPU-safe; compilation is explicit."""
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import load
    return load(name='cuda_summary_decode_v1',
                sources=[str(Path(__file__).with_name('kernels')/'decode_attention.cu')],
                extra_cuda_cflags=['-O3','-lineinfo'],verbose=True)


def decode_attention(q,k,v,splits=1):
    """Q[B,Hq,128], K/V[B,Hkv,S,128]; all cached keys valid, no masks/dropout."""
    return extension().decode(q,k,v,splits)
