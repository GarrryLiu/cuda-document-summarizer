"""Lazy JIT CUDA C++ extension. Import is CPU-safe; compilation is explicit."""
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import load
    return load(name='cuda_summary_decode_v2',
                sources=[str(Path(__file__).with_name('kernels')/'decode_attention.cu')],
                extra_cuda_cflags=['-O3','-lineinfo'],verbose=True)


def decode_attention(q,k,v,splits=1,grouped=False,group_size=None):
    """Q[B,Hq,128], K/V[B,Hkv,S,128]; all cached keys valid, no masks/dropout."""
    module=extension()
    if group_size is not None or grouped:
        return module.decode_grouped(q,k,v,splits,4 if group_size is None else group_size)
    return module.decode(q,k,v,splits)
