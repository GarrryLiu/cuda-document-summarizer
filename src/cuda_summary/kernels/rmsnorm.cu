// Qwen-compatible inference RMSNorm: FP32 variance, then FP16 rounding BEFORE weight multiplication.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/core/GradMode.h>
#include <cuda_fp16.h>
#include <cmath>

constexpr int WIDTH=2048, THREADS=256;
__global__ void rmsnorm_2048(const half* x,const half* weight,half* out,float eps) {
    const int tid=threadIdx.x,lane=tid%32,warp=tid/32;
    const int64_t base=int64_t(blockIdx.x)*WIDTH;
    float values[WIDTH/THREADS];
    float sum=0.f;
    #pragma unroll
    for(int j=0;j<WIDTH/THREADS;++j) {
        values[j]=__half2float(x[base+tid+j*THREADS]);
        sum+=values[j]*values[j];
    }
    for(int offset=16;offset;offset/=2) sum+=__shfl_down_sync(0xffffffff,sum,offset);
    __shared__ float partial[8];
    if(!lane) partial[warp]=sum;
    __syncthreads();
    if(!warp) {
        sum=lane<8?partial[lane]:0.f;
        for(int offset=16;offset;offset/=2) sum+=__shfl_down_sync(0xffffffff,sum,offset);
        if(!lane) partial[0]=rsqrtf(sum/float(WIDTH)+eps);
    }
    __syncthreads();
    const float inv=partial[0];
    #pragma unroll
    for(int j=0;j<WIDTH/THREADS;++j) {
        int d=tid+j*THREADS;
        half normalized=__float2half_rn(values[j]*inv);
        out[base+d]=__float2half_rn(__half2float(normalized)*__half2float(weight[d]));
    }
}

torch::Tensor rmsnorm(torch::Tensor x,torch::Tensor weight,double eps) {
    TORCH_CHECK(x.is_cuda()&&weight.is_cuda(),"CUDA tensors required");
    TORCH_CHECK(x.device()==weight.device(),"same CUDA device required");
    TORCH_CHECK(x.scalar_type()==torch::kFloat16&&weight.scalar_type()==torch::kFloat16,"FP16 required");
    TORCH_CHECK(x.dim()>=2&&x.dim()<=3&&x.size(-1)==WIDTH,"expected 2D/3D input with width 2048");
    TORCH_CHECK(weight.dim()==1&&weight.numel()==WIDTH,"weight must have width 2048");
    TORCH_CHECK(x.is_contiguous()&&weight.is_contiguous(),"contiguous tensors required");
    TORCH_CHECK(std::isfinite(eps)&&eps>0&&eps<=1,"epsilon must be finite and in (0,1]");
    TORCH_CHECK(!c10::GradMode::is_enabled()||(!x.requires_grad()&&!weight.requires_grad()),"inference only; gradients unsupported");
    const int64_t rows=x.numel()/WIDTH;
    TORCH_CHECK(rows<=1048576,"at most 1048576 rows supported");
    c10::cuda::CUDAGuard guard(x.device());
    auto out=torch::empty_like(x);
    if(rows) {
        rmsnorm_2048<<<rows,THREADS,0,at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
            reinterpret_cast<half*>(out.data_ptr<at::Half>()),float(eps));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("rmsnorm",&rmsnorm,"Fused Qwen FP16 RMSNorm D2048");}
