// Inference-only, one query token, FP16 contiguous GQA, head dimension 128.
// Fused online-softmax tiles; optional split-KV exposes sequence parallelism.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cmath>

constexpr int D=128, TILE=64, THREADS=256;
__device__ float warp_sum(float x) {
  for (int s=16;s;s>>=1) x+=__shfl_down_sync(0xffffffff,x,s);
  return x;
}
__device__ float warp_max(float x) {
  for (int s=16;s;s>>=1) x=fmaxf(x,__shfl_down_sync(0xffffffff,x,s));
  return x;
}
template<bool MAX> __device__ float block_reduce(float x,float* scratch) {
  const int lane=threadIdx.x%32,warp=threadIdx.x/32;
  x=MAX?warp_max(x):warp_sum(x);
  if (!lane) scratch[warp]=x;
  __syncthreads();
  if (!warp) {
    x=lane<8?scratch[lane]:(MAX?-INFINITY:0.f);
    x=MAX?warp_max(x):warp_sum(x);
    if (!lane) scratch[0]=x;
  }
  __syncthreads();
  float result=scratch[0];
  __syncthreads(); // protect scratch readers before the next reduction
  return result;
}

template<bool SINGLE>
__global__ void attend(const half* q,const half* k,const half* v,half* out,
                       float* parts,float* lse,int Hq,int Hkv,int S,int splits) {
  const int tid=threadIdx.x,lane=tid%32,warp=tid/32;
  const int bh=blockIdx.x,h=bh%Hq,b=bh/Hq,split=blockIdx.y;
  const int kh=h/(Hq/Hkv),chunk=(S+splits-1)/splits;
  const int begin=split*chunk,end=min(S,begin+chunk);
  const int64_t kvbase=(int64_t(b)*Hkv+kh)*S*D;
  __shared__ float query[D],scores[TILE],scratch[8];
  if (tid<D) query[tid]=__half2float(q[int64_t(bh)*D+tid]);
  __syncthreads();
  float maximum=-INFINITY,denominator=0.f,accumulator=0.f;
  for (int start=begin;start<end;start+=TILE) {
    const int count=min(TILE,end-start);
    for (int t=warp;t<TILE;t+=8) {
      float score=0.f;
      if (t<count) {
        #pragma unroll
        for (int j=0;j<4;++j) {
          int d=lane+j*32;
          score+=query[d]*__half2float(k[kvbase+int64_t(start+t)*D+d]);
        }
      }
      score=warp_sum(score);
      if (!lane) scores[t]=t<count?score*0.08838834764831845f:-INFINITY;
    }
    __syncthreads();
    float tilemax=block_reduce<true>(tid<TILE?scores[tid]:-INFINITY,scratch);
    float nextmax=fmaxf(maximum,tilemax);
    float alpha=maximum==-INFINITY?0.f:expf(maximum-nextmax);
    float weight=tid<TILE?expf(scores[tid]-nextmax):0.f;
    if (tid<TILE) scores[tid]=weight;
    float tilesum=block_reduce<false>(weight,scratch);
    if (tid<D) {
      float value=0.f;
      for (int t=0;t<count;++t)
        value+=scores[t]*__half2float(v[kvbase+int64_t(start+t)*D+tid]);
      accumulator=accumulator*alpha+value;
    }
    denominator=denominator*alpha+tilesum;
    maximum=nextmax;
    __syncthreads(); // don't overwrite weights while V readers use them
  }
  if (SINGLE) {
    if (tid<D) out[int64_t(bh)*D+tid]=__float2half(accumulator/denominator);
  } else {
    if (tid<D) parts[(int64_t(bh)*splits+split)*D+tid]=denominator>0?accumulator/denominator:0.f;
    if (!tid) lse[int64_t(bh)*splits+split]=denominator>0?maximum+logf(denominator):-INFINITY;
  }
}

__global__ void merge_splits(const float* parts,const float* lse,half* out,int splits) {
  int bh=blockIdx.x,d=threadIdx.x;
  float maximum=-INFINITY;
  for (int s=0;s<splits;++s) maximum=fmaxf(maximum,lse[int64_t(bh)*splits+s]);
  float sum=0.f,value=0.f;
  for (int s=0;s<splits;++s) {
    float weight=expf(lse[int64_t(bh)*splits+s]-maximum);
    sum+=weight;value+=weight*parts[(int64_t(bh)*splits+s)*D+d];
  }
  out[int64_t(bh)*D+d]=__float2half(value/sum);
}

torch::Tensor decode(torch::Tensor q,torch::Tensor k,torch::Tensor v,int64_t splits) {
  TORCH_CHECK(q.is_cuda()&&k.is_cuda()&&v.is_cuda(),"CUDA tensors required");
  TORCH_CHECK(q.device()==k.device()&&q.device()==v.device(),"same CUDA device required");
  TORCH_CHECK(q.scalar_type()==torch::kFloat16&&k.scalar_type()==torch::kFloat16&&v.scalar_type()==torch::kFloat16,"FP16 required");
  TORCH_CHECK(q.is_contiguous()&&k.is_contiguous()&&v.is_contiguous(),"contiguous tensors required");
  TORCH_CHECK(q.dim()==3&&k.dim()==4&&v.sizes()==k.sizes(),"expected Q[B,Hq,128], K/V[B,Hkv,S,128]");
  TORCH_CHECK(q.size(0)==k.size(0)&&q.size(0)>0&&q.size(0)<=32,"batch must match and be 1..32");
  TORCH_CHECK(q.size(2)==D&&k.size(3)==D,"head dimension 128 required");
  TORCH_CHECK(k.size(1)>0&&q.size(1)>0&&q.size(1)<=128&&q.size(1)%k.size(1)==0,"invalid GQA heads");
  TORCH_CHECK(k.size(2)>0&&k.size(2)<=131072&&splits>=1&&splits<=128,"invalid sequence length or split count");
  TORCH_CHECK(!q.requires_grad()&&!k.requires_grad()&&!v.requires_grad(),"inference only; gradients unsupported");
  c10::cuda::CUDAGuard guard(q.device());
  auto out=torch::empty_like(q);
  auto stream=at::cuda::getCurrentCUDAStream();
  auto qp=reinterpret_cast<const half*>(q.data_ptr<at::Half>());
  auto kp=reinterpret_cast<const half*>(k.data_ptr<at::Half>());
  auto vp=reinterpret_cast<const half*>(v.data_ptr<at::Half>());
  auto op=reinterpret_cast<half*>(out.data_ptr<at::Half>());
  int B=q.size(0),Hq=q.size(1),Hkv=k.size(1),S=k.size(2);
  if (splits==1) {
    attend<true><<<dim3(B*Hq,1),THREADS,0,stream>>>(qp,kp,vp,op,nullptr,nullptr,Hq,Hkv,S,1);
  } else {
    auto parts=torch::empty({B,Hq,splits,D},q.options().dtype(torch::kFloat32));
    auto lse=torch::empty({B,Hq,splits},q.options().dtype(torch::kFloat32));
    attend<false><<<dim3(B*Hq,splits),THREADS,0,stream>>>(qp,kp,vp,op,parts.data_ptr<float>(),lse.data_ptr<float>(),Hq,Hkv,S,splits);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    merge_splits<<<B*Hq,D,0,stream>>>(parts.data_ptr<float>(),lse.data_ptr<float>(),op,splits);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("decode",&decode,"FP16 D128 split-KV decode attention"); }
