// SPDX-License-Identifier: Apache-2.0
// DeepSpeed Team
// Experimental decode kernels. No production dispatch is changed by this file.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

using bf16 = __nv_bfloat16;
template <typename T>
__device__ float state_float(T x)
{
    return x;
}
template <>
__device__ float state_float<bf16>(bf16 x)
{
    return __bfloat162float(x);
}
template <typename T>
__device__ T state_round(float x)
{
    return x;
}
template <>
__device__ bf16 state_round<bf16>(float x)
{
    return __float2bfloat16_rn(x);
}

__device__ __forceinline__ float warp_sum(float x)
{
    for (int d = 16; d; d >>= 1) x += __shfl_down_sync(0xffffffff, x, d);
    return __shfl_sync(0xffffffff, x, 0);
}

// Keep the unrounded residual sum in registers for the normalization, while
// storing its BF16 rounding to the residual stream (the existing DS contract).
template <int D, int NT>
__global__ void add_norm(bf16* __restrict__ hidden,
                         const bf16* __restrict__ residual,
                         const bf16* __restrict__ weight,
                         bf16* __restrict__ out,
                         float eps,
                         bool zero_centered)
{
    constexpr int ITEMS = (D + 2 * NT - 1) / (2 * NT);
    const int tid = threadIdx.x, row = blockIdx.x;
    float2 values[ITEMS];
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        const int col = 2 * (tid + j * NT);
        float2 v = {0.f, 0.f};
        if (col < D) {
            const auto h = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(hidden + row * D + col));
            const auto r = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(residual + row * D + col));
            v = {h.x + r.x, h.y + r.y};
            *reinterpret_cast<__nv_bfloat162*>(hidden + row * D + col) =
                __floats2bfloat162_rn(v.x, v.y);
            ss += v.x * v.x + v.y * v.y;
        }
        values[j] = v;
    }
    __shared__ float partial[NT / 32];
    ss = warp_sum(ss);
    if (!(tid & 31)) partial[tid / 32] = ss;
    __syncthreads();
    if (tid < 32) {
        ss = warp_sum(tid < NT / 32 ? partial[tid] : 0.f);
        if (!tid) partial[0] = rsqrtf(ss / D + eps);
    }
    __syncthreads();
    const float rms = partial[0];
#pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        const int col = 2 * (tid + j * NT);
        if (col < D) {
            auto w = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(weight + col));
            if (zero_centered) {
                w.x += 1.f;
                w.y += 1.f;
            }
            *reinterpret_cast<__nv_bfloat162*>(out + row * D + col) =
                __floats2bfloat162_rn(values[j].x * rms * w.x, values[j].y * rms * w.y);
        }
    }
}

// Native DS state is [batch, value_head, K, V]. Lanes span V so state loads
// and stores are coalesced. Warps partition K; two small shared reductions
// combine their predictions. Q/K head grouping and gates stay in this launch.
template <int WARPS, typename ST>
__global__ void packed_gdn(const bf16* qkv,
                           const bf16* a,
                           const bf16* b,
                           const float* a_log,
                           const float* bias,
                           const ST* state,
                           ST* next,
                           bf16* out,
                           int H,
                           int HV,
                           int qs,
                           int as,
                           int bs)
{
    constexpr int K = 128, V = 128, RK = K / WARPS;
    const int tile = blockIdx.x % 4, bh = blockIdx.x / 4;
    const int batch = bh / HV, head = bh % HV, qhead = head / (HV / H);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int vi = tile * 32 + lane;
    __shared__ float qn[K], kn[K], partial[WARPS][32], gates[2];
    if (warp == 0) {
        float q[4], k[4], sq = 0.f, sk = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            q[j] = __bfloat162float(qkv[batch * qs + qhead * K + lane + j * 32]);
            k[j] = __bfloat162float(qkv[batch * qs + H * K + qhead * K + lane + j * 32]);
            sq += q[j] * q[j];
            sk += k[j] * k[j];
        }
        const float iq = rsqrtf(warp_sum(sq) + 1e-6f) * rsqrtf(float(K));
        const float ik = rsqrtf(warp_sum(sk) + 1e-6f);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            qn[lane + j * 32] = q[j] * iq;
            kn[lane + j * 32] = k[j] * ik;
        }
        if (!lane) {
            const float x = __bfloat162float(a[batch * as + head]) + bias[head];
            const float sp = x > 20.f ? x : (x < -10.f ? log1pf(expf(x)) : logf(1.f + expf(x)));
            // Preserve the existing BF16 intermediate gate rounding.
            const float g = __bfloat162float(__float2bfloat16_rn(-expf(a_log[head]) * sp));
            gates[0] = expf(g);
            gates[1] = __bfloat162float(
                __float2bfloat16_rn(1.f / (1.f + expf(-__bfloat162float(b[batch * bs + head])))));
        }
    }
    __syncthreads();
    float s[RK], keys[RK], queries[RK], pred = 0.f;
    const int64_t base = int64_t(bh) * K * V + vi;
#pragma unroll
    for (int j = 0; j < RK; ++j) {
        const int ki = j * WARPS + warp;
        s[j] = state_float(state[base + ki * V]) * gates[0];
        keys[j] = kn[ki];
        queries[j] = qn[ki];
        pred += s[j] * keys[j];
    }
    partial[warp][lane] = pred;
    __syncthreads();
    pred = 0.f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) pred += partial[w][lane];
    const float val = __bfloat162float(qkv[batch * qs + 2 * H * K + head * V + vi]);
    const float delta = (val - pred) * gates[1];
    // All warps must finish reading partial before it is reused.
    __syncthreads();
    float result = 0.f;
#pragma unroll
    for (int j = 0; j < RK; ++j) {
        const int ki = j * WARPS + warp;
        s[j] += keys[j] * delta;
        next[base + ki * V] = state_round<ST>(s[j]);
        result += s[j] * queries[j];
    }
    partial[warp][lane] = result;
    __syncthreads();
    if (warp == 0) {
        result = 0.f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) result += partial[w][lane];
        out[bh * V + vi] = __float2bfloat16_rn(result);
    }
}

void check_tensor(const at::Tensor& x,
                  const at::Tensor& ref,
                  at::ScalarType type,
                  bool contiguous = true)
{
    TORCH_CHECK(x.is_cuda() && x.device() == ref.device() && x.scalar_type() == type,
                "device or dtype mismatch");
    TORCH_CHECK(!contiguous || x.is_contiguous(), "expected contiguous tensor");
}

void norm_launch(at::Tensor h,
                 at::Tensor r,
                 at::Tensor w,
                 at::Tensor o,
                 double eps,
                 bool zero_centered,
                 int threads)
{
    for (const auto& x : {h, r, w, o}) check_tensor(x, h, at::kBFloat16);
    TORCH_CHECK(h.dim() >= 2 && h.sizes() == r.sizes() && h.sizes() == o.sizes(),
                "norm shape mismatch");
    const int d = h.size(-1), rows = h.numel() / d;
    TORCH_CHECK(w.numel() == d && (threads == 128 || threads == 256),
                "unsupported norm configuration");
    // The experimental caller supplies separate allocations for all buffers.
    const at::Tensor buffers[] = {h, r, w, o};
    for (int i = 0; i < 4; ++i)
        for (int j = i + 1; j < 4; ++j)
            TORCH_CHECK(
                buffers[i].storage().data_ptr().get() != buffers[j].storage().data_ptr().get(),
                "norm buffers must not alias");
    const c10::cuda::CUDAGuard guard(h.device());
    auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH_NORM(D, N)                                       \
    add_norm<D, N><<<rows, N, 0, stream>>>((bf16*)h.data_ptr(), \
                                           (bf16*)r.data_ptr(), \
                                           (bf16*)w.data_ptr(), \
                                           (bf16*)o.data_ptr(), \
                                           eps,                 \
                                           zero_centered)
    if (d == 2560) {
        if (threads == 128) {
            LAUNCH_NORM(2560, 128);
        } else {
            LAUNCH_NORM(2560, 256);
        }
    } else if (d == 4096) {
        if (threads == 128) {
            LAUNCH_NORM(4096, 128);
        } else {
            LAUNCH_NORM(4096, 256);
        }
    } else
        TORCH_CHECK(false, "prototype supports hidden widths 2560 and 4096");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gdn_launch(at::Tensor qkv,
                at::Tensor a,
                at::Tensor b,
                at::Tensor a_log,
                at::Tensor bias,
                at::Tensor state,
                at::Tensor next,
                at::Tensor out,
                int H,
                int warps)
{
    for (const auto& x : {qkv, a, b}) {
        check_tensor(x, qkv, at::kBFloat16, false);
        TORCH_CHECK(x.dim() == 2 && x.stride(1) == 1, "packed inputs must be row-strided matrices");
    }
    for (const auto& x : {a_log, bias}) check_tensor(x, qkv, at::kFloat);
    TORCH_CHECK(state.scalar_type() == at::kFloat || state.scalar_type() == at::kBFloat16,
                "state must be FP32 or BF16");
    for (const auto& x : {state, next}) check_tensor(x, qkv, state.scalar_type());
    check_tensor(out, qkv, at::kBFloat16);
    TORCH_CHECK(state.dim() == 4 && state.size(2) == 128 && state.size(3) == 128,
                "expected [B,HV,128,128] state");
    const int B = state.size(0), HV = state.size(1);
    TORCH_CHECK(H > 0 && HV % H == 0 && qkv.size(0) == B && qkv.size(1) == 2 * H * 128 + HV * 128,
                "invalid head packing");
    TORCH_CHECK(a.size(0) == B && b.size(0) == B && a.size(1) == HV && b.size(1) == HV,
                "gate shape mismatch");
    TORCH_CHECK(a_log.numel() == HV && bias.numel() == HV && next.sizes() == state.sizes() &&
                    out.numel() == B * HV * 128,
                "output or parameter shape mismatch");
    TORCH_CHECK(warps == 4 || warps == 8, "expected 4 or 8 warps");
    const c10::cuda::CUDAGuard guard(qkv.device());
    auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH_GDN(W, ST)                                                         \
    packed_gdn<W, ST><<<B * HV * 4, W * 32, 0, stream>>>((bf16*)qkv.data_ptr(),   \
                                                         (bf16*)a.data_ptr(),     \
                                                         (bf16*)b.data_ptr(),     \
                                                         a_log.data_ptr<float>(), \
                                                         bias.data_ptr<float>(),  \
                                                         (ST*)state.data_ptr(),   \
                                                         (ST*)next.data_ptr(),    \
                                                         (bf16*)out.data_ptr(),   \
                                                         H,                       \
                                                         HV,                      \
                                                         qkv.stride(0),           \
                                                         a.stride(0),             \
                                                         b.stride(0))
    if (state.scalar_type() == at::kFloat) {
        if (warps == 4) {
            LAUNCH_GDN(4, float);
        } else {
            LAUNCH_GDN(8, float);
        }
    } else {
        if (warps == 4) {
            LAUNCH_GDN(4, bf16);
        } else {
            LAUNCH_GDN(8, bf16);
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("add_norm", &norm_launch);
    m.def("packed_gdn", &gdn_launch);
}
