#include <stdio.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// read from data
#include "data/config.h.in"

#if 0
constexpr int N = 7;
constexpr int D = 120;
constexpr int IH = 64;
constexpr int IW = 120;
constexpr int C = 128;
constexpr int OH = 80;
constexpr int OW = 160;
constexpr int P = 1606542;
#endif

extern "C"
int get_config(char *config) {
  int ret = -1;
  if (config == nullptr)
    return ret;

  char c1 = *config, c2 = *(config + 1);
  switch (c1) {
  case 'N':
    ret = N;
    break;
  case 'D':
    ret = D;
    break;
  case 'I':
    ret = (c2 == 'H') ? IH : (c2 == 'W') ? IW : -1;
    break;
  case 'O':
    ret = (c2 == 'H') ? OH : (c2 == 'W') ? OW : -1;
    break;
  case 'C':
    ret = C;
    break;
  case 'P':
    ret = P;
    break;
  default:
    ret = -1;
    break;
  }
  return ret;
}

void read_file(const char *filename, size_t element_sz, size_t size, void *buffer) {
  FILE *fp = fopen(filename, "rb");
  if (fp == nullptr) {
    printf("fopen error: %s\n", filename);
    return;
  }
  size_t num = fread(buffer, element_sz, size, fp);
  if (num != size) {
    printf("read error: %s\n", filename);
  };
  fclose(fp);
}

extern "C"
void tensor_init(int *ranks_depth,
                 int *ranks_feat,
                 int *ranks_bev,
                 int *interval_starts,
                 int *interval_lengths,
                 int8_t *ranks_bev_mask,
                 int *interval_starts_e,
                 int *interval_lengths_e);
void tensor_init(int *ranks_depth,
                 int *ranks_feat,
                 int *ranks_bev,
                 int *interval_starts,
                 int *interval_lengths,
                 int8_t *ranks_bev_mask,
                 int *interval_starts_e,
                 int *interval_lengths_e) {

  read_file("data/ranks_depth.bin", sizeof(float), 4000000, ranks_depth);
  read_file("data/ranks_feat.bin", sizeof(float), 4000000, ranks_feat);
  read_file("data/ranks_bev.bin", sizeof(float), 4000000, ranks_bev);
  read_file("data/interval_starts.bin", sizeof(float), 50000, interval_starts);
  read_file("data/interval_lengths.bin", sizeof(float), 50000, interval_lengths);

  for (int i = 0; i < 4000000; i++) {
    *(int*)&ranks_bev[i] = (int)*(float*)&ranks_bev[i];
    *(int*)&ranks_depth[i] = (int)*(float*)&ranks_depth[i];
    *(int*)&ranks_feat[i] = (int)*(float*)&ranks_feat[i];
    int idx = ranks_bev[i];
    if (idx != -1 && ranks_bev_mask[idx] == 0)
      ranks_bev_mask[idx] = 1;
  }
  for (int i = 0; i < 50000; i++) {
    *(int*)&interval_starts[i] = (int)*(float*)&interval_starts[i];
    *(int*)&interval_lengths[i] = (int)*(float*)&interval_lengths[i];
  }
  int j = 0;
  for (int i = 0; i < 50000; i++) {
    if (ranks_bev_mask[i] == 0) {
      interval_starts_e[i] = 0;
      interval_lengths_e[i] = 0;
    } else {
      interval_starts_e[i] = interval_starts[j];
      interval_lengths_e[i] = interval_lengths[j];
      j++;
    }
  }
}


extern "C" void tensor_NDHW_to_NHWD(int *ranks_depth, size_t num, int N, int D, int H, int W);
void tensor_NDHW_to_NHWD(int *ranks_depth, size_t num, int N, int D, int H, int W) {
#pragma omp parallel for
  for (int i = 0; i < num; i++) {
    int idx = ranks_depth[i];
      if (idx != -1) {

        int n = idx / (D*H*W);
        int d = (idx - n*D*H*W)/(H*W);
        int h = (idx - n*D*H*W - d*H*W)/W;
        int w = (idx - n*D*H*W - d*H*W - h*W);

        int new_idx = n*H*W*D + h*W*D + w*D + d;
        ranks_depth[i] = new_idx;
      }
  }
}


template<typename TensorType, typename AccType, const int TC, const int TN>
__global__ void bev_pool_flatmap_kernel(
    int C, int N, const TensorType *__restrict__ depth,
    const TensorType *__restrict__ feat, const int *__restrict__ ranks_depth,
    const int *__restrict__ ranks_feat, const int *__restrict__ ranks_bev,
    const int *__restrict__ interval_starts,
    const int *__restrict__ interval_lengths, TensorType *__restrict__ out) {

  int tc_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tn_idx = blockIdx.y * blockDim.y + threadIdx.y;

  for (int tc = 0; tc < TC; tc++) {
    int c_idx = tc_idx * TC + tc;
    if (c_idx >= C) continue;

    int b_idx_last = -1;
    int b_idx = 0;
    TensorType psum = 0;

    for (int tn = 0; tn < TN; tn++) {
      int n_idx = tn_idx * TN + tn;
      if (n_idx >= N) continue;
      b_idx = ranks_bev[n_idx];

      TensorType d = depth[ranks_depth[n_idx]];
      TensorType f = feat[ranks_feat[n_idx]*C + c_idx];

      if (b_idx == b_idx_last) {
        psum += d*f;
      } else {
        if (b_idx_last != -1)
          atomicAdd(&out[b_idx_last*C + c_idx], psum);
        b_idx_last = b_idx;
        psum = d * f;
      }
    }

    if (b_idx_last != -1)
      atomicAdd(&out[b_idx_last*C + c_idx], psum);
  }
}

extern "C"
void bev_pool_flatmap(int C, int n_intervals, const float *depth, const float *feat,
                      const int *ranks_depth, const int *ranks_feat,
                      const int *ranks_bev, const int *interval_starts,
                      const int *interval_lengths, float *out) {

  constexpr int local_TC = 1;
  constexpr int local_TN = 90;
  constexpr int local_BC = 96;
  constexpr int local_BN = 2;
  dim3 gridSize((C + local_TC * local_BC - 1)/(local_TC * local_BC), (P + local_TN * local_BN - 1)/(local_TN * local_BN));
  dim3 blockSize(local_BC, local_BN);

  cudaMemset(out, 0, OH*OW*C*sizeof(float));
  bev_pool_flatmap_kernel<float, float, local_TC, local_TN><<<gridSize, blockSize>>>(
      C, P, depth, feat, ranks_depth, ranks_feat, ranks_bev,
      interval_starts, interval_lengths, out);
}

template<typename InType, typename AccType, typename OutType, const int TC, const int TN, const int IHW=1, const int OHW=1, bool FEAT_CL=true>
__global__ void bev_pool_kernel(
    int C, int n_intervals,
    const InType *__restrict__ depth,
    const InType *__restrict__ feat,
    const int *__restrict__ ranks_depth,
    const int *__restrict__ ranks_feat,
    const int *__restrict__ interval_starts,
    const int *__restrict__ interval_lengths,
    OutType *__restrict__ out) {

  int tc_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tn_idx = blockIdx.y * blockDim.y + threadIdx.y;

#pragma unroll
  for (int tn = 0; tn < TN; tn++) {
    AccType psum[TC];
    int n_idx = tn_idx * TN + tn;
    if (n_idx >= n_intervals) break;

    int interval_start = __ldg(&interval_starts[n_idx]);
    int interval_length = __ldg(&interval_lengths[n_idx]);

    if (interval_start == -1) break;

    for (int tc = 0; tc < TC; tc++) {
      psum[tc] = 0;
    }

    for (int i = 0; i < interval_length; i++) {
      InType d = __ldg(&depth[ranks_depth[interval_start + i]]);
#pragma unroll
      for (int tc = 0; tc < TC; tc++) {
        int c_idx = tc_idx * TC + tc;
        if (c_idx >= C) continue;

        InType f;
        if constexpr (FEAT_CL)
          f = __ldg(&feat[ranks_feat[interval_start + i] * C + c_idx]);
        else {// NCHW
          int idx = ranks_feat[interval_start + i];
          int n = idx / IHW;
          int hw = idx % IHW;
          f = __ldg(&feat[n*C*IHW + c_idx*IHW + hw]);
        }

        if constexpr (std::is_same<InType, __half>::value && std::is_same<AccType, float>::value)
          psum[tc] = __fmaf_rn(__half2float(d), __half2float(f), psum[tc]);
        else if constexpr (std::is_same<InType, __nv_bfloat16>::value && std::is_same<AccType, float>::value)
          psum[tc] = __fmaf_rn(__bfloat162float(d), __bfloat162float(f), psum[tc]);
        else if constexpr (std::is_same<InType, float>::value && std::is_same<AccType, float>::value)
          psum[tc] = __fmaf_rn(d, f, psum[tc]);
        else
          psum[tc] += d * f;
      }
    }

#pragma unroll
    for (int tc = 0; tc < TC; tc++) {
      int c_idx = tc_idx * TC + tc;
      if (c_idx >= C) break;
      int tid;
      if constexpr (FEAT_CL)
        tid = n_idx * C + c_idx;
      else {
        int n = n_idx / OHW;
        int hw = n_idx % OHW;
        tid = n*C*OHW + c_idx*OHW + hw;
      }
      if constexpr (std::is_same<OutType, __half>::value && std::is_same<AccType, float>::value)
        out[tid] = __float2half(psum[tc]);
      else if constexpr (std::is_same<OutType, __nv_bfloat16>::value && std::is_same<AccType, float>::value)
        out[tid] = __float2bfloat16(psum[tc]);
      else
        out[tid] = psum[tc];
    }
  }
}

extern "C"
void bev_pool_float_float_float(int c, int n_intervals,
                                const float *depth,
                                const float *feat,
                                const int *ranks_depth,
                                const int *ranks_feat,
                                const int *interval_starts,
                                const int *interval_lengths,
                                float *out) {
  dim3 gridSize((c + TC * BC - 1)/(TC * BC), (n_intervals + TN * BN - 1)/(TN * BN));
  dim3 blockSize(BC, BN);
  bev_pool_kernel<float, float, float, TC, TN><<<gridSize, blockSize>>>(
      c, n_intervals, depth, feat, ranks_depth, ranks_feat,
      interval_starts, interval_lengths, out);
}


extern "C"
void bev_pool_float_float_float_nchw(int c, int n_intervals,
                                     const float *depth,
                                     const float *feat,
                                     const int *ranks_depth,
                                     const int *ranks_feat,
                                     const int *interval_starts,
                                     const int *interval_lengths,
                                     float *out) {
  dim3 gridSize((c + TC * BC - 1)/(TC * BC), (n_intervals + TN * BN - 1)/(TN * BN));
  dim3 blockSize(BC, BN);
  bev_pool_kernel<float, float, float, TC, TN, IH*IW, OH*OW, false><<<gridSize, blockSize>>>(
      c, n_intervals, depth, feat, ranks_depth, ranks_feat,
      interval_starts, interval_lengths, out);
}

extern "C"
void bev_pool_half_float_half(int c, int n_intervals,
                              const __half *depth,
                              const __half *feat,
                              const int *ranks_depth,
                              const int *ranks_feat,
                              const int *interval_starts,
                              const int *interval_lengths,
                              __half *out) {
  dim3 gridSize((c + TC * BC - 1)/(TC * BC), (n_intervals + TN * BN - 1)/(TN * BN));
  dim3 blockSize(BC, BN);
  bev_pool_kernel<__half, float, __half, TC, TN><<<gridSize, blockSize>>>(
      c, n_intervals, depth, feat, ranks_depth, ranks_feat,
      interval_starts, interval_lengths, out);
}

extern "C"
void bev_pool_half_float_float(int c, int n_intervals,
                                const __half *depth,
                                const __half *feat,
                                const int *ranks_depth,
                                const int *ranks_feat,
                                const int *interval_starts,
                                const int *interval_lengths,
                                float *out) {
  dim3 gridSize((c + TC * BC - 1)/(TC * BC), (n_intervals + TN * BN - 1)/(TN * BN));
  dim3 blockSize(BC, BN);
  bev_pool_kernel<__half, float, float, TC, TN><<<gridSize, blockSize>>>(
      c, n_intervals, depth, feat, ranks_depth, ranks_feat,
      interval_starts, interval_lengths, out);
}


extern "C"
void bev_pool_bf16_float_bf16(int c, int n_intervals,
                              const __nv_bfloat16 *depth,
                              const __nv_bfloat16 *feat,
                              const int *ranks_depth,
                              const int *ranks_feat,
                              const int *interval_starts,
                              const int *interval_lengths,
                              __nv_bfloat16 *out) {
  dim3 gridSize((c + TC * BC - 1)/(TC * BC), (n_intervals + TN * BN - 1)/(TN * BN));
  dim3 blockSize(BC, BN);
  bev_pool_kernel<__nv_bfloat16, float, __nv_bfloat16, TC, TN><<<gridSize, blockSize>>>(
      c, n_intervals, depth, feat, ranks_depth, ranks_feat,
      interval_starts, interval_lengths, out);
}

extern "C"
void bev_pool_bf16_bf16_bf16(int c, int n_intervals,
                             const __nv_bfloat16 *depth,
                             const __nv_bfloat16 *feat,
                             const int *ranks_depth,
                             const int *ranks_feat,
                             const int *interval_starts,
                             const int *interval_lengths,
                             __nv_bfloat16 *out) {
  dim3 gridSize((c + TC * BC - 1)/(TC * BC), (n_intervals + TN * BN - 1)/(TN * BN));
  dim3 blockSize(BC, BN);
  bev_pool_kernel<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, TC, TN><<<gridSize, blockSize>>>(
      c, n_intervals, depth, feat, ranks_depth, ranks_feat,
      interval_starts, interval_lengths, out);
}

extern "C"
void bev_pool_bf16_float_float(int c, int n_intervals,
                               const __nv_bfloat16 *depth,
                               const __nv_bfloat16 *feat,
                               const int *ranks_depth,
                               const int *ranks_feat,
                               const int *interval_starts,
                               const int *interval_lengths,
                               float *out) {
  dim3 gridSize((c + TC * BC - 1)/(TC * BC), (n_intervals + TN * BN - 1)/(TN * BN));
  dim3 blockSize(BC, BN);
  bev_pool_kernel<__nv_bfloat16, float, float, TC, TN><<<gridSize, blockSize>>>(
      c, n_intervals, depth, feat, ranks_depth, ranks_feat,
      interval_starts, interval_lengths, out);
}

extern "C"
void bev_pool_half_half_half(int c, int n_intervals,
                        const __half *depth,
                        const __half *feat,
                        const int *ranks_depth,
                        const int *ranks_feat,
                        const int *interval_starts,
                        const int *interval_lengths,
                        __half *out) {
  dim3 gridSize((c + TC * BC - 1)/(TC * BC), (n_intervals + TN * BN - 1)/(TN * BN));
  dim3 blockSize(BC, BN);
  bev_pool_kernel<__half, __half, __half, TC, TN><<<gridSize, blockSize>>>(
      c, n_intervals, depth, feat, ranks_depth, ranks_feat,
      interval_starts, interval_lengths, out);
}

template<typename DType, typename FType, typename OType, const int TC, const int TN>
__global__ void bev_pool_kernel_v2(
    int C, int n_intervals,
    const DType *__restrict__ depth,
    const FType *__restrict__ feat,
    const int *__restrict__ ranks_depth,
    const int *__restrict__ ranks_feat,
    const int *__restrict__ interval_starts,
    const int *__restrict__ interval_lengths,
    OType *__restrict__ out) {

  int tc_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tn_idx = blockIdx.y * blockDim.y + threadIdx.y;

#pragma unroll
  for (int tn = 0; tn < TN; tn++) {
    float psum[TC];
    int n_idx = tn_idx * TN + tn;
    if (n_idx >= n_intervals) break;

    int interval_start = __ldg(&interval_starts[n_idx]);
    int interval_length = __ldg(&interval_lengths[n_idx]);

    if (interval_start == -1) break;

    for (int tc = 0; tc < TC; tc++) {
      psum[tc] = 0;
    }

    for (int i = 0; i < interval_length; i++) {
      float d = (float)__ldg(&depth[ranks_depth[interval_start + i]]);
#pragma unroll
      for (int tc = 0; tc < TC; tc++) {
        int c_idx = tc_idx * TC + tc;
        if (c_idx >= C) continue;

        float f = (float)__ldg(&feat[ranks_feat[interval_start + i] * C + c_idx]);
        psum[tc] = __fmaf_rn(d, f, psum[tc]);
      }
    }

#pragma unroll
    for (int tc = 0; tc < TC; tc++) {
      int c_idx = tc_idx * TC + tc;
      if (c_idx >= C) break;
      int tid;
      tid = n_idx * C + c_idx;
      out[tid] = psum[tc];
    }
  }
}

extern "C"
void bev_pool_v2_float_float_float(int c, int n_intervals,
                                   const float *depth,
                                   const float *feat,
                                   const int *ranks_depth,
                                   const int *ranks_feat,
                                   const int *interval_starts,
                                   const int *interval_lengths,
                                   float *out) {
    dim3 gridSize((c + TC * BC - 1)/(TC * BC), (n_intervals + TN * BN - 1)/(TN * BN));
    dim3 blockSize(BC, BN);
    bev_pool_kernel_v2<float, float, float, TC, TN><<<gridSize, blockSize>>>(
        c, n_intervals, depth, feat, ranks_depth, ranks_feat,
        interval_starts, interval_lengths, out);
}

extern "C"
void bev_pool_v2_float_half_half(int c, int n_intervals,
                                 const float *depth,
                                 const __half *feat,
                                 const int *ranks_depth,
                                 const int *ranks_feat,
                                 const int *interval_starts,
                                 const int *interval_lengths,
                                 __half *out) {
    dim3 gridSize((c + TC * BC - 1)/(TC * BC), (n_intervals + TN * BN - 1)/(TN * BN));
    dim3 blockSize(BC, BN);
    bev_pool_kernel_v2<float, __half, __half, TC, TN><<<gridSize, blockSize>>>(
        c, n_intervals, depth, feat, ranks_depth, ranks_feat,
        interval_starts, interval_lengths, out);
}

extern "C"
void bev_pool_v2_float_half_float(int c, int n_intervals,
                                 const float *depth,
                                 const __half *feat,
                                 const int *ranks_depth,
                                 const int *ranks_feat,
                                 const int *interval_starts,
                                 const int *interval_lengths,
                                 float *out) {
    dim3 gridSize((c + TC * BC - 1)/(TC * BC), (n_intervals + TN * BN - 1)/(TN * BN));
    dim3 blockSize(BC, BN);
    bev_pool_kernel_v2<float, __half, float, TC, TN><<<gridSize, blockSize>>>(
        c, n_intervals, depth, feat, ranks_depth, ranks_feat,
        interval_starts, interval_lengths, out);
}

extern "C"
void bev_pool_v2_half_half_float(int c, int n_intervals,
                                  const __half *depth,
                                  const __half *feat,
                                  const int *ranks_depth,
                                  const int *ranks_feat,
                                  const int *interval_starts,
                                  const int *interval_lengths,
                                  float *out) {
    dim3 gridSize((c + TC * BC - 1)/(TC * BC), (n_intervals + TN * BN - 1)/(TN * BN));
    dim3 blockSize(BC, BN);
    bev_pool_kernel_v2<__half, __half, float, TC, TN><<<gridSize, blockSize>>>(
        c, n_intervals, depth, feat, ranks_depth, ranks_feat,
        interval_starts, interval_lengths, out);
}

extern "C"
void bev_pool_v2_half_float_float(int c, int n_intervals,
                                  const __half *depth,
                                  const float *feat,
                                  const int *ranks_depth,
                                  const int *ranks_feat,
                                  const int *interval_starts,
                                  const int *interval_lengths,
                                  float *out) {
    dim3 gridSize((c + TC * BC - 1)/(TC * BC), (n_intervals + TN * BN - 1)/(TN * BN));
    dim3 blockSize(BC, BN);
    bev_pool_kernel_v2<__half, float, float, TC, TN><<<gridSize, blockSize>>>(
        c, n_intervals, depth, feat, ranks_depth, ranks_feat,
        interval_starts, interval_lengths, out);
}

