
#include <stdio.h>
#include <stdlib.h>
#include "cuda_fp16.h"

extern "C"
void tensor_init(int *ranks_depth,
                 int *ranks_feat,
                 int *ranks_bev,
                 int *interval_starts,
                 int *interval_lengths,
                 int N, int K);
void tensor_init(int *ranks_depth,
                 int *ranks_feat,
                 int *ranks_bev,
                 int *interval_starts,
                 int *interval_lengths,
                 int N, int K) {

#pragma omp parallel for collapse(2)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < K; j++) {
      interval_starts[i] = i * K;
      interval_lengths[i] = K;
      ranks_bev[i * K + j] = i;
      ranks_feat[i * K+ j] = i + j / 2;
      ranks_depth[i * K + j] = i * 120 + j / 2;
    }
  }
}

extern "C" void bev_pool_baseline(int c, int n_intervals, const float *depth, const float *feat,
                                  const int *ranks_depth, const int *ranks_feat,
                                  const int *ranks_bev, const int *interval_starts,
                                  const int *interval_lengths, float *out);
__global__ void bev_pool_baseline_kernel(
    int c, int n_intervals, const float *__restrict__ depth,
    const float *__restrict__ feat, const int *__restrict__ ranks_depth,
    const int *__restrict__ ranks_feat, const int *__restrict__ ranks_bev,
    const int *__restrict__ interval_starts,
    const int *__restrict__ interval_lengths, float *__restrict__ out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = idx / c;
  int cur_c = idx % c;
  if (index >= n_intervals)
    return;
  int interval_start = interval_starts[index];
  int interval_length = interval_lengths[index];
  float psum = 0;
  const float *cur_depth;
  const float *cur_feat;
  for (int i = 0; i < interval_length; i++) {
    cur_depth = depth + ranks_depth[interval_start + i];
    cur_feat = feat + ranks_feat[interval_start + i] * c + cur_c;
    psum += *cur_feat * *cur_depth;
  }

  const int *cur_rank = ranks_bev + interval_start;
  float *cur_out = out + *cur_rank * c + cur_c;
  *cur_out = psum;
}

void bev_pool_baseline(int c, int n_intervals, const float *depth, const float *feat,
                       const int *ranks_depth, const int *ranks_feat,
                       const int *ranks_bev, const int *interval_starts,
                       const int *interval_lengths, float *out) {
  bev_pool_baseline_kernel<<<(int)ceil(((double)n_intervals * c / 256)), 256>>>(
      c, n_intervals, depth, feat, ranks_depth, ranks_feat, ranks_bev,
      interval_starts, interval_lengths, out);
}

template<typename TensorType, typename AccType, const int TC, const int TN>
__global__ void bev_pool_kernel(
    int c, int n_intervals,
    const TensorType *__restrict__ depth,
    const TensorType *__restrict__ feat,
    const int *__restrict__ ranks_depth,
    const int *__restrict__ ranks_feat,
    const int *__restrict__ ranks_bev,
    const int *__restrict__ interval_starts,
    const int *__restrict__ interval_lengths,
    TensorType *__restrict__ out) {

  int tc_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tn_idx = blockIdx.y * blockDim.y + threadIdx.y;

#pragma unroll
  for (int tn = 0; tn < TN; tn++) {
    AccType psum[TC];
    int n_idx = tn_idx * TN + tn;
    if (n_idx >= n_intervals) return;

    int interval_start = interval_starts[n_idx];
    int interval_length = interval_lengths[n_idx];

    for (int tc = 0; tc < TC; tc++) {
      psum[tc] = 0;
    }

    for (int i = 0; i < interval_length; i++) {
      TensorType d = depth[ranks_depth[interval_start + i]];
#pragma unroll
      for (int tc = 0; tc < TC; tc++) { 
        int c_idx = tc_idx * TC + tc;
        if (c_idx >= c) continue;

        TensorType f = feat[ranks_feat[interval_start + i] * c + c_idx];
        if (std::is_same<TensorType, __half>::value && std::is_same<AccType, __half>::value)
          psum[tc] = __hfma(d, f, psum[tc]);
        else if (std::is_same<TensorType, __half>::value && std::is_same<AccType, float>::value)
          psum[tc] = __fmaf_rn(__half2float(d), __half2float(f), psum[tc]);
        else // (std::is_same<TensorType, float>::value && std::is_same<AccType, float>::value)
          psum[tc] = __fmaf_rn(d, f, psum[tc]);
      }
    }

#pragma unroll
    for (int tc = 0; tc < TC; tc++) {
      int c_idx = tc_idx * TC + tc;
      if (c_idx >= c) continue;
      if (std::is_same<TensorType, __half>::value && std::is_same<AccType, float>::value)
        out[ranks_bev[interval_start] * c + c_idx] = __float2half(psum[tc]);
      else
        out[ranks_bev[interval_start] * c + c_idx] = psum[tc];
    }

  }
}


extern "C"
void bev_pool_float_float_1_1(int c, int n_intervals, const float *depth,
                              const float *feat,
                              const int *ranks_depth,
                              const int *ranks_feat,
                              const int *ranks_bev,
                              const int *interval_starts,
                              const int *interval_lengths,
                              float *out);

void bev_pool_float_float_1_1(int c, int n_intervals,
                              const float *depth,
                              const float *feat,
                              const int *ranks_depth,
                              const int *ranks_feat,
                              const int *ranks_bev,
                              const int *interval_starts,
                              const int *interval_lengths,
                              float *out) {
  constexpr int TC = 1;
  constexpr int TN = 1;
  constexpr int BC = 32;
  constexpr int BN = 8;
  dim3 gridSize((c + TC * BC - 1)/(TC * BC), (n_intervals + TN * BN - 1)/(TN * BN));
  dim3 blockSize(BC, BN);
  bev_pool_kernel<float, float, TC, TN><<<gridSize, blockSize>>>(
      c, n_intervals, depth, feat, ranks_depth, ranks_feat, ranks_bev,
      interval_starts, interval_lengths, out);
}

extern "C"
void bev_pool_float_float_2_2(int c, int n_intervals, const float *depth,
                              const float *feat,
                              const int *ranks_depth,
                              const int *ranks_feat,
                              const int *ranks_bev,
                              const int *interval_starts,
                              const int *interval_lengths,
                              float *out);

void bev_pool_float_float_2_2(int c, int n_intervals,
                              const float *depth,
                              const float *feat,
                              const int *ranks_depth,
                              const int *ranks_feat,
                              const int *ranks_bev,
                              const int *interval_starts,
                              const int *interval_lengths,
                              float *out) {
  constexpr int TC = 2;
  constexpr int TN = 2;
  constexpr int BC = 32;
  constexpr int BN = 8;
  dim3 gridSize((c + TC * BC - 1)/(TC * BC), (n_intervals + TN * BN - 1)/(TN * BN));
  dim3 blockSize(BC, BN);
  bev_pool_kernel<float, float, TC, TN><<<gridSize, blockSize>>>(
      c, n_intervals, depth, feat, ranks_depth, ranks_feat, ranks_bev,
      interval_starts, interval_lengths, out);
}

extern "C"
void bev_pool_half_float_2_2(int c, int n_intervals,
                             const __half *depth,
                             const __half *feat,
                             const int *ranks_depth,
                             const int *ranks_feat,
                             const int *ranks_bev,
                             const int *interval_starts,
                             const int *interval_lengths,
                             __half *out);

void bev_pool_half_float_2_2(int c, int n_intervals,
                             const __half *depth,
                             const __half *feat,
                             const int *ranks_depth,
                             const int *ranks_feat,
                             const int *ranks_bev,
                             const int *interval_starts,
                             const int *interval_lengths,
                             __half *out) {
  constexpr int TC = 2;
  constexpr int TN = 2;
  constexpr int BC = 32;
  constexpr int BN = 8;
  dim3 gridSize((c + TC * BC - 1)/(TC * BC), (n_intervals + TN * BN - 1)/(TN * BN));
  dim3 blockSize(BC, BN);
  bev_pool_kernel<__half, float, TC, TN><<<gridSize, blockSize>>>(
      c, n_intervals, depth, feat, ranks_depth, ranks_feat, ranks_bev,
      interval_starts, interval_lengths, out);
}

extern "C"
void bev_pool_half_half_2_2(int c, int n_intervals,
                            const __half *depth,
                            const __half *feat,
                            const int *ranks_depth,
                            const int *ranks_feat,
                            const int *ranks_bev,
                            const int *interval_starts,
                            const int *interval_lengths,
                            __half *out);

void bev_pool_half_half_2_2(int c, int n_intervals,
                            const __half *depth,
                            const __half *feat,
                            const int *ranks_depth,
                            const int *ranks_feat,
                            const int *ranks_bev,
                            const int *interval_starts,
                            const int *interval_lengths,
                            __half *out) {
  constexpr int TC = 2;
  constexpr int TN = 2;
  constexpr int BC = 32;
  constexpr int BN = 8;
  dim3 gridSize((c + TC * BC - 1)/(TC * BC), (n_intervals + TN * BN - 1)/(TN * BN));
  dim3 blockSize(BC, BN);
  bev_pool_kernel<__half, __half, TC, TN><<<gridSize, blockSize>>>(
      c, n_intervals, depth, feat, ranks_depth, ranks_feat, ranks_bev,
      interval_starts, interval_lengths, out);
}
