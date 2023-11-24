
#include <stdio.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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

  {FILE *fp = fopen("data/ranks_depth.bin", "rb"); size_t num = fread(ranks_depth, sizeof(int), 4000000, fp); if (num != 4000000) { printf("read error: ranks_depth\n");}; fclose(fp);}
  {FILE *fp = fopen("data/ranks_feat.bin", "rb"); size_t num = fread(ranks_feat, sizeof(int), 4000000, fp); if (num != 4000000) { printf("read error: ranks_feat\n");}; fclose(fp);}
  {FILE *fp = fopen("data/ranks_bev.bin", "rb"); size_t num = fread(ranks_bev, sizeof(int), 4000000, fp); if (num != 4000000) { printf("read error: ranks_bev\n");}; fclose(fp);}
  {FILE *fp = fopen("data/interval_starts.bin", "rb"); size_t num = fread(interval_starts, sizeof(int), 50000, fp); if (num != 50000) { printf("read error: interval_starts\n");}; fclose(fp);}
  {FILE *fp = fopen("data/interval_lengths.bin", "rb"); size_t num = fread(interval_lengths, sizeof(int), 50000, fp); if (num != 50000) { printf("read error: interval_lengths\n");}; fclose(fp);}

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

extern "C" void bev_pool_flatmap(int c, int n_intervals, const float *depth, const float *feat,
                                 const int *ranks_depth, const int *ranks_feat,
                                 const int *ranks_bev, const int *interval_starts,
                                 const int *interval_lengths, float *out);
void bev_pool_flatmap(int C, int n_intervals, const float *depth, const float *feat,
                      const int *ranks_depth, const int *ranks_feat,
                      const int *ranks_bev, const int *interval_starts,
                      const int *interval_lengths, float *out) {

  constexpr int N = 2487077;
  constexpr int TC = 1;
  constexpr int TN = 90;
  constexpr int BC = 128;
  constexpr int BN = 2;
  dim3 gridSize((C + TC * BC - 1)/(TC * BC), (N + TN * BN - 1)/(TN * BN));
  dim3 blockSize(BC, BN);

  cudaMemset(out, 0, 192*256*C*sizeof(float));
  bev_pool_flatmap_kernel<float, float, TC, TN><<<gridSize, blockSize>>>(
      C, N, depth, feat, ranks_depth, ranks_feat, ranks_bev,
      interval_starts, interval_lengths, out);
}

template<typename TensorType, typename AccType, const int TC, const int TN, const int IHW=1, const int OHW=1, bool FEAT_CL=true>
__global__ void bev_pool_kernel(
    int C, int n_intervals,
    const TensorType *__restrict__ depth,
    const TensorType *__restrict__ feat,
    const int *__restrict__ ranks_depth,
    const int *__restrict__ ranks_feat,
    const int *__restrict__ interval_starts,
    const int *__restrict__ interval_lengths,
    TensorType *__restrict__ out) {

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
      TensorType d = __ldg(&depth[ranks_depth[interval_start + i]]);
#pragma unroll
      for (int tc = 0; tc < TC; tc++) {
        int c_idx = tc_idx * TC + tc;
        if (c_idx >= C) continue;

        TensorType f;
        if (FEAT_CL)
          f = __ldg(&feat[ranks_feat[interval_start + i] * C + c_idx]);
        else {// NCHW
          int idx = ranks_feat[interval_start + i];
          int n = idx / IHW;
          int hw = idx % IHW;
          f = __ldg(&feat[n*C*IHW + c_idx*IHW + hw]);
        }

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
      if (c_idx >= C) break;
      int tid;
      if (FEAT_CL)
        tid = n_idx * C + c_idx;
      else {
        int n = n_idx / OHW;
        int hw = n_idx % OHW;
        tid = n*C*OHW + c_idx*OHW + hw;
      }
      if (std::is_same<TensorType, __half>::value && std::is_same<AccType, float>::value)
        out[tid] = __float2half(psum[tc]);
      else
        out[tid] = psum[tc];
    }

  }
}


extern "C"
void bev_pool_float_float(int c, int n_intervals, const float *depth,
                          const float *feat,
                          const int *ranks_depth,
                          const int *ranks_feat,
                          const int *interval_starts,
                          const int *interval_lengths,
                          float *out);

void bev_pool_float_float(int c, int n_intervals,
                          const float *depth,
                          const float *feat,
                          const int *ranks_depth,
                          const int *ranks_feat,
                          const int *interval_starts,
                          const int *interval_lengths,
                          float *out) {
  constexpr int TC = 2;
  constexpr int TN = 1;
  constexpr int BC = 64;
  constexpr int BN = 8;
  dim3 gridSize((c + TC * BC - 1)/(TC * BC), (n_intervals + TN * BN - 1)/(TN * BN));
  dim3 blockSize(BC, BN);
  bev_pool_kernel<float, float, TC, TN><<<gridSize, blockSize>>>(
      c, n_intervals, depth, feat, ranks_depth, ranks_feat,
      interval_starts, interval_lengths, out);
}

extern "C"
void bev_pool_float_float_nchw(int c, int n_intervals, const float *depth,
                          const float *feat,
                          const int *ranks_depth,
                          const int *ranks_feat,
                          const int *interval_starts,
                          const int *interval_lengths,
                          float *out);

void bev_pool_float_float_nchw(int c, int n_intervals,
                          const float *depth,
                          const float *feat,
                          const int *ranks_depth,
                          const int *ranks_feat,
                          const int *interval_starts,
                          const int *interval_lengths,
                          float *out) {
  constexpr int TC = 2;
  constexpr int TN = 1;
  constexpr int BC = 64;
  constexpr int BN = 8;
  dim3 gridSize((c + TC * BC - 1)/(TC * BC), (n_intervals + TN * BN - 1)/(TN * BN));
  dim3 blockSize(BC, BN);
  bev_pool_kernel<float, float, TC, TN, 64*120, 192*256, false><<<gridSize, blockSize>>>(
      c, n_intervals, depth, feat, ranks_depth, ranks_feat,
      interval_starts, interval_lengths, out);
}

extern "C"
void bev_pool_half_float(int c, int n_intervals,
                         const __half *depth,
                         const __half *feat,
                         const int *ranks_depth,
                         const int *ranks_feat,
                         const int *interval_starts,
                         const int *interval_lengths,
                         __half *out);

void bev_pool_half_float(int c, int n_intervals,
                         const __half *depth,
                         const __half *feat,
                         const int *ranks_depth,
                         const int *ranks_feat,
                         const int *interval_starts,
                         const int *interval_lengths,
                         __half *out) {
  constexpr int TC = 2;
  constexpr int TN = 1;
  constexpr int BC = 64;
  constexpr int BN = 8;
  dim3 gridSize((c + TC * BC - 1)/(TC * BC), (n_intervals + TN * BN - 1)/(TN * BN));
  dim3 blockSize(BC, BN);
  bev_pool_kernel<__half, float, TC, TN><<<gridSize, blockSize>>>(
      c, n_intervals, depth, feat, ranks_depth, ranks_feat,
      interval_starts, interval_lengths, out);
}

extern "C"
void bev_pool_half_float_nchw(int c, int n_intervals,
                              const __half *depth,
                              const __half *feat,
                              const int *ranks_depth,
                              const int *ranks_feat,
                              const int *interval_starts,
                              const int *interval_lengths,
                              __half *out);

void bev_pool_half_float_nchw(int c, int n_intervals,
                              const __half *depth,
                              const __half *feat,
                              const int *ranks_depth,
                              const int *ranks_feat,
                              const int *interval_starts,
                              const int *interval_lengths,
                              __half *out) {
  constexpr int TC = 2;
  constexpr int TN = 1;
  constexpr int BC = 64;
  constexpr int BN = 8;
  dim3 gridSize((c + TC * BC - 1)/(TC * BC), (n_intervals + TN * BN - 1)/(TN * BN));
  dim3 blockSize(BC, BN);
  bev_pool_kernel<__half, float, TC, TN, 64*120, 192*256, false><<<gridSize, blockSize>>>(
      c, n_intervals, depth, feat, ranks_depth, ranks_feat,
      interval_starts, interval_lengths, out);
}


extern "C"
void bev_pool_half_half(int c, int n_intervals,
                        const __half *depth,
                        const __half *feat,
                        const int *ranks_depth,
                        const int *ranks_feat,
                        const int *interval_starts,
                        const int *interval_lengths,
                        __half *out);

void bev_pool_half_half(int c, int n_intervals,
                        const __half *depth,
                        const __half *feat,
                        const int *ranks_depth,
                        const int *ranks_feat,
                        const int *interval_starts,
                        const int *interval_lengths,
                        __half *out) {
  constexpr int TC = 2;
  constexpr int TN = 1;
  constexpr int BC = 64;
  constexpr int BN = 8;
  dim3 gridSize((c + TC * BC - 1)/(TC * BC), (n_intervals + TN * BN - 1)/(TN * BN));
  dim3 blockSize(BC, BN);
  bev_pool_kernel<__half, __half, TC, TN><<<gridSize, blockSize>>>(
      c, n_intervals, depth, feat, ranks_depth, ranks_feat,
      interval_starts, interval_lengths, out);
}

extern "C"
void bev_pool_half_half_nchw(int c, int n_intervals,
                             const __half *depth,
                             const __half *feat,
                             const int *ranks_depth,
                             const int *ranks_feat,
                             const int *interval_starts,
                             const int *interval_lengths,
                             __half *out);

void bev_pool_half_half_nchw(int c, int n_intervals,
                             const __half *depth,
                             const __half *feat,
                             const int *ranks_depth,
                             const int *ranks_feat,
                             const int *interval_starts,
                             const int *interval_lengths,
                             __half *out) {
  constexpr int TC = 1;
  constexpr int TN = 4;
  constexpr int BC = 64;
  constexpr int BN = 8;
  dim3 gridSize((c + TC * BC - 1)/(TC * BC), (n_intervals + TN * BN - 1)/(TN * BN));
  dim3 blockSize(BC, BN);
  bev_pool_kernel<__half, __half, TC, TN, 64*120, 192*256, false><<<gridSize, blockSize>>>(
      c, n_intervals, depth, feat, ranks_depth, ranks_feat,
      interval_starts, interval_lengths, out);
}
