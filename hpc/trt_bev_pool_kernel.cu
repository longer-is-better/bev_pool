#include <stdio.h>
#include <stdlib.h>
#include "bev_pool.h"
// #include "trt_bev_pool_kernel.hpp"
/*
  Function: pillar pooling
  Args:
    c                : number of channels
    n_intervals      : number of unique points
    depth            : input depth, FloatTensor[b,n,d,h,w]
    feat             : input feat, FloatTensor[b,n,h,w,c]
    ranks_depth      : input index of depth, IntTensor[n_points]
    ranks_feat       : input index of feat, IntTensor[n_points]
    ranks_bev        : output index, IntTensor[n_points]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
    out              : output features, FloatTensor[b, z, h, w, c]
*/
__global__ void bev_pool_v2_kernel(int c, int n_intervals,
                                  int n_valid_points, int n_out_grid_points,
                                  int n_total_depth_score, int n_total_img_feat,
                                  const float *__restrict__ depth,
                                  const float *__restrict__ feat,
                                  const int *__restrict__ ranks_depth,
                                  const int *__restrict__ ranks_feat,
                                  const int *__restrict__ ranks_bev,
                                  const int *__restrict__ interval_starts,  // 155989
                                  const int *__restrict__ interval_lengths,
                                  float* __restrict__ out) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = idx / c;
  int cur_c = idx % c;

  if (index >= n_intervals) return;
  int interval_start = interval_starts[index];
  int interval_length = interval_lengths[index];
  // printf("interval_start:%d",interval_start);
  // printf("interval_length:%d",interval_length);
  if (interval_start < 0) return;
  if (interval_start >= n_valid_points) return;
  if (interval_length < 0) return;
  if (interval_length > (n_valid_points - interval_start)) return;

  float psum = 0;
  const float* cur_depth;
  const float* cur_feat;
#pragma unroll
  for(int i = 0; i < interval_length; i++){
    if (interval_start+i < 0) continue;
    if (interval_start+i >= n_valid_points) continue;
    if (ranks_depth[interval_start+i] < 0) continue;
    if (ranks_feat[interval_start+i] < 0) continue;
    if (ranks_depth[interval_start+i] >= n_total_depth_score) continue;
    if (ranks_feat[interval_start+i] >= n_total_img_feat) continue;
    cur_depth = depth + ranks_depth[interval_start+i];
    cur_feat = feat + ranks_feat[interval_start+i] * c + cur_c;
    psum += *cur_feat * *cur_depth;
  }

  const int* cur_rank = ranks_bev + interval_start;
  if (*cur_rank < 0) return;
  if (*cur_rank * c + cur_c < 0) return;
  if (*cur_rank * c + cur_c >= n_out_grid_points) return;
  float* cur_out = out + *cur_rank * c + cur_c;
  // printf("pusm: %.f\n",psum);
  *cur_out = psum;
}

// __global__ void bev_pool_v2_set_zero_kernel(int n_points, float* __restrict__ out) {
//   int idx = blockIdx.x * blockDim.x + threadIdx.x;
//   if (idx >= n_points) return;
//   float* cur_out = out + idx;
//   *cur_out = 0.0;
// }

void bev_pool_v2(int c, int n_intervals,  int n_valid_points, int n_out_grid_points, int n_total_depth_score, int n_total_img_feat,
  const float* depth, const float* feat, const int* ranks_depth,
  const int* ranks_feat, const int* ranks_bev, const int* interval_starts, const int* interval_lengths, float* out,
  cudaStream_t stream) {
  cudaMemsetAsync((float *)out, 0, n_out_grid_points * sizeof(float), stream);
  bev_pool_v2_kernel<<<(int)ceil(((double)n_intervals * c / 1024)), 1024, 0, stream>>>(
    c, n_intervals, n_valid_points, n_out_grid_points, n_total_depth_score, n_total_img_feat, depth, feat, ranks_depth, ranks_feat,
    ranks_bev, interval_starts, interval_lengths, out
  );
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
      printf("error in bev_pool_v2: %s\n" ,cudaGetErrorString(err));
  }
}


// void bev_pool_v2_set_zero(int n_points, float* out, cudaStream_t stream) {
//   bev_pool_v2_set_zero_kernel<<<(int)ceil(((double)n_points / 256)), 256, 0, stream>>>(n_points, out);
// }

