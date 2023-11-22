#include <stdio.h>
#include <stdlib.h>
#include <cuda_fp16.h>
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
#define TILE_SIZE 4
typedef struct {
  unsigned int val[TILE_SIZE/2];
} combined_half;
__global__ void bev_pool_pack32_kernel_half(int c, int n_intervals,
                                  int n_valid_points, int n_out_grid_points,
                                  int n_total_depth_score, int n_total_img_feat,
                                  const half *__restrict__ depth,
                                  const half *__restrict__ feat,
                                  const int *__restrict__ ranks_depth,
                                  const int *__restrict__ ranks_feat,
                                  const int *__restrict__ ranks_bev,
                                  const int *__restrict__ interval_starts,  // 155989
                                  const int *__restrict__ interval_lengths,
                                  half* __restrict__ out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = idx / c;
  int cur_c = (idx % c)*TILE_SIZE;
  if (index >= n_intervals) return;
  int interval_start = interval_starts[index];
  int interval_length = interval_lengths[index];
  if (interval_start < 0) return;
  if (interval_start >= n_valid_points) return;
  if (interval_length < 0) return;
  if (interval_length > (n_valid_points - interval_start)) return;
  half psum[TILE_SIZE]={0};
  const half* cur_depth;
  // const combined_half cur_feat;
  int interval_index = 0;
  int depth_index = 0;
  int feat_index = 0;
#pragma unroll
  for(int i = 0; i < interval_length; i++){
    interval_index = interval_start+i;
    if (interval_index < 0) continue;
    if (interval_index >= n_valid_points) continue;
    depth_index= __ldg(&ranks_depth[interval_index]);
    feat_index= __ldg(&ranks_feat[interval_index]);
    if (depth_index < 0) continue;
    if (feat_index< 0) continue;
    if (depth_index >= n_total_depth_score) continue;
    if (feat_index >= n_total_img_feat) continue;
    cur_depth = depth + depth_index;
    const combined_half cur_feat = *(combined_half*)(feat + feat_index * c*TILE_SIZE + cur_c);
    for(int j=0;j<TILE_SIZE;j++){
      psum[j]=__hfma(((half*)&cur_feat)[j], *cur_depth,psum[j]);
      // psum[j]=__fmaf_rn(((half*)&cur_feat[0])[i], *cur_depth,psum[j]);
      // psum[j] += ((float*)&cur_feat[0])[i] * *cur_depth;
    }
    // printf("PSUM: %.6f\n",psum[0]);
  }

  const int* cur_rank = ranks_bev + interval_start;
  if (*cur_rank < 0) return;
  if (*cur_rank * c*TILE_SIZE + cur_c < 0) return;
  if (*cur_rank * c*TILE_SIZE + cur_c >= n_out_grid_points) return;
  half* cur_out = out + *cur_rank * c*TILE_SIZE + cur_c;
  // combined_half* cur_out = (combined_half*)(out + *cur_rank * c*TILE_SIZE + cur_c);
  for(int j=0;j<TILE_SIZE;j++){
    // __stcg(&cur_out[j], psum[j]);
    cur_out[j] = psum[j];
    // ((half*)&cur_out[0])[j] = psum[j];
  }
}

// __global__ void bev_pool_v2_set_zero_kernel(int n_points, float* __restrict__ out) {
//   int idx = blockIdx.x * blockDim.x + threadIdx.x;
//   if (idx >= n_points) return;
//   float* cur_out = out + idx;
//   *cur_out = 0.0;
// }

void bev_pool_pack32_half(int c, int n_intervals,  int n_valid_points, int n_out_grid_points, int n_total_depth_score, int n_total_img_feat,
  const half* depth, const half* feat, const int* ranks_depth,
  const int* ranks_feat, const int* ranks_bev, const int* interval_starts, const int* interval_lengths, half* out,
  cudaStream_t stream) {
  cudaMemset((half *)out, 0, n_out_grid_points * sizeof(half));
  bev_pool_pack32_kernel_half<<<(int)ceil(((double)n_intervals * (c/TILE_SIZE) / 256)), 256, 0, stream>>>(
    c/TILE_SIZE, n_intervals, n_valid_points, n_out_grid_points, n_total_depth_score, n_total_img_feat, depth, feat, ranks_depth, ranks_feat,
    ranks_bev, interval_starts, interval_lengths, out
  );
}


// void bev_pool_v2_set_zero(int n_points, float* out, cudaStream_t stream) {
//   bev_pool_v2_set_zero_kernel<<<(int)ceil(((double)n_points / 256)), 256, 0, stream>>>(n_points, out);
// }

