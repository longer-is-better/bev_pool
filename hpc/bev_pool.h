#ifndef __BEV_POOL_H__
#define __BEV_POOL_H__
#include <cuda_fp16.h>
#include <cuda_runtime.h>
extern "C" void bev_pool_v2(int c, int n_intervals,  int n_valid_points, int n_out_grid_points, int n_total_depth_score, int n_total_img_feat,
  const float* depth, const float* feat, const int* ranks_depth,
  const int* ranks_feat, const int* ranks_bev, const int* interval_starts, const int* interval_lengths, float* out,
  cudaStream_t stream);

extern "C" void bev_pool_pack16(int c, int n_intervals,  int n_valid_points, int n_out_grid_points, int n_total_depth_score, int n_total_img_feat,
  const float* depth, const float* feat, const int* ranks_depth,
  const int* ranks_feat, const int* ranks_bev, const int* interval_starts, const int* interval_lengths, float* out,
  cudaStream_t stream);


extern "C" void bev_pool_pack32_half(int c, int n_intervals,  int n_valid_points, int n_out_grid_points, int n_total_depth_score, int n_total_img_feat,
  const half* depth, const half* feat, const int* ranks_depth,
  const int* ranks_feat, const int* ranks_bev, const int* interval_starts, const int* interval_lengths, half* out,
  cudaStream_t stream);

// void bev_pool_v2(int c, int n_intervals,  int n_valid_points, int n_out_grid_points, int n_total_depth_score, int n_total_img_feat,
//   const __half* depth, const __half2* feat, const int* ranks_depth,
//   const int* ranks_feat, const int* ranks_bev, const int* interval_starts, const int* interval_lengths, __half2* out,
//   cudaStream_t stream);

// void bev_pool_v3(int c, int n_intervals,  int n_valid_points, int n_out_grid_points, int n_total_depth_score, int n_total_img_feat,
//   const half* depth, const half* feat, const int* ranks_depth,
//   const int* ranks_feat, const int* ranks_bev, const int* interval_starts, const int* interval_lengths,half* out,
//   cudaStream_t stream);
// void bev_pool_v22(int c, int n_intervals,  int n_valid_points, int n_out_grid_points, int n_total_depth_score, int n_total_img_feat,
//   const float* depth, const float* feat, const int* ranks_depth,
//   const int* ranks_feat, const int* ranks_bev, const int* interval_starts, const int* interval_lengths, float* out,
//   cudaStream_t stream);


// void bev_pool_v22(int c, int n_intervals,  int n_valid_points, int n_out_grid_points, int n_total_depth_score, int n_total_img_feat,
//   const half* depth, const half* feat, const int* ranks_depth,
//   const int* ranks_feat, const int* ranks_bev, const int* interval_starts, const int* interval_lengths, half* out,
//   cudaStream_t stream);
#endif

