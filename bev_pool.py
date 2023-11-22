import ctypes
import time
import torch

diff = True

E = ctypes.cdll.LoadLibrary("./bev_pool.so")
H = ctypes.cdll.LoadLibrary("hpc/b/libbev_pool_shared.so")

depth = torch.randn((7, 120, 64, 120), dtype=torch.float)
feat = torch.randn((7, 64, 120, 128), dtype=torch.float)
bev = torch.ones(size=[1, 192, 256, 128], dtype=torch.float)

bev_baseline_float = torch.zeros(size=[1, 192, 256, 128], dtype=torch.float, device='cuda')
bev_baseline_half = torch.zeros(size=[1, 192, 256, 128], dtype=torch.half, device='cuda')

ranks_depth = torch.zeros((4000000), dtype=torch.int32)
ranks_feat = torch.zeros((4000000), dtype=torch.int32)
ranks_bev = torch.zeros((4000000), dtype=torch.int32)
interval_starts = torch.zeros((50000), dtype=torch.int32)
interval_lengths = torch.zeros((50000), dtype=torch.int32)
interval_starts_e = torch.zeros((50000), dtype=torch.int32)
interval_lengths_e = torch.zeros((50000), dtype=torch.int32)



n_intervals = 192 * 256 #48495
ranks_bev_mask = torch.zeros((192*256), dtype=torch.int8)

E.tensor_init(ctypes.c_void_p(ranks_depth.data_ptr()),
              ctypes.c_void_p(ranks_feat.data_ptr()),
              ctypes.c_void_p(ranks_bev.data_ptr()),
              ctypes.c_void_p(interval_starts.data_ptr()),
              ctypes.c_void_p(interval_lengths.data_ptr()),
              ctypes.c_void_p(ranks_bev_mask.data_ptr()),
              ctypes.c_void_p(interval_starts_e.data_ptr()),
              ctypes.c_void_p(interval_lengths_e.data_ptr()))


def test_baseline_fp32_old():
    depth_local = depth.cuda()
    feat_local = feat.cuda()
    ranks_depth_local = ranks_depth.cuda()
    ranks_feat_local = ranks_feat.cuda()
    ranks_bev_local = ranks_bev.cuda()
    interval_starts_local = interval_starts.cuda()
    interval_lengths_local = interval_lengths.cuda()

    t0 = time.time()
    E.bev_pool_baseline(ctypes.c_int(128),
                        ctypes.c_int(n_intervals),
                        ctypes.c_void_p(depth_local.data_ptr()),
                        ctypes.c_void_p(feat_local.data_ptr()),
                        ctypes.c_void_p(ranks_depth_local.data_ptr()),
                        ctypes.c_void_p(ranks_feat_local.data_ptr()),
                        ctypes.c_void_p(ranks_bev_local.data_ptr()),
                        ctypes.c_void_p(interval_starts_local.data_ptr()),
                        ctypes.c_void_p(interval_lengths_local.data_ptr()),
                        ctypes.c_void_p(bev_baseline_float.data_ptr()))
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"baseline: {(t1-t0)*1000:.3f} ms")


def test_float_float():
    depth_local = depth.cuda()
    feat_local = feat.cuda()
    bev_local = bev.cuda()
    ranks_depth_local = ranks_depth.cuda()
    ranks_feat_local = ranks_feat.cuda()
    interval_starts_local = interval_starts_e.cuda()
    interval_lengths_local = interval_lengths_e.cuda()

    t0 = time.time()
    E.bev_pool_float_float(ctypes.c_int(128),
                           ctypes.c_int(n_intervals),
                           ctypes.c_void_p(depth_local.data_ptr()),
                           ctypes.c_void_p(feat_local.data_ptr()),
                           ctypes.c_void_p(ranks_depth_local.data_ptr()),
                           ctypes.c_void_p(ranks_feat_local.data_ptr()),
                           ctypes.c_void_p(interval_starts_local.data_ptr()),
                           ctypes.c_void_p(interval_lengths_local.data_ptr()),
                           ctypes.c_void_p(bev_local.data_ptr()))
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"float_float: {(t1-t0)*1000:.3f} ms")
    if diff and not torch.allclose(bev_local, bev_baseline_float, equal_nan=False):
      print("bev_baseline_float:", bev_baseline_float)
      print("bev_lcoal:", bev_local)


def test_half_float():
    depth_local = depth.half().cuda()
    feat_local = feat.half().cuda()
    bev_local = bev.half().cuda()
    ranks_depth_local = ranks_depth.cuda()
    ranks_feat_local = ranks_feat.cuda()
    interval_starts_local = interval_starts_e.cuda()
    interval_lengths_local = interval_lengths_e.cuda()

    t0 = time.time()
    E.bev_pool_half_float(ctypes.c_int(128),
                          ctypes.c_int(n_intervals),
                          ctypes.c_void_p(depth_local.data_ptr()),
                          ctypes.c_void_p(feat_local.data_ptr()),
                          ctypes.c_void_p(ranks_depth_local.data_ptr()),
                          ctypes.c_void_p(ranks_feat_local.data_ptr()),
                          ctypes.c_void_p(interval_starts_local.data_ptr()),
                          ctypes.c_void_p(interval_lengths_local.data_ptr()),
                          ctypes.c_void_p(bev_local.data_ptr()))
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"half_float: {(t1-t0)*1000:.3f} ms")
    if diff and not torch.allclose(bev_local.float(), bev_baseline_float, rtol=1e01, atol=1e-01, equal_nan=False):
      print("bev_baseline_float:", bev_baseline_float)
      print("bev_local:", bev_local)

def test_half_half():
    depth_local = depth.half().cuda()
    feat_local = feat.half().cuda()
    bev_local = bev.half().cuda()
    ranks_depth_local = ranks_depth.cuda()
    ranks_feat_local = ranks_feat.cuda()
    interval_starts_local = interval_starts_e.cuda()
    interval_lengths_local = interval_lengths_e.cuda()

    t0 = time.time()
    E.bev_pool_half_half(ctypes.c_int(128),
                         ctypes.c_int(n_intervals),
                         ctypes.c_void_p(depth_local.data_ptr()),
                         ctypes.c_void_p(feat_local.data_ptr()),
                         ctypes.c_void_p(ranks_depth_local.data_ptr()),
                         ctypes.c_void_p(ranks_feat_local.data_ptr()),
                         ctypes.c_void_p(interval_starts_local.data_ptr()),
                         ctypes.c_void_p(interval_lengths_local.data_ptr()),
                         ctypes.c_void_p(bev_local.data_ptr()))
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"half_half: {(t1-t0)*1000:.3f} ms")
    if diff and not torch.allclose(bev_local, bev_baseline_half, equal_nan=False):
      print("bev_baseline_half:", bev_baseline_half)
      print("bev_lcoal:", bev_local)


n_interval = 50000
n_valid_points = 4000000
n_out_grid_points = 1 * 192 * 256 * 128
n_total_depth_score = 7 * 120 * 64 * 120
n_total_img_feat = 7 * 64 * 120 * 128
def test_hpc_bev_pool_v2():
    depth_local = depth.cuda()
    feat_local = feat.cuda()
    ranks_depth_local = ranks_depth.cuda()
    ranks_feat_local = ranks_feat.cuda()
    ranks_bev_local = ranks_bev.cuda()
    interval_starts_local = interval_starts.cuda()
    interval_lengths_local = interval_lengths.cuda()

    t0 = time.time()
    H.bev_pool_v2(ctypes.c_int(128),
                  ctypes.c_int(n_interval),
                  ctypes.c_int(n_valid_points),
                  ctypes.c_int(n_out_grid_points),
                  ctypes.c_int(n_total_depth_score),
                  ctypes.c_int(n_total_img_feat),

                  ctypes.c_void_p(depth_local.data_ptr()),
                  ctypes.c_void_p(feat_local.data_ptr()),
                  ctypes.c_void_p(ranks_depth_local.data_ptr()),
                  ctypes.c_void_p(ranks_feat_local.data_ptr()),
                  ctypes.c_void_p(ranks_bev_local.data_ptr()),
                  ctypes.c_void_p(interval_starts_local.data_ptr()),
                  ctypes.c_void_p(interval_lengths_local.data_ptr()),
                  ctypes.c_void_p(bev_baseline_float.data_ptr()),
                  ctypes.c_longlong(0))
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"hpc:bev_pool_v2: {(t1-t0)*1000:.3f} ms")

def test_hpc_bev_pool_pack32_half():
    depth_local = depth.half().cuda()
    feat_local = feat.half().cuda()
    ranks_depth_local = ranks_depth.cuda()
    ranks_feat_local = ranks_feat.cuda()
    ranks_bev_local = ranks_bev.cuda()
    interval_starts_local = interval_starts.cuda()
    interval_lengths_local = interval_lengths.cuda()

    t0 = time.time()
    H.bev_pool_pack32_half(ctypes.c_int(128),
                           ctypes.c_int(n_interval),
                           ctypes.c_int(n_valid_points),
                           ctypes.c_int(n_out_grid_points),
                           ctypes.c_int(n_total_depth_score),
                           ctypes.c_int(n_total_img_feat),

                           ctypes.c_void_p(depth_local.data_ptr()),
                           ctypes.c_void_p(feat_local.data_ptr()),
                           ctypes.c_void_p(ranks_depth_local.data_ptr()),
                           ctypes.c_void_p(ranks_feat_local.data_ptr()),
                           ctypes.c_void_p(ranks_bev_local.data_ptr()),
                           ctypes.c_void_p(interval_starts_local.data_ptr()),
                           ctypes.c_void_p(interval_lengths_local.data_ptr()),
                           ctypes.c_void_p(bev_baseline_half.data_ptr()),
                           ctypes.c_longlong(0))
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"hpc:bev_pool_pack32_half: {(t1-t0)*1000:.3f} ms")




if __name__ == "__main__":
  if diff:
    test_hpc_bev_pool_v2()
    test_hpc_bev_pool_pack32_half()

  for i in range(0, 2):
    print("=== Iteration: ", i)
    test_hpc_bev_pool_v2()
    test_hpc_bev_pool_pack32_half()
    test_float_float()
    test_half_float()
    test_half_half()

