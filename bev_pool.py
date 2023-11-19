import ctypes
import time
import torch



E = ctypes.cdll.LoadLibrary("./bev_pool.so")

depth = torch.randn((7, 120, 64, 120), dtype=torch.float)
feat = torch.randn((7, 64, 120, 128), dtype=torch.float)
bev = torch.zeros(size=[1, 192, 256, 128], dtype=torch.float)

bev_baseline = torch.zeros(size=[1, 192, 256, 128], dtype=torch.float, device='cuda')

# each interval has len pixels
len = 30
ranks_depth = torch.zeros((192 * 256 * len), dtype=torch.int32)
ranks_feat = torch.zeros((192 * 256 * len), dtype=torch.int32)
ranks_bev = torch.zeros((192 * 256 * len), dtype=torch.int32)
interval_starts = torch.zeros((192 * 256), dtype=torch.int32)
interval_lengths = torch.zeros((192 * 256), dtype=torch.int32)

E.tensor_init(ctypes.c_void_p(ranks_depth.data_ptr()),
              ctypes.c_void_p(ranks_feat.data_ptr()),
              ctypes.c_void_p(ranks_bev.data_ptr()),
              ctypes.c_void_p(interval_starts.data_ptr()),
              ctypes.c_void_p(interval_lengths.data_ptr()),
              ctypes.c_int(192 * 256), ctypes.c_int(len))

ranks = torch.stack((ranks_depth, ranks_feat), dim=-1).reshape(-1)
intervals = torch.stack((interval_starts, interval_lengths), dim=-1).reshape(-1)


def test_baseline():
    depth_local = depth.cuda()
    feat_local = feat.cuda()
    ranks_depth_local = ranks_depth.cuda()
    ranks_feat_local = ranks_feat.cuda()
    ranks_bev_local = ranks_bev.cuda()
    interval_starts_local = interval_starts.cuda()
    interval_lengths_local = interval_lengths.cuda()

    t0 = time.time()
    E.bev_pool_baseline(ctypes.c_int(128),
                        ctypes.c_int(192 * 256),
                        ctypes.c_void_p(depth_local.data_ptr()),
                        ctypes.c_void_p(feat_local.data_ptr()),
                        ctypes.c_void_p(ranks_depth_local.data_ptr()),
                        ctypes.c_void_p(ranks_feat_local.data_ptr()),
                        ctypes.c_void_p(ranks_bev_local.data_ptr()),
                        ctypes.c_void_p(interval_starts_local.data_ptr()),
                        ctypes.c_void_p(interval_lengths_local.data_ptr()),
                        ctypes.c_void_p(bev_baseline.data_ptr()))
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"baseline: {(t1-t0)*1000:.3f} ms")

test_baseline()

def test_float_float_1_1():
    depth_local = depth.cuda()
    feat_local = feat.cuda()
    bev_local = bev.cuda()
    ranks_depth_local = ranks_depth.cuda()
    ranks_feat_local = ranks_feat.cuda()
    ranks_bev_local = ranks_bev.cuda()
    interval_starts_local = interval_starts.cuda()
    interval_lengths_local = interval_lengths.cuda()

    t0 = time.time()
    E.bev_pool_float_float_1_1(ctypes.c_int(128),
                               ctypes.c_int(192 * 256),
                               ctypes.c_void_p(depth_local.data_ptr()),
                               ctypes.c_void_p(feat_local.data_ptr()),
                               ctypes.c_void_p(ranks_depth_local.data_ptr()),
                               ctypes.c_void_p(ranks_feat_local.data_ptr()),
                               ctypes.c_void_p(ranks_bev_local.data_ptr()),
                               ctypes.c_void_p(interval_starts_local.data_ptr()),
                               ctypes.c_void_p(interval_lengths_local.data_ptr()),
                               ctypes.c_void_p(bev_local.data_ptr()))
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"float_float_1_1: {(t1-t0)*1000:.3f} ms")
    if not torch.allclose(bev_local, bev_baseline, equal_nan=False):
      print("bev_lcoal:", bev_local)
      print("bev_baseline:", bev_baseline)




def test_float_float_2_2():
    depth_local = depth.cuda()
    feat_local = feat.cuda()
    bev_local = bev.cuda()
    ranks_depth_local = ranks_depth.cuda()
    ranks_feat_local = ranks_feat.cuda()
    ranks_bev_local = ranks_bev.cuda()
    interval_starts_local = interval_starts.cuda()
    interval_lengths_local = interval_lengths.cuda()

    t0 = time.time()
    E.bev_pool_float_float_2_2(ctypes.c_int(128),
                               ctypes.c_int(192 * 256),
                               ctypes.c_void_p(depth_local.data_ptr()),
                               ctypes.c_void_p(feat_local.data_ptr()),
                               ctypes.c_void_p(ranks_depth_local.data_ptr()),
                               ctypes.c_void_p(ranks_feat_local.data_ptr()),
                               ctypes.c_void_p(ranks_bev_local.data_ptr()),
                               ctypes.c_void_p(interval_starts_local.data_ptr()),
                               ctypes.c_void_p(interval_lengths_local.data_ptr()),
                               ctypes.c_void_p(bev_local.data_ptr()))
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"float_float_2_2: {(t1-t0)*1000:.3f} ms")
    if not torch.allclose(bev_local, bev_baseline, equal_nan=False):
      print("bev_lcoal:", bev_local)
      print("bev_baseline:", bev_baseline)



def test_half_float_2_2():
    depth_local = depth.half().cuda()
    feat_local = feat.half().cuda()
    bev_local = bev.half().cuda()
    ranks_depth_local = ranks_depth.cuda()
    ranks_feat_local = ranks_feat.cuda()
    ranks_bev_local = ranks_bev.cuda()
    interval_starts_local = interval_starts.cuda()
    interval_lengths_local = interval_lengths.cuda()

    t0 = time.time()
    E.bev_pool_half_float_2_2(ctypes.c_int(128),
                              ctypes.c_int(192 * 256),
                              ctypes.c_void_p(depth_local.data_ptr()),
                              ctypes.c_void_p(feat_local.data_ptr()),
                              ctypes.c_void_p(ranks_depth_local.data_ptr()),
                              ctypes.c_void_p(ranks_feat_local.data_ptr()),
                              ctypes.c_void_p(ranks_bev_local.data_ptr()),
                              ctypes.c_void_p(interval_starts_local.data_ptr()),
                              ctypes.c_void_p(interval_lengths_local.data_ptr()),
                              ctypes.c_void_p(bev_local.data_ptr()))
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"half_float_2_2: {(t1-t0)*1000:.3f} ms")
    if not torch.allclose(bev_local.float(), bev_baseline, rtol=1e-01, atol=1e-01, equal_nan=False):
      print("bev_lcoal:", bev_local)
      print("bev_baseline:", bev_baseline)

def test_half_half_2_2():
    depth_local = depth.half().cuda()
    feat_local = feat.half().cuda()
    bev_local = bev.half().cuda()
    ranks_depth_local = ranks_depth.cuda()
    ranks_feat_local = ranks_feat.cuda()
    ranks_bev_local = ranks_bev.cuda()
    interval_starts_local = interval_starts.cuda()
    interval_lengths_local = interval_lengths.cuda()

    t0 = time.time()
    E.bev_pool_half_half_2_2(ctypes.c_int(128),
                             ctypes.c_int(192 * 256),
                             ctypes.c_void_p(depth_local.data_ptr()),
                             ctypes.c_void_p(feat_local.data_ptr()),
                             ctypes.c_void_p(ranks_depth_local.data_ptr()),
                             ctypes.c_void_p(ranks_feat_local.data_ptr()),
                             ctypes.c_void_p(ranks_bev_local.data_ptr()),
                             ctypes.c_void_p(interval_starts_local.data_ptr()),
                             ctypes.c_void_p(interval_lengths_local.data_ptr()),
                             ctypes.c_void_p(bev_local.data_ptr()))
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"half_half_2_2: {(t1-t0)*1000:.3f} ms")
    if not torch.allclose(bev_local.float(), bev_baseline, rtol=1e-01, atol=1e-01, equal_nan=False):
      print("bev_lcoal:", bev_local)
      print("bev_baseline:", bev_baseline)


if __name__ == "__main__":
  for i in range(0, 3):
    print("=== Iteration: ", i)
    test_baseline()
    test_float_float_1_1()
    test_float_float_2_2()
    test_half_float_2_2()
    test_half_half_2_2()
