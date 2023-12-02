import ctypes
import time
import torch

diff = True

EXE = ctypes.cdll.LoadLibrary("./bev_pool.so")
HPC = ctypes.cdll.LoadLibrary("hpc/build/libbev_pool_shared.so")

N = EXE.get_config(ctypes.c_char_p(b'N'))
D = EXE.get_config(ctypes.c_char_p(b'D'))
IH = EXE.get_config(ctypes.c_char_p(b'IH'))
IW = EXE.get_config(ctypes.c_char_p(b'IW'))
C = EXE.get_config(ctypes.c_char_p(b'C'))
OH = EXE.get_config(ctypes.c_char_p(b"OH"))
OW = EXE.get_config(ctypes.c_char_p(b"OW"))
P = EXE.get_config(ctypes.c_char_p(b'P'))

print(f"N={N}, D={D}, IH={IH}, IW={IW}, {C}, OH={OH}, OW={OW}, P={P}")

depth = torch.randn((N, D, IH, IW), dtype=torch.float)
feat = torch.randn((N, IH, IW, C), dtype=torch.float)
bev = torch.ones(size=[1, OH, OW, C], dtype=torch.float)
bev_nchw = torch.clone(bev).reshape(1, C, OH, OW)

bev_baseline_float = torch.zeros(size=[1, OH, OW, C], dtype=torch.float, device='cuda')
bev_baseline_half = torch.zeros(size=[1, OH, OW, C], dtype=torch.half, device='cuda')
bev_baseline_float_nchw = torch.clone(bev).reshape(1, C, OH, OW).cuda()
bev_baseline_half_nchw = torch.clone(bev).half().reshape(1, C, OH, OW).cuda()

ranks_depth = torch.zeros((4000000), dtype=torch.int32)
ranks_feat = torch.zeros((4000000), dtype=torch.int32)
ranks_bev = torch.zeros((4000000), dtype=torch.int32)
interval_starts = torch.zeros((50000), dtype=torch.int32)
interval_lengths = torch.zeros((50000), dtype=torch.int32)
interval_starts_e = torch.zeros((50000), dtype=torch.int32)
interval_lengths_e = torch.zeros((50000), dtype=torch.int32)




n_intervals = OH * OW
ranks_bev_mask = torch.zeros(OH * OW, dtype=torch.int8)

EXE.tensor_init(ctypes.c_void_p(ranks_depth.data_ptr()),
              ctypes.c_void_p(ranks_feat.data_ptr()),
              ctypes.c_void_p(ranks_bev.data_ptr()),
              ctypes.c_void_p(interval_starts.data_ptr()),
              ctypes.c_void_p(interval_lengths.data_ptr()),
              ctypes.c_void_p(ranks_bev_mask.data_ptr()),
              ctypes.c_void_p(interval_starts_e.data_ptr()),
              ctypes.c_void_p(interval_lengths_e.data_ptr()))

depth_nhwd = torch.permute(depth, (0, 2, 3, 1)).contiguous()
ranks_depth_nhwd = ranks_depth.clone()
EXE.tensor_NDHW_to_NHWD(ctypes.c_void_p(ranks_depth_nhwd.data_ptr()),
                   ctypes.c_ulonglong(4000000),
                   ctypes.c_int(N),
                   ctypes.c_int(D),
                   ctypes.c_int(IH),
                   ctypes.c_int(IW))

feat_nchw = torch.permute(feat, (0, 3, 1, 2)).contiguous()

def test_flatmap():
    depth_local = depth.cuda()
    feat_local = feat.cuda()
    bev_local = bev.cuda()
    ranks_depth_local = ranks_depth.cuda()
    ranks_feat_local = ranks_feat.cuda()
    ranks_bev_local = ranks_bev.cuda()
    interval_starts_local = interval_starts.cuda()
    interval_lengths_local = interval_lengths.cuda()

    t0 = time.time()
    EXE.bev_pool_flatmap(ctypes.c_int(C),
                         ctypes.c_int(n_intervals),
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
    print(f"flatmap: {(t1-t0)*1000:.3f} ms")
    if diff and not torch.allclose(bev_local, bev_baseline_float, rtol=1e-03, atol=1e-04, equal_nan=False):
      print("bev_baseline_float:", bev_baseline_float)
      print("bev_local:", bev_local)


def test_float_float():
    depth_local = depth.cuda()
    feat_local = feat.cuda()
    bev_local = bev.cuda()
    ranks_depth_local = ranks_depth.cuda()
    ranks_feat_local = ranks_feat.cuda()
    interval_starts_local = interval_starts_e.cuda()
    interval_lengths_local = interval_lengths_e.cuda()

    t0 = time.time()
    EXE.bev_pool_float_float(ctypes.c_int(C),
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
      print("bev_local:", bev_local)

def test_float_float_nhwd():
    depth_local = depth_nhwd.cuda()
    feat_local = feat.cuda()
    bev_local = bev.cuda()
    ranks_depth_local = ranks_depth_nhwd.cuda()
    ranks_feat_local = ranks_feat.cuda()
    interval_starts_local = interval_starts_e.cuda()
    interval_lengths_local = interval_lengths_e.cuda()

    t0 = time.time()
    EXE.bev_pool_float_float(ctypes.c_int(C),
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
    print(f"float_float_nhwd: {(t1-t0)*1000:.3f} ms")
    if diff and not torch.allclose(bev_local, bev_baseline_float, equal_nan=False):
      print("bev_baseline_float:", bev_baseline_float)
      print("bev_local:", bev_local)

def test_float_float_nchw():
    depth_local = depth.cuda()
    feat_local = feat_nchw.cuda()
    bev_local = bev_nchw.cuda()
    ranks_depth_local = ranks_depth.cuda()
    ranks_feat_local = ranks_feat.cuda()
    interval_starts_local = interval_starts_e.cuda()
    interval_lengths_local = interval_lengths_e.cuda()

    t0 = time.time()
    EXE.bev_pool_float_float_nchw(ctypes.c_int(C),
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
    print(f"float_float_nchw: {(t1-t0)*1000:.3f} ms")
    if diff and not torch.allclose(bev_local, bev_baseline_float_nchw, equal_nan=False):
      print("bev_baseline_float_nchw:", bev_baseline_float_nchw)
      print("bev_local:", bev_local)



def test_half_float():
    depth_local = depth.half().cuda()
    feat_local = feat.half().cuda()
    bev_local = bev.half().cuda()
    ranks_depth_local = ranks_depth.cuda()
    ranks_feat_local = ranks_feat.cuda()
    interval_starts_local = interval_starts_e.cuda()
    interval_lengths_local = interval_lengths_e.cuda()

    t0 = time.time()
    EXE.bev_pool_half_float(ctypes.c_int(C),
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
      print("bev_baseline_float_nchw:", bev_baseline_float_nchw)
      print("bev_local:", bev_local)

def test_half_float_nhwd():
    depth_local = depth_nhwd.half().cuda()
    feat_local = feat.half().cuda()
    bev_local = bev.half().cuda()
    ranks_depth_local = ranks_depth_nhwd.cuda()
    ranks_feat_local = ranks_feat.cuda()
    interval_starts_local = interval_starts_e.cuda()
    interval_lengths_local = interval_lengths_e.cuda()

    t0 = time.time()
    EXE.bev_pool_half_float(ctypes.c_int(C),
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
    print(f"half_float_nhwd: {(t1-t0)*1000:.3f} ms")
    if diff and not torch.allclose(bev_local.float(), bev_baseline_float, rtol=1e01, atol=1e-01, equal_nan=False):
      print("bev_baseline_float:", bev_baseline_float)
      print("bev_local:", bev_local)


def test_half_float_nchw():
    depth_local = depth.half().cuda()
    feat_local = feat_nchw.half().cuda()
    bev_local = bev_nchw.half().cuda()
    ranks_depth_local = ranks_depth.cuda()
    ranks_feat_local = ranks_feat.cuda()
    interval_starts_local = interval_starts_e.cuda()
    interval_lengths_local = interval_lengths_e.cuda()

    t0 = time.time()
    EXE.bev_pool_half_float_nchw(ctypes.c_int(C),
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
    print(f"half_float_nchw: {(t1-t0)*1000:.3f} ms")
    if diff and not torch.allclose(bev_local.float(), bev_baseline_float_nchw, rtol=1e01, atol=1e-01, equal_nan=False):
      print("bev_baseline_float_nchw:", bev_baseline_float_nchw)
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
    EXE.bev_pool_half_half(ctypes.c_int(C),
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
      print("bev_local:", bev_local)

def test_half_half_nhwd():
    depth_local = depth_nhwd.half().cuda()
    feat_local = feat.half().cuda()
    bev_local = bev.half().cuda()
    ranks_depth_local = ranks_depth_nhwd.cuda()
    ranks_feat_local = ranks_feat.cuda()
    interval_starts_local = interval_starts_e.cuda()
    interval_lengths_local = interval_lengths_e.cuda()

    t0 = time.time()
    EXE.bev_pool_half_half(ctypes.c_int(C),
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
    print(f"half_half_nhwd: {(t1-t0)*1000:.3f} ms")
    if diff and not torch.allclose(bev_local, bev_baseline_half, equal_nan=False):
      print("bev_baseline_half:", bev_baseline_half)
      print("bev_local:", bev_local)


def test_half_half_nchw():
    depth_local = depth.half().cuda()
    feat_local = feat_nchw.half().cuda()
    bev_local = bev_nchw.half().cuda()
    ranks_depth_local = ranks_depth.cuda()
    ranks_feat_local = ranks_feat.cuda()
    interval_starts_local = interval_starts_e.cuda()
    interval_lengths_local = interval_lengths_e.cuda()

    t0 = time.time()
    EXE.bev_pool_half_half_nchw(ctypes.c_int(C),
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
    print(f"half_half_nchw: {(t1-t0)*1000:.3f} ms")
    if diff and not torch.allclose(bev_local, bev_baseline_half_nchw, equal_nan=False):
      print("bev_baseline_half_nchw:", bev_baseline_half_nchw)
      print("bev_local:", bev_local)


n_interval = 50000
n_valid_points = 4000000
n_out_grid_points = 1 * OH * OW * C #192 * 256 * 128
n_total_depth_score = N * D * IH * IW #7 * 120 * 64 * 120
n_total_img_feat = N * C * IH * IW #7 * 64 * 120 * 128
def test_hpc_bev_pool_v2():
    depth_local = depth.cuda()
    feat_local = feat.cuda()
    ranks_depth_local = ranks_depth.cuda()
    ranks_feat_local = ranks_feat.cuda()
    ranks_bev_local = ranks_bev.cuda()
    interval_starts_local = interval_starts.cuda()
    interval_lengths_local = interval_lengths.cuda()

    t0 = time.time()
    HPC.bev_pool_v2(ctypes.c_int(C),
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
    global bev_baseline_float_nchw
    bev_baseline_float_nchw = torch.permute(bev_baseline_float, (0, 3, 1, 2)).contiguous()
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
    HPC.bev_pool_pack32_half(ctypes.c_int(C),
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
    global bev_baseline_half_nchw
    bev_baseline_half_nchw = torch.permute(bev_baseline_half, (0, 3, 1, 2)).cuda().contiguous()
    print(f"hpc:bev_pool_pack32_half: {(t1-t0)*1000:.3f} ms")




if __name__ == "__main__":
  if diff:
    test_hpc_bev_pool_v2()
    test_hpc_bev_pool_pack32_half()

  for i in range(0, 2):
    print("=== Iteration: ", i)
    test_hpc_bev_pool_v2()
    test_hpc_bev_pool_pack32_half()
    test_flatmap()
    test_float_float()
    test_float_float_nhwd()
    test_float_float_nchw()
    test_half_float()
    test_half_float_nhwd()
    test_half_float_nchw()
    test_half_half()
    test_half_half_nhwd()
    test_half_half_nchw()

