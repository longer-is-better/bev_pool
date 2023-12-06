import ctypes
import time
import torch

print_diff = False
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

print(f"N={N}, D={D}, IH={IH}, IW={IW}, C={C}, OH={OH}, OW={OW}, P={P}")

depth = torch.randn((N, D, IH, IW), dtype=torch.float)
feat = torch.randn((N, IH, IW, C), dtype=torch.float)
bev = torch.ones(size=[1, OH, OW, C], dtype=torch.float)
bev_nchw = torch.clone(bev).reshape(1, C, OH, OW)

bev_baseline_float = torch.zeros(size=[1, OH, OW, C], dtype=torch.float, device='cuda')
bev_baseline_float_nchw = torch.clone(bev).reshape(1, C, OH, OW).cuda()

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

def compare_tensors(tensor1, tensor2, rtol=1e-03, atol=1e-05):
    closeness = torch.isclose(tensor1, tensor2, atol=atol, rtol=rtol)
    if not torch.all(closeness):
        num_total = tensor1.numel()
        num_not_close = torch.sum(~closeness).item()
        percentage_not_close = (num_not_close / num_total) * 100

        print(f"\t Percentage of elements not close: {percentage_not_close:.2f}%")
        if print_diff is True:
            print("\t First 10 elements that are not close:")

            # Find the indices of the first 10 non-close elements
            differing_indices = torch.where(~closeness)
            count = 0
            for idx in zip(*differing_indices):
                print(f"\t got: {tensor1[idx]} \t expect: {tensor2[idx]}")
                count += 1
                if count >= 10:
                    break

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
    compare_tensors(bev_local, bev_baseline_float)

def test_float_float_float():
    depth_local = depth.cuda()
    feat_local = feat.cuda()
    bev_local = bev.cuda()
    ranks_depth_local = ranks_depth.cuda()
    ranks_feat_local = ranks_feat.cuda()
    interval_starts_local = interval_starts_e.cuda()
    interval_lengths_local = interval_lengths_e.cuda()

    t0 = time.time()
    EXE.bev_pool_float_float_float(ctypes.c_int(C),
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
    print(f"float_float_float: {(t1-t0)*1000:.3f} ms")
    compare_tensors(bev_local, bev_baseline_float)

def test_float_float_float_nhwd():
    depth_local = depth_nhwd.cuda()
    feat_local = feat.cuda()
    bev_local = bev.cuda()
    ranks_depth_local = ranks_depth_nhwd.cuda()
    ranks_feat_local = ranks_feat.cuda()
    interval_starts_local = interval_starts_e.cuda()
    interval_lengths_local = interval_lengths_e.cuda()

    t0 = time.time()
    EXE.bev_pool_float_float_float(ctypes.c_int(C),
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
    print(f"float_float_float_nhwd: {(t1-t0)*1000:.3f} ms")
    compare_tensors(bev_local, bev_baseline_float)

def test_float_float_float_nchw():
    depth_local = depth.cuda()
    feat_local = feat_nchw.cuda()
    bev_local = bev_nchw.cuda()
    ranks_depth_local = ranks_depth.cuda()
    ranks_feat_local = ranks_feat.cuda()
    interval_starts_local = interval_starts_e.cuda()
    interval_lengths_local = interval_lengths_e.cuda()

    t0 = time.time()
    EXE.bev_pool_float_float_float_nchw(ctypes.c_int(C),
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
    print(f"float_float_float_nchw: {(t1-t0)*1000:.3f} ms")
    compare_tensors(bev_local, bev_baseline_float_nchw)



def test_half_float_half():
    depth_local = depth.half().cuda()
    feat_local = feat.half().cuda()
    bev_local = bev.half().cuda()
    ranks_depth_local = ranks_depth.cuda()
    ranks_feat_local = ranks_feat.cuda()
    interval_starts_local = interval_starts_e.cuda()
    interval_lengths_local = interval_lengths_e.cuda()

    t0 = time.time()
    EXE.bev_pool_half_float_half(ctypes.c_int(C),
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
    print(f"half_float_half: {(t1-t0)*1000:.3f} ms")
    compare_tensors(bev_local.float(), bev_baseline_float)

def test_half_float_float():
    depth_local = depth.half().cuda()
    feat_local = feat.half().cuda()
    bev_local = bev.cuda()
    ranks_depth_local = ranks_depth.cuda()
    ranks_feat_local = ranks_feat.cuda()
    interval_starts_local = interval_starts_e.cuda()
    interval_lengths_local = interval_lengths_e.cuda()

    t0 = time.time()
    EXE.bev_pool_half_float_float(ctypes.c_int(C),
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
    print(f"half_float_float: {(t1-t0)*1000:.3f} ms")
    compare_tensors(bev_local, bev_baseline_float)


def test_bf16_float_bf16():
    depth_local = depth.bfloat16().cuda()
    feat_local = feat.bfloat16().cuda()
    bev_local = bev.bfloat16().cuda()
    ranks_depth_local = ranks_depth.cuda()
    ranks_feat_local = ranks_feat.cuda()
    interval_starts_local = interval_starts_e.cuda()
    interval_lengths_local = interval_lengths_e.cuda()

    t0 = time.time()
    EXE.bev_pool_bf16_float_bf16(ctypes.c_int(C),
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
    print(f"bf16_float_bf16: {(t1-t0)*1000:.3f} ms")
    compare_tensors(bev_local.float(), bev_baseline_float)

def test_bf16_float_float():
    depth_local = depth.bfloat16().cuda()
    feat_local = feat.bfloat16().cuda()
    bev_local = bev.cuda()
    ranks_depth_local = ranks_depth.cuda()
    ranks_feat_local = ranks_feat.cuda()
    interval_starts_local = interval_starts_e.cuda()
    interval_lengths_local = interval_lengths_e.cuda()

    t0 = time.time()
    EXE.bev_pool_bf16_float_float(ctypes.c_int(C),
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
    print(f"bf16_float_float: {(t1-t0)*1000:.3f} ms")
    compare_tensors(bev_local, bev_baseline_float)



def test_half_half_half():
    depth_local = depth.half().cuda()
    feat_local = feat.half().cuda()
    bev_local = bev.half().cuda()
    ranks_depth_local = ranks_depth.cuda()
    ranks_feat_local = ranks_feat.cuda()
    interval_starts_local = interval_starts_e.cuda()
    interval_lengths_local = interval_lengths_e.cuda()

    t0 = time.time()
    EXE.bev_pool_half_half_half(ctypes.c_int(C),
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
    print(f"half_half_half: {(t1-t0)*1000:.3f} ms")
    compare_tensors(bev_local.float(), bev_baseline_float)

def test_bf16_bf16_bf16():
    depth_local = depth.bfloat16().cuda()
    feat_local = feat.bfloat16().cuda()
    bev_local = bev.bfloat16().cuda()
    ranks_depth_local = ranks_depth.cuda()
    ranks_feat_local = ranks_feat.cuda()
    interval_starts_local = interval_starts_e.cuda()
    interval_lengths_local = interval_lengths_e.cuda()

    t0 = time.time()
    EXE.bev_pool_bf16_bf16_bf16(ctypes.c_int(C),
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
    print(f"bf16_bf16_bf16: {(t1-t0)*1000:.3f} ms")
    compare_tensors(bev_local.float(), bev_baseline_float)

def test_v2_float_float_float():
    depth_local = depth.cuda()
    feat_local = feat.cuda()
    bev_local = bev.cuda()
    ranks_depth_local = ranks_depth.cuda()
    ranks_feat_local = ranks_feat.cuda()
    interval_starts_local = interval_starts_e.cuda()
    interval_lengths_local = interval_lengths_e.cuda()

    t0 = time.time()
    EXE.bev_pool_v2_float_float_float(ctypes.c_int(C),
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
    print(f"v2: float_float_float: {(t1-t0)*1000:.3f} ms")
    compare_tensors(bev_local.float(), bev_baseline_float)

def test_v2_float_half_half():
    depth_local = depth.cuda()
    feat_local = feat.half().cuda()
    bev_local = bev.half().cuda()
    ranks_depth_local = ranks_depth.cuda()
    ranks_feat_local = ranks_feat.cuda()
    interval_starts_local = interval_starts_e.cuda()
    interval_lengths_local = interval_lengths_e.cuda()

    t0 = time.time()
    EXE.bev_pool_v2_float_half_half(ctypes.c_int(C),
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
    print(f"v2: float_half_half: {(t1-t0)*1000:.3f} ms")
    compare_tensors(bev_local.float(), bev_baseline_float)

def test_v2_float_half_float():
    depth_local = depth.cuda()
    feat_local = feat.half().cuda()
    bev_local = bev.cuda()
    ranks_depth_local = ranks_depth.cuda()
    ranks_feat_local = ranks_feat.cuda()
    interval_starts_local = interval_starts_e.cuda()
    interval_lengths_local = interval_lengths_e.cuda()

    t0 = time.time()
    EXE.bev_pool_v2_float_half_float(ctypes.c_int(C),
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
    print(f"v2: float_half_float: {(t1-t0)*1000:.3f} ms")
    compare_tensors(bev_local, bev_baseline_float)

def test_v2_half_half_float():
    depth_local = depth.half().cuda()
    feat_local = feat.half().cuda()
    bev_local = bev.cuda()
    ranks_depth_local = ranks_depth.cuda()
    ranks_feat_local = ranks_feat.cuda()
    interval_starts_local = interval_starts_e.cuda()
    interval_lengths_local = interval_lengths_e.cuda()

    t0 = time.time()
    EXE.bev_pool_v2_half_half_float(ctypes.c_int(C),
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
    print(f"v2: half_half_float: {(t1-t0)*1000:.3f} ms")
    compare_tensors(bev_local, bev_baseline_float)

def test_v2_half_float_float():
    depth_local = depth.half().cuda()
    feat_local = feat.cuda()
    bev_local = bev.cuda()
    ranks_depth_local = ranks_depth.cuda()
    ranks_feat_local = ranks_feat.cuda()
    interval_starts_local = interval_starts_e.cuda()
    interval_lengths_local = interval_lengths_e.cuda()

    t0 = time.time()
    EXE.bev_pool_v2_half_float_float(ctypes.c_int(C),
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
    print(f"v2: half_float_float: {(t1-t0)*1000:.3f} ms")
    compare_tensors(bev_local, bev_baseline_float)

def test_v2_half_float_half():
    depth_local = depth.half().cuda()
    feat_local = feat.cuda()
    bev_local = bev.half().cuda()
    ranks_depth_local = ranks_depth.cuda()
    ranks_feat_local = ranks_feat.cuda()
    interval_starts_local = interval_starts_e.cuda()
    interval_lengths_local = interval_lengths_e.cuda()

    t0 = time.time()
    EXE.bev_pool_v2_half_float_half(ctypes.c_int(C),
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
    print(f"v2: half_float_half: {(t1-t0)*1000:.3f} ms")
    compare_tensors(bev_local.float(), bev_baseline_float)



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
    bev_local = bev.half().cuda()
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
                             ctypes.c_void_p(bev_local.data_ptr()),
                             ctypes.c_longlong(0))
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"hpc:bev_pool_pack32_half: {(t1-t0)*1000:.3f} ms")
    compare_tensors(bev_local.float(), bev_baseline_float)




if __name__ == "__main__":
  test_hpc_bev_pool_v2()

  for i in range(0, 2):
    if i >= 1:
      print_diff = False

    print("=== Iteration: ", i)
    test_hpc_bev_pool_v2()
    test_hpc_bev_pool_pack32_half()
    test_flatmap()

    test_float_float_float_nhwd()
    test_float_float_float_nchw()
    test_float_float_float()

    test_half_float_float()
    test_half_half_half()
    test_half_float_half()

    test_bf16_float_bf16()
    test_bf16_bf16_bf16()
    test_bf16_float_float()

    # v2
    test_v2_float_float_float()
    test_v2_float_half_half()
    test_v2_float_half_float()
    test_v2_half_half_float()
    test_v2_half_float_half()
    test_v2_half_float_float()
