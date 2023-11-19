# BEV_Pooling Kernel Tuning
---


## Tuning Parameters


#### Inner Tiling
* Channel Tiling:   ```  constexpr int TC = 2;  ```
* Spatial/Batch Tiling: ```constexpr int TN = 2;```

#### Thread Blocking:
* Channel Thread Blocking: ```constexpr int BC = 32;```
* Spatial/Batch Thread Blocking: ```constexpr int BN = 8;```

#### Tata type
* float with float accumulation: ```bev_pool_kernel<float, float, TC, TN>...```
* float16 with float accumulation: ```bev_pool_kernel<__half, float, TC, TN>...```
* float16 with float16 accumulation: ```bev_pool_kernel<__half, __half, TC, TN>...```

Please check bev_pool.cu for details.

## Test

```python
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

```
Please check bev_pool.py for details.

## Build & Run

  ```bash
  $ ./r.sh
  baseline: 2.984 ms
  === Iteration:  0
  baseline: 0.437 ms
  float_float_1_1: 0.485 ms
  float_float_2_2: 0.417 ms
  half_float_2_2: 0.398 ms
  half_half_2_2: 0.475 ms
  === Iteration:  1
  baseline: 0.422 ms
  float_float_1_1: 0.402 ms
  float_float_2_2: 0.332 ms
  half_float_2_2: 0.316 ms
  half_half_2_2: 0.345 ms
  === Iteration:  2
  baseline: 0.430 ms
  float_float_1_1: 0.388 ms
  float_float_2_2: 0.314 ms
  half_float_2_2: 0.303 ms
  half_half_2_2: 0.349 ms
  ```
