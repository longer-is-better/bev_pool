# BEV_Pooling Kernel Tuning
---

* Build, Run

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

* Tuning Parameters
  * Inner Tiling
    * Channel Tiling: constexpr int TC = 2;
    * Spatial/Batch Tiling: constexpr int TN = 2;

  * Thread Blocking:
    * Channel Thread Blocking: constexpr int BC = 32;
    * Spatial/Batch Thread Blocking: constexpr int BN = 8;

  * Tata type
    * float with float accumulation: bev_pool_kernel<float, float, TC, TN>...
    * float16 with float accumulation: bev_pool_kernel<__half, float, TC, TN>...
    * float16 with float16 accumulation: bev_pool_kernel<__half, __half, TC, TN>...


