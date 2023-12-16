/ota/dongwei/space/NsightSystems-linux-nda-2023.1.3.51-3228107/opt/nvidia/nsight-systems/2023.1.3/target-linux-tegra-armv8/nsys \
    profile \
        -t cuda,cudnn,osrt,nvtx,tegra-accelerators \
        --accelerator-trace=tegra-accelerators \
        --gpuctxsw=true \
        --gpu-metrics-device=all \
        --cuda-graph-trace=node \
        -w true -f true \
        -o bev_pool_v2 \
    python3 ./bev_pool.py


sudo /ota/dongwei/space/nsight-compute-addon-l4t-2021.2.8_2021.2.8.1-1_all/opt/nvidia/nsight-compute/2021.2.8/target/linux-v4l_l4t-t210-a64/ncu -f -o bev_pool --import-source on --set full  bev_pool