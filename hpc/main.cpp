#include "cnpy.h"
#include <cuda_runtime.h>
#include "cuda_gadget.h"
#include <cuda_fp16.h>
using namespace std;

struct DIM {
    int x = 1;
    int y = 1;
    int z = 1;
    int w = 1;
    DIM(int inx, int iny, int inz, int inw) :
        x(inx), y(iny), z(inz), w(inw){};
    DIM(int inx, int iny, int inz) :
        x(inx), y(iny), z(inz){};
    DIM(int inx, int iny) :
        x(inx), y(iny){};
    DIM(int inx) :
        x(inx){};
    template <typename T>
    size_t size() {
        return x * y * z * w * sizeof(T);
    }
    size_t nums() {
        return x * y * z * w;
    }
};
#define DATA_TYPE float
#define DATA_TYPE2 int
#define DATA_TYPE3 float
#define DATA_TYPE4 half
int main(){
    cnpy::NpyArray interval_lengths = cnpy::npy_load("./interval_lengths.npz.npy");
    cnpy::NpyArray interval_starts = cnpy::npy_load("./interval_starts.npz.npy");
    cnpy::NpyArray ranks_bev = cnpy::npy_load("./ranks_bev.npz.npy");
    cnpy::NpyArray ranks_depth = cnpy::npy_load("./ranks_depth.npz.npy");
    cnpy::NpyArray ranks_feat = cnpy::npy_load("./ranks_feat.npz.npy");

    DIM depth_shape(7,120,64,120);
    DIM feat_shape(7,64,120,128);
    DIM out_shape(1,128,80,160);
    DATA_TYPE3* depth = (DATA_TYPE3*)malloc(depth_shape.size<DATA_TYPE3>());
    half* depth_opt = (half*)malloc(depth_shape.size<half>());

    DATA_TYPE* feat = (DATA_TYPE*)malloc(feat_shape.size<DATA_TYPE>());
    half* feat_opt = (half*)malloc(feat_shape.size<half>());


    DATA_TYPE* out = (DATA_TYPE*)malloc(out_shape.size<DATA_TYPE>());
    DATA_TYPE4* out_opt = (DATA_TYPE4*)malloc(out_shape.size<DATA_TYPE4>());

    DATA_TYPE2* interval_lengths_host = (DATA_TYPE2*)interval_lengths.data<DATA_TYPE2>();
    cout<<std::numeric_limits<float>::max()<<endl;

    cout<< interval_lengths_host[0]<<endl;
    DATA_TYPE2* interval_starts_host = interval_starts.data<DATA_TYPE2>();
    cout<< interval_starts_host[2]<<endl;
    DATA_TYPE2* ranks_bev_host = ranks_bev.data<DATA_TYPE2>();
    DATA_TYPE2* ranks_depth_host = ranks_depth.data<DATA_TYPE2>();
    DATA_TYPE2* ranks_feat_host = ranks_feat.data<DATA_TYPE2>();
    //init data
    for(int i=0;i<depth_shape.nums();i++){
        depth[i] = DATA_TYPE3(1.0*0.01);
    }
    for(int i=0;i<feat_shape.nums();i++){
        feat[i] = DATA_TYPE3(1.0*0.01);
    }
    for(int i=0;i<out_shape.nums();i++){
        out[i] = DATA_TYPE3(0.0);
    }

    for(int i=0;i<depth_shape.nums();i++){
        depth_opt[i] = half(1.0*0.01);
    }
    for(int i=0;i<feat_shape.nums();i++){
        feat_opt[i] = half(1.0*0.01);
    }
    for(int i=0;i<out_shape.nums();i++){
        out_opt[i] = DATA_TYPE4(0.0);
    }

    DATA_TYPE3 * depth_d;
    DATA_TYPE * feat_d;
    DATA_TYPE * out_d;
    // DATA_TYPE * out_d_opt;
    DATA_TYPE2* interval_lengths_d;
    DATA_TYPE2* interval_starts_d;
    DATA_TYPE2* ranks_bev_d;
    DATA_TYPE2* ranks_depth_d;
    DATA_TYPE2* ranks_feat_d;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMalloc(&depth_d,depth_shape.size<DATA_TYPE3>());
    cudaMalloc(&feat_d,feat_shape.size<DATA_TYPE>());
    cudaMalloc(&out_d,out_shape.size<DATA_TYPE>());

    half * depth_d_opt;
    half * feat_d_opt;
    DATA_TYPE4 * out_d_opt;
    cudaMalloc(&depth_d_opt,depth_shape.size<half>());
    cudaMalloc(&feat_d_opt,feat_shape.size<half>());
    cudaMalloc(&out_d_opt,out_shape.size<DATA_TYPE4>());

    cudaMalloc(&interval_lengths_d,interval_lengths.num_vals*sizeof(DATA_TYPE2));
    cudaMalloc(&interval_starts_d,interval_starts.num_vals*sizeof(DATA_TYPE2));
    cudaMalloc(&ranks_bev_d,   ranks_bev.num_vals*sizeof(DATA_TYPE2));
    cudaMalloc(&ranks_depth_d, ranks_depth.num_vals*sizeof(DATA_TYPE2));
    cudaMalloc(&ranks_feat_d,  ranks_feat.num_vals*sizeof(DATA_TYPE2));


    // copy data
    cudaMemcpyAsync(depth_d, depth, depth_shape.size<DATA_TYPE3>(),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(feat_d,  feat,  feat_shape.size<DATA_TYPE>(),cudaMemcpyHostToDevice,stream);

    cudaMemcpyAsync(depth_d_opt, depth_opt, depth_shape.size<half>(),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(feat_d_opt,  feat_opt,  feat_shape.size<half>(),cudaMemcpyHostToDevice,stream);


    cudaMemcpyAsync(interval_lengths_d,interval_lengths_host, interval_lengths.num_vals*sizeof(DATA_TYPE2),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(interval_starts_d, interval_starts_host,  interval_starts.num_vals*sizeof(DATA_TYPE2),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(ranks_bev_d,    ranks_bev_host,           ranks_bev.num_vals*sizeof(DATA_TYPE2),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(ranks_depth_d,  ranks_depth_host,         ranks_depth.num_vals*sizeof(DATA_TYPE2),cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(ranks_feat_d,   ranks_feat_host ,         ranks_feat.num_vals*sizeof(DATA_TYPE2),cudaMemcpyHostToDevice,stream);
    // depth            : input depth, FloatTensor[n,d,h,w]
    // feat             : input feat, FloatTensor[n,h,w,c]
    // ranks_depth      : input index of depth, IntTensor[n_points]
    // ranks_feat       : input index of feat, IntTensor[n_points]
    // ranks_bev        : output index, IntTensor[n_points]
    // interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    // interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
    // out              : output features, FloatTensor[1, h, w, c]
    int n_out_grid_points = out_shape.x*out_shape.y*out_shape.z*out_shape.w;
    int n_valid_points = ranks_depth.num_vals;
    int n_intervals = interval_lengths.num_vals;
    int n_feat_c = feat_shape.w;
    int n_total_depth_score = depth_shape.x*depth_shape.y*depth_shape.z*depth_shape.w;
    int n_total_img_feat = feat_shape.x*feat_shape.y*feat_shape.z*feat_shape.w;

    // GPU_Time(bev_pool_pack16(n_feat_c,
    //             n_intervals,
    //             n_valid_points,
    //             n_out_grid_points,
    //             n_total_depth_score,
    //             n_total_img_feat,
    //             (const DATA_TYPE3*)depth_d,
    //             (const float*)feat_d,
    //             (const DATA_TYPE2*)ranks_depth_d,
    //             (const DATA_TYPE2*)ranks_feat_d,
    //             (const DATA_TYPE2*)ranks_bev_d,
    //             (const DATA_TYPE2*)interval_starts_d,
    //             (const DATA_TYPE2*)interval_lengths_d,
    //             (float*) out_d_opt,
    //             stream),stream);

    get_last_cuda_err();
    cudaMemcpyAsync(out,   out_d,   out_shape.size<DATA_TYPE>(),cudaMemcpyDeviceToHost,stream);
    cudaMemcpyAsync(out_opt,   out_d_opt,   out_shape.size<DATA_TYPE4>(),cudaMemcpyDeviceToHost,stream);
    float mean_err=0;
    float max_err=0;
    for(int i=0;i<out_shape.nums();i++){
        // cout<<out[i]<<endl;
        // float err=out[i];
        float opt = __half2float(out_opt[i]);
        // float opt = out_opt[i];
        // cout<<opt<<endl;
        //float err= std::fabs(out_opt[i]-out[i]);
        float err= std::fabs( __half2float(out_opt[i])-out[i]);
        if(err>max_err){
            max_err=err;
        }
        mean_err+=err;
    }
    cout<<"max err: "<<max_err<<endl;
    cout<<"mean err: "<<mean_err<<endl;

    cudaFree(depth_d);
    cudaFree(depth_d_opt);

    cudaFree(feat_d);
    cudaFree(feat_d_opt);

    cudaFree(out_d);
    cudaFree(out_d_opt);
    cudaFree(interval_lengths_d);
    cudaFree(interval_starts_d);
    cudaFree(ranks_bev_d);
    cudaFree(ranks_depth_d);
    cudaFree(ranks_feat_d);
    cudaStreamDestroy(stream);
    return 0;
}

