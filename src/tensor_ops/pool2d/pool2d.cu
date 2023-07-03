#include "cuda_utils.cuh"

enum Pool2dKind {
    AVG,
    MIN,
    MAX,
};

struct Pool2dOp {
    Pool2dKind kind;
    size_t kernel;
    size_t stride;
    size_t padding;
    size_t dilation;
    size_t batch;
    size_t chan;
    size_t h_in;
    size_t h_out;
    size_t w_in;
    size_t w_out;
};

__device__ double init(const Pool2dOp op) {
    switch(op.kind) {
        case AVG:
            return 0.0;
        case MIN:
            return INFINITY;
        case MAX:
            return -INFINITY;
    }
}

template<typename T>
__device__ T accum(const Pool2dOp op, const T accum, const T item) {
    switch(op.kind) {
        case AVG:
            return accum + item;
        case MIN:
            return ming(accum, item);
        case MAX:
            return maxg(accum, item);
    }
}

template<typename T>
__device__ T normalize(const Pool2dOp op, const T item, const size_t num_elements) {
    double num_f64 = num_elements;
    double scale_f64 = 1.0 / num_f64;
    T scale = scale_f64;
    switch(op.kind) {
        case AVG:
            return item * scale;
        case MIN:
            return item;
        case MAX:
            return item;
    }
}

template<typename T>
__device__ T filter(const Pool2dOp op, const T item, const T needle, const T haystack) {
    T zero = 0.0;
    switch(op.kind){
        case AVG:
            return item;
        case MIN:
            return (needle == haystack) ? item : zero;
        case MAX:
            return (needle == haystack) ? item : zero;
    }
}

template<typename T>
__device__ void pool2d_fwd(
    const Pool2dOp op,
    const size_t *inp_strides,
    const size_t *out_strides,
    const T *inp, // 4d (Batch, Channels, Height, Width)
    T *out // 4d (Batch, Channels, HeightOut, WidthOut)
) {
    const size_t numel = op.batch * op.chan * op.h_out * op.w_out;
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        unsigned int idx = i;
        const size_t ow = idx % op.w_out;
        idx /= op.w_out;
        const size_t oh = idx % op.h_out;
        idx /= op.h_out;
        const size_t c = idx % op.chan;
        idx /= op.chan;
        const size_t b = idx % op.batch;
        idx /= op.batch;
        
        T tmp = init(op);
        for(size_t k1 = 0; k1 < op.kernel; k1++) {
            for (size_t k2 = 0; k2 < op.kernel; k2++) {
                const size_t y_plus_p = oh * op.stride + op.dilation * k1;
                if (y_plus_p < op.padding) { continue; }
                const size_t y = y_plus_p - op.padding;
                if (y >= op.h_in) { continue; }
                const size_t x_plus_p = ow * op.stride + op.dilation * k2;
                if (x_plus_p < op.padding) { continue; }
                const size_t x = x_plus_p - op.padding;
                if (x >= op.w_in) { continue; }
    
                auto inp_i = b * inp_strides[0] + c * inp_strides[1] + y * inp_strides[2] + x * inp_strides[3];
                tmp = accum(op, tmp, inp[inp_i]);
            }
        }
    
        out[i] = normalize(op, tmp, op.kernel * op.kernel);
    }
}

template<typename T>
__device__ void pool2d_bwd(
    const Pool2dOp op,
    const size_t *inp_strides,
    const size_t *out_strides,
    const T *inp, // 4d (Batch, Channels, Height, Width)
    T *grad_inp,
    const T *out, // 4d (Batch, Channels, HeightOut, WidthOut)
    const T *grad_out
) {
    const size_t numel = op.batch * op.chan * op.h_in * op.w_in;
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        unsigned int idx = i;
        const size_t x = idx % op.w_in;
        idx /= op.w_in;
        const size_t y = idx % op.h_in;
        idx /= op.h_in;
        const size_t c = idx % op.chan;
        idx /= op.chan;
        const size_t b = idx % op.batch;
        idx /= op.batch;
    
        const T inp_v = inp[i];
    
        T tmp = 0.0;
        for(size_t k1 = 0; k1 < op.kernel; k1++) {
            for (size_t k2 = 0; k2 < op.kernel; k2++) {
                size_t oh = y + op.padding;
                if (oh < op.dilation * k1) { continue; }
                oh -= op.dilation * k1;
                if (oh % op.stride != 0) { continue; }
                oh /= op.stride;
                if (oh >= op.h_out) { continue; }
    
                size_t ow = x + op.padding;
                if (ow < op.dilation * k2) { continue; }
                ow -= op.dilation * k2;
                if (ow % op.stride != 0) { continue; }
                ow /= op.stride;
                if (ow >= op.w_out) { continue; }
    
                auto out_i = b * out_strides[0] + c * out_strides[1] + oh * out_strides[2] + ow * out_strides[3];
                tmp += filter(op, grad_out[out_i], out[out_i], inp_v);
            }
        }
        grad_inp[i] += normalize(op, tmp, op.kernel * op.kernel);
    }
}

#define POOL_OP(TYPENAME, fwd, bwd) \
extern "C" __global__ void fwd( \
    const Pool2dOp op, \
    const size_t *inp_strides, \
    const size_t *out_strides, \
    const TYPENAME *inp, \
    TYPENAME *out \
) { \
    pool2d_fwd(op, inp_strides, out_strides, inp, out); \
} \
extern "C" __global__ void bwd( \
    const Pool2dOp op, \
    const size_t *inp_strides, \
    const size_t *out_strides, \
    const TYPENAME *inp, \
    TYPENAME *grad_inp, \
    const TYPENAME *out, \
    const TYPENAME *grad_out \
) { \
    pool2d_bwd(op, inp_strides, out_strides, inp, grad_inp, out, grad_out); \
}

POOL_OP(__half, pool2d_fwd_f16, pool2d_bwd_f16);
POOL_OP(float, pool2d_fwd_f32, pool2d_bwd_f32);
POOL_OP(double, pool2d_fwd_f64, pool2d_bwd_f64);
