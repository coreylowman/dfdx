#include "cuda_utils.cuh"

struct Pool2dOp {
    size_t kernel;
    size_t stride;
    size_t padding;
    size_t batch;
    size_t chan;
    size_t h_in;
    size_t h_out;
    size_t w_in;
    size_t w_out;
};

template<typename T>
__device__ void avg_pool2d_fwd(
    const Pool2dOp op,
    const size_t *inp_strides,
    const size_t *out_strides,
    const T *inp, // 4d (Batch, Channels, Height, Width)
    T *out // 4d (Batch, Channels, HeightOut, WidthOut)
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t numel = op.batch * op.chan * op.h_out * op.w_out;
    if (i >= numel) {
        return;
    }

    unsigned int idx = i;
    const size_t ow = idx % op.w_out;
    idx /= op.w_out;
    const size_t oh = idx % op.h_out;
    idx /= op.h_out;
    const size_t c = idx % op.chan;
    idx /= op.chan;
    const size_t b = idx % op.batch;
    idx /= op.batch;
    
    T tmp = 0.0;
    for(size_t k1 = 0; k1 < op.kernel; k1++) {
        for (size_t k2 = 0; k2 < op.kernel; k2++) {
            const size_t y_plus_p = oh * op.stride + k1;
            if (y_plus_p < op.padding) { continue; }
            const size_t y = y_plus_p - op.padding;
            if (y >= op.h_in) { continue; }
            const size_t x_plus_p = ow * op.stride + k2;
            if (x_plus_p < op.padding) { continue; }
            const size_t x = x_plus_p - op.padding;
            if (x >= op.w_in) { continue; }

            auto inp_i = b * inp_strides[0] + c * inp_strides[1] + y * inp_strides[2] + x * inp_strides[3];
            tmp += inp[inp_i];
        }
    }

    tmp /= static_cast<T>(op.kernel * op.kernel);
    out[i] = tmp;
}

template<typename T>
__device__ void avg_pool2d_bwd(
    const Pool2dOp op,
    const size_t *inp_strides,
    const size_t *out_strides,
    const T *inp, // 4d (Batch, Channels, Height, Width)
    T *grad_inp,
    const T *out, // 4d (Batch, Channels, HeightOut, WidthOut)
    const T *grad_out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t numel = op.batch * op.chan * op.h_in * op.w_in;
    if (i >= numel) {
        return;
    }

    unsigned int idx = i;
    const size_t x = idx % op.w_in;
    idx /= op.w_in;
    const size_t y = idx % op.h_in;
    idx /= op.h_in;
    const size_t c = idx % op.chan;
    idx /= op.chan;
    const size_t b = idx % op.batch;
    idx /= op.batch;

    T tmp = 0.0;
    for(size_t k1 = 0; k1 < op.kernel; k1++) {
        for (size_t k2 = 0; k2 < op.kernel; k2++) {
            size_t oh = y + op.padding;
            if (oh < k1) { continue; }
            oh -= k1;
            if (oh % op.stride != 0) { continue; }
            oh /= op.stride;
            if (oh >= op.h_out) { continue; }

            size_t ow = x + op.padding;
            if (ow < k2) { continue; }
            ow -= k2;
            if (ow % op.stride != 0) { continue; }
            ow /= op.stride;
            if (ow >= op.w_out) { continue; }

            auto out_i = b * out_strides[0] + c * out_strides[1] + oh * out_strides[2] + ow * out_strides[3];
            tmp += grad_out[out_i];
        }
    }

    grad_inp[i] += tmp / static_cast<T>(op.kernel * op.kernel);
}

template<typename T>
__device__ void max_pool2d_fwd(
    const Pool2dOp op,
    const size_t *inp_strides,
    const size_t *out_strides,
    const T *inp, // 4d (Batch, Channels, Height, Width)
    T *out // 4d (Batch, Channels, HeightOut, WidthOut)
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t numel = op.batch * op.chan * op.h_out * op.w_out;
    if (i >= numel) {
        return;
    }

    unsigned int idx = i;
    const size_t ow = idx % op.w_out;
    idx /= op.w_out;
    const size_t oh = idx % op.h_out;
    idx /= op.h_out;
    const size_t c = idx % op.chan;
    idx /= op.chan;
    const size_t b = idx % op.batch;
    idx /= op.batch;

    T tmp = -INFINITY;
    for(size_t k1 = 0; k1 < op.kernel; k1++) {
        for (size_t k2 = 0; k2 < op.kernel; k2++) {
            const size_t y_plus_p = oh * op.stride + k1;
            if (y_plus_p < op.padding) { continue; }
            const size_t y = y_plus_p - op.padding;
            if (y >= op.h_in) { continue; }
            const size_t x_plus_p = ow * op.stride + k2;
            if (x_plus_p < op.padding) { continue; }
            const size_t x = x_plus_p - op.padding;
            if (x >= op.w_in) { continue; }

            auto inp_i = b * inp_strides[0] + c * inp_strides[1] + y * inp_strides[2] + x * inp_strides[3];
            tmp = maxg(tmp, inp[inp_i]);
        }
    }

    out[i] = tmp;
}

template<typename T>
__device__ void max_pool2d_bwd(
    const Pool2dOp op,
    const size_t *inp_strides,
    const size_t *out_strides,
    const T *inp, // 4d (Batch, Channels, Height, Width)
    T *grad_inp,
    const T *out, // 4d (Batch, Channels, HeightOut, WidthOut)
    const T *grad_out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t numel = op.batch * op.chan * op.h_in * op.w_in;
    if (i >= numel) {
        return;
    }

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
            if (oh < k1) { continue; }
            oh -= k1;
            if (oh % op.stride != 0) { continue; }
            oh /= op.stride;
            if (oh >= op.h_out) { continue; }

            size_t ow = x + op.padding;
            if (ow < k2) { continue; }
            ow -= k2;
            if (ow % op.stride != 0) { continue; }
            ow /= op.stride;
            if (ow >= op.w_out) { continue; }

            auto out_i = b * out_strides[0] + c * out_strides[1] + oh * out_strides[2] + ow * out_strides[3];

            if (out[out_i] == inp_v) {
                tmp += grad_out[out_i];
            }
        }
    }

    grad_inp[i] += tmp;
}

template<typename T>
__device__ void min_pool2d_fwd(
    const Pool2dOp op,
    const size_t *inp_strides,
    const size_t *out_strides,
    const T *inp, // 4d (Batch, Channels, Height, Width)
    T *out // 4d (Batch, Channels, HeightOut, WidthOut)
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t numel = op.batch * op.chan * op.h_out * op.w_out;
    if (i >= numel) {
        return;
    }

    unsigned int idx = i;
    const size_t ow = idx % op.w_out;
    idx /= op.w_out;
    const size_t oh = idx % op.h_out;
    idx /= op.h_out;
    const size_t c = idx % op.chan;
    idx /= op.chan;
    const size_t b = idx % op.batch;
    idx /= op.batch;

    T tmp = INFINITY;
    for(size_t k1 = 0; k1 < op.kernel; k1++) {
        for (size_t k2 = 0; k2 < op.kernel; k2++) {
            const size_t y_plus_p = oh * op.stride + k1;
            if (y_plus_p < op.padding) { continue; }
            const size_t y = y_plus_p - op.padding;
            if (y >= op.h_in) { continue; }
            const size_t x_plus_p = ow * op.stride + k2;
            if (x_plus_p < op.padding) { continue; }
            const size_t x = x_plus_p - op.padding;
            if (x >= op.w_in) { continue; }

            auto inp_i = b * inp_strides[0] + c * inp_strides[1] + y * inp_strides[2] + x * inp_strides[3];
            tmp = ming(tmp, inp[inp_i]);
        }
    }

    out[i] = tmp;
}

template<typename T>
__device__ void min_pool2d_bwd(
    const Pool2dOp op,
    const size_t *inp_strides,
    const size_t *out_strides,
    const T *inp, // 4d (Batch, Channels, Height, Width)
    T *grad_inp,
    const T *out, // 4d (Batch, Channels, HeightOut, WidthOut)
    const T *grad_out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t numel = op.batch * op.chan * op.h_in * op.w_in;
    if (i >= numel) {
        return;
    }

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
            if (oh < k1) { continue; }
            oh -= k1;
            if (oh % op.stride != 0) { continue; }
            oh /= op.stride;
            if (oh >= op.h_out) { continue; }

            size_t ow = x + op.padding;
            if (ow < k2) { continue; }
            ow -= k2;
            if (ow % op.stride != 0) { continue; }
            ow /= op.stride;
            if (ow >= op.w_out) { continue; }

            auto out_i = b * out_strides[0] + c * out_strides[1] + oh * out_strides[2] + ow * out_strides[3];

            if (out[out_i] == inp_v) {
                tmp += grad_out[out_i];
            }
        }
    }

    grad_inp[i] += tmp;
}

#define POOL_OP(TYPENAME, fwd, bwd, fwd_FN, bwd_FN) \
extern "C" __global__ void fwd( \
    const Pool2dOp op, \
    const size_t *inp_strides, \
    const size_t *out_strides, \
    const TYPENAME *inp, \
    TYPENAME *out \
) { \
    fwd_FN(op, inp_strides, out_strides, inp, out); \
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
    bwd_FN(op, inp_strides, out_strides, inp, grad_inp, out, grad_out); \
}

POOL_OP(
    float,
    avg_pool2d_fwd_f32, avg_pool2d_bwd_f32,
    avg_pool2d_fwd, avg_pool2d_bwd
);
POOL_OP(
    float,
    min_pool2d_fwd_f32, min_pool2d_bwd_f32,
    min_pool2d_fwd, min_pool2d_bwd
);
POOL_OP(
    float,
    max_pool2d_fwd_f32, max_pool2d_bwd_f32,
    max_pool2d_fwd, max_pool2d_bwd
);

POOL_OP(
    double,
    avg_pool2d_fwd_f64, avg_pool2d_bwd_f64,
    avg_pool2d_fwd, avg_pool2d_bwd
);
POOL_OP(
    double,
    min_pool2d_fwd_f64, min_pool2d_bwd_f64,
    min_pool2d_fwd, min_pool2d_bwd
);
POOL_OP(
    double,
    max_pool2d_fwd_f64, max_pool2d_bwd_f64,
    max_pool2d_fwd, max_pool2d_bwd
);
