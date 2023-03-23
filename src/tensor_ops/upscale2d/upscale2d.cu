#include "cuda_utils.cuh"

struct Upscale2dOp {
    size_t batch;
    size_t chan;
    size_t h_in;
    size_t h_out;
    size_t w_in;
    size_t w_out;
};

template<typename T>
__device__ void nearest_upscale2d_fwd(
    const Upscale2dOp op,
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

    float h_scale = ((float)(inp_strides[1]/inp_strides[2]))/(out_strides[1]/out_strides[2]);
    float w_scale = ((float)(inp_strides[2]/inp_strides[3]))/(out_strides[2]/out_strides[3]);

    unsigned int idx = i;
    const size_t ow = idx % op.w_out;
    idx /= op.w_out;
    const size_t oh = idx % op.h_out;
    idx /= op.h_out;
    const size_t c = idx % op.chan;
    idx /= op.chan;
    const size_t b = idx % op.batch;
    idx /= op.batch;

    size_t ih = h_scale * oh;
    size_t iw = w_scale * ow;

    size_t inp_i = b * inp_strides[0] + c * inp_strides[1] + ih * inp_strides[2] + iw * inp_strides[3];
    
    out[i] = inp[inp_i];
}

template<typename T>
__device__ void nearest_upscale2d_bwd(
    const Upscale2dOp op,
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

    float h_scale = ((float)(inp_strides[1]/inp_strides[2]))/(out_strides[1]/out_strides[2]);
    float w_scale = ((float)(inp_strides[2]/inp_strides[3]))/(out_strides[2]/out_strides[3]);

    unsigned int idx = i;
    const size_t x = idx % op.w_in;
    idx /= op.w_in;
    const size_t y = idx % op.h_in;
    idx /= op.h_in;
    const size_t c = idx % op.chan;
    idx /= op.chan;
    const size_t b = idx % op.batch;
    idx /= op.batch;

    // Probably isn't efficient, but it works
    size_t oh_s = 0;
    size_t ow_s = 0;
    size_t oh_e = op.h_out;
    size_t ow_e = op.w_out;
    while (oh_s*h_scale < y) {
        oh_s++;
    }
    while (ow_s*w_scale < x) {
        ow_s++;
    }
    while (oh_e*h_scale >= y+1) {
        oh_e--;
    }
    while (ow_e*w_scale >= x+1) {
        ow_e--;
    }

    for (int oh = oh_s; oh <= oh_e; oh++) {
        for (int ow = ow_s; ow <= ow_e; ow++) {
            size_t out_i = b * out_strides[0] + c * out_strides[1] + oh * out_strides[2] + ow * out_strides[3];
            grad_inp[i] += grad_out[out_i];
        }
    }
}

#define UPSCALE_OP(TYPENAME, fwd, bwd, fwd_FN, bwd_FN) \
extern "C" __global__ void fwd( \
    const Upscale2dOp op, \
    const size_t *inp_strides, \
    const size_t *out_strides, \
    const TYPENAME *inp, \
    TYPENAME *out \
) { \
    fwd_FN(op, inp_strides, out_strides, inp, out); \
} \
extern "C" __global__ void bwd( \
    const Upscale2dOp op, \
    const size_t *inp_strides, \
    const size_t *out_strides, \
    const TYPENAME *inp, \
    TYPENAME *grad_inp, \
    const TYPENAME *out, \
    const TYPENAME *grad_out \
) { \
    bwd_FN(op, inp_strides, out_strides, inp, grad_inp, out, grad_out); \
}

UPSCALE_OP(
    float,
    nearest_upscale2d_fwd_f32, nearest_upscale2d_bwd_f32,
    nearest_upscale2d_fwd, nearest_upscale2d_bwd
);
UPSCALE_OP(
    double,
    nearest_upscale2d_fwd_f64, nearest_upscale2d_bwd_f64,
    nearest_upscale2d_fwd, nearest_upscale2d_bwd
);