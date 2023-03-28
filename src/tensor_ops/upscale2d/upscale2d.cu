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
    const size_t *inp_sizes,
    const size_t *out_strides,
    const size_t *out_sizes,
    const T *inp, // 4d (Batch, Channels, Height, Width)
    T *out // 4d (Batch, Channels, HeightOut, WidthOut)
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t numel = op.batch * op.chan * op.h_out * op.w_out;
    if (i >= numel) {
        return;
    }

    float h_scale = ((float)inp_sizes[2])/out_sizes[2];
    float w_scale = ((float)inp_sizes[3])/out_sizes[3];

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
    const size_t *inp_sizes,
    const size_t *out_strides,
    const size_t *out_sizes,
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

    float h_scale = ((float)inp_sizes[2])/out_sizes[2];
    float w_scale = ((float)inp_sizes[3])/out_sizes[3];

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
    size_t oh_e = op.h_out-1;
    size_t ow_e = op.w_out-1;
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

template<typename T>
__device__ void bilinear_upscale2d_fwd(
    const Upscale2dOp op,
    const size_t *inp_strides,
    const size_t *inp_sizes,
    const size_t *out_strides,
    const size_t *out_sizes,
    const T *inp, // 4d (Batch, Channels, Height, Width)
    T *out // 4d (Batch, Channels, HeightOut, WidthOut)
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t numel = op.batch * op.chan * op.h_out * op.w_out;
    if (i >= numel) {
        return;
    }

    float h_scale = ((float)inp_sizes[2]-1)/(out_sizes[2]-1);
    float w_scale = ((float)inp_sizes[3]-1)/(out_sizes[3]-1);

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

    T hs = h_scale * oh - ih;
    T ws = w_scale * ow - iw;

    T ll = inp[b * inp_strides[0] + c * inp_strides[1] + ih * inp_strides[2] + iw * inp_strides[3]] * (1-hs) * (1-ws);
    T lh = ws != 0 ? inp[b * inp_strides[0] + c * inp_strides[1] + ih * inp_strides[2] + (iw+1) * inp_strides[3]] * (1-hs) * ws : 0;
    T hl = hs != 0 ? inp[b * inp_strides[0] + c * inp_strides[1] + (ih+1) * inp_strides[2] + iw * inp_strides[3]] * hs * (1-ws) : 0;
    T hh = hs != 0 && ws != 0 ? inp[b * inp_strides[0] + c * inp_strides[1] + (ih+1) * inp_strides[2] + (iw+1) * inp_strides[3]] * hs * ws : 0;

    out[i] = ll + lh + hl + hh;
}

template<typename T>
__device__ void bilinear_upscale2d_bwd(
    const Upscale2dOp op,
    const size_t *inp_strides,
    const size_t *inp_sizes,
    const size_t *out_strides,
    const size_t *out_sizes,
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

    float h_scale = ((float)inp_sizes[2]-1)/(out_sizes[2]-1);
    float w_scale = ((float)inp_sizes[3]-1)/(out_sizes[3]-1);

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
    size_t oh_e = op.h_out-1;
    size_t ow_e = op.w_out-1;
    while (ceil(oh_s*h_scale) < y) {
        oh_s++;
    }
    while (ceil(ow_s*w_scale) < x) {
        ow_s++;
    }
    while (floor(oh_e*h_scale) > y) {
        oh_e--;
    }
    while (floor(ow_e*w_scale) > x) {
        ow_e--;
    }

    for (int oh = oh_s; oh <= oh_e; oh++) {
        for (int ow = ow_s; ow <= ow_e; ow++) {
            size_t out_i = b * out_strides[0] + c * out_strides[1] + oh * out_strides[2] + ow * out_strides[3];

            T hs = abs(h_scale * oh - y);
            T ws = abs(w_scale * ow - x);

            grad_inp[i] += grad_out[out_i] * (1-hs)*(1-ws);
        }
    }
}

#define UPSCALE_OP(TYPENAME, fwd, bwd, fwd_FN, bwd_FN) \
extern "C" __global__ void fwd( \
    const Upscale2dOp op, \
    const size_t *inp_strides, \
    const size_t *inp_sizes, \
    const size_t *out_strides, \
    const size_t *out_sizes, \
    const TYPENAME *inp, \
    TYPENAME *out \
) { \
    fwd_FN(op, inp_strides, inp_sizes, out_strides, out_sizes, inp, out); \
} \
extern "C" __global__ void bwd( \
    const Upscale2dOp op, \
    const size_t *inp_strides, \
    const size_t *inp_sizes, \
    const size_t *out_strides, \
    const size_t *out_sizes, \
    const TYPENAME *inp, \
    TYPENAME *grad_inp, \
    const TYPENAME *out, \
    const TYPENAME *grad_out \
) { \
    bwd_FN(op, inp_strides, inp_sizes, out_strides, out_sizes, inp, grad_inp, out, grad_out); \
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
UPSCALE_OP(
    float,
    bilinear_upscale2d_fwd_f32, bilinear_upscale2d_bwd_f32,
    bilinear_upscale2d_fwd, bilinear_upscale2d_bwd
);
UPSCALE_OP(
    double,
    bilinear_upscale2d_fwd_f64, bilinear_upscale2d_bwd_f64,
    bilinear_upscale2d_fwd, bilinear_upscale2d_bwd
);