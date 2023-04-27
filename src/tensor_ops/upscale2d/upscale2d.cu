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
    const T *inp, // 4d (Batch, Channels, Height, Width)
    T *out // 4d (Batch, Channels, HeightOut, WidthOut)
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= op.batch * op.chan * op.h_out * op.w_out) {
        return;
    }

    float h_scale = static_cast<float>(op.h_in)/static_cast<float>(op.h_out);
    float w_scale = static_cast<float>(op.w_in)/static_cast<float>(op.w_out);

    unsigned int idx = i;
    const size_t ow = idx % op.w_out;
    idx /= op.w_out;
    const size_t oh = idx % op.h_out;
    idx /= op.h_out;
    const size_t c = idx % op.chan;
    idx /= op.chan;
    const size_t b = idx % op.batch;

    size_t ih = min(static_cast<size_t>(h_scale * oh), op.h_in - 1);
    size_t iw = min(static_cast<size_t>(w_scale * ow), op.w_in - 1);

    size_t inp_i = b * inp_strides[0] + c * inp_strides[1] + ih * inp_strides[2] + iw * inp_strides[3];
    
    out[i] = inp[inp_i];
}

template<typename T>
__device__ void nearest_upscale2d_bwd(
    const Upscale2dOp op,
    const size_t *inp_strides,
    T *grad_inp,
    const T *grad_out // 4d (Batch, Channels, HeightOut, WidthOut)
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= op.batch * op.chan * op.h_out * op.w_out) {
        return;
    }

    float h_scale = static_cast<float>(op.h_in)/static_cast<float>(op.h_out);
    float w_scale = static_cast<float>(op.w_in)/static_cast<float>(op.w_out);

    unsigned int idx = i;
    const size_t ow = idx % op.w_out;
    idx /= op.w_out;
    const size_t oh = idx % op.h_out;
    idx /= op.h_out;
    const size_t c = idx % op.chan;
    idx /= op.chan;
    const size_t b = idx % op.batch;

    size_t ih = min(static_cast<size_t>(h_scale * oh), op.h_in - 1);
    size_t iw = min(static_cast<size_t>(w_scale * ow), op.w_in - 1);

    size_t inp_i = b * inp_strides[0] + c * inp_strides[1] + ih * inp_strides[2] + iw * inp_strides[3];
    atomicAdd(grad_inp + inp_i, grad_out[i]);
}

template<typename T>
__device__ void bilinear_upscale2d_fwd(
    const Upscale2dOp op,
    const size_t *inp_strides,
    const T *inp, // 4d (Batch, Channels, Height, Width)
    T *out // 4d (Batch, Channels, HeightOut, WidthOut)
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= op.batch * op.chan * op.h_out * op.w_out) {
        return;
    }

    float h_scale = ((float)op.h_in-1)/(op.h_out-1);
    float w_scale = ((float)op.w_in-1)/(op.w_out-1);

    unsigned int idx = i;
    const size_t ow = idx % op.w_out;
    idx /= op.w_out;
    const size_t oh = idx % op.h_out;
    idx /= op.h_out;
    const size_t c = idx % op.chan;
    idx /= op.chan;
    const size_t b = idx % op.batch;

    size_t y0 = min(static_cast<size_t>(h_scale * oh), op.h_in - 1);
    size_t y1 = min(y0 + 1, op.h_in - 1);
    size_t x0 = min(static_cast<size_t>(w_scale * ow), op.w_in - 1);
    size_t x1 = min(x0 + 1, op.w_in - 1);

    T hs = h_scale * oh - y0;
    T ws = w_scale * ow - x0;

    inp += b * inp_strides[0] + c * inp_strides[1];

    T one = 1.0;

    T ll = inp[y0 * inp_strides[2] + x0 * inp_strides[3]] * (one-hs) * (one-ws);
    T lh = inp[y0 * inp_strides[2] + x1 * inp_strides[3]] * (one-hs) * ws;
    T hl = inp[y1 * inp_strides[2] + x0 * inp_strides[3]] * hs * (one-ws);
    T hh = inp[y1 * inp_strides[2] + x1 * inp_strides[3]] * hs * ws;

    out[i] = ll + lh + hl + hh;
}

template<typename T>
__device__ void bilinear_upscale2d_bwd(
    const Upscale2dOp op,
    const size_t *inp_strides,
    T *grad_inp, // 4d (Batch, Channels, Height, Width)
    const T *grad_out // 4d (Batch, Channels, HeightOut, WidthOut)
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= op.batch * op.chan * op.h_out * op.w_out) {
        return;
    }

    float h_scale = ((float)op.h_in-1)/(op.h_out-1);
    float w_scale = ((float)op.w_in-1)/(op.w_out-1);

    unsigned int idx = i;
    const size_t ow = idx % op.w_out;
    idx /= op.w_out;
    const size_t oh = idx % op.h_out;
    idx /= op.h_out;
    const size_t c = idx % op.chan;
    idx /= op.chan;
    const size_t b = idx % op.batch;

    size_t y0 = min(static_cast<size_t>(h_scale * oh), op.h_in - 1);
    size_t y1 = min(y0 + 1, op.h_in - 1);
    size_t x0 = min(static_cast<size_t>(w_scale * ow), op.w_in - 1);
    size_t x1 = min(x0 + 1, op.w_in - 1);

    T hs = h_scale * oh - y0;
    T ws = w_scale * ow - x0;

    T go = grad_out[i];

    grad_inp += b * inp_strides[0] + c * inp_strides[1];

    const T one = 1.0;

    atomicAdd(grad_inp + y0 * inp_strides[2] + x0 * inp_strides[3], go * (one-hs) * (one-ws));
    atomicAdd(grad_inp + y0 * inp_strides[2] + x1 * inp_strides[3], go * (one-hs) * ws);
    atomicAdd(grad_inp + y1 * inp_strides[2] + x0 * inp_strides[3], go * hs * (one-ws));
    atomicAdd(grad_inp + y1 * inp_strides[2] + x1 * inp_strides[3], go * hs * ws);
}

#define UPSCALE_OP(TYPENAME, fwd, bwd, fwd_FN, bwd_FN) \
extern "C" __global__ void fwd( \
    const Upscale2dOp op, \
    const size_t *inp_strides, \
    const TYPENAME *inp, \
    TYPENAME *out \
) { \
    fwd_FN(op, inp_strides, inp, out); \
} \
extern "C" __global__ void bwd( \
    const Upscale2dOp op, \
    const size_t *inp_strides, \
    TYPENAME *grad_inp, \
    const TYPENAME *grad_out \
) { \
    bwd_FN(op, inp_strides, grad_inp, grad_out); \
}

UPSCALE_OP(
    __half,
    nearest_upscale2d_fwd_f16, nearest_upscale2d_bwd_f16,
    nearest_upscale2d_fwd, nearest_upscale2d_bwd
);
UPSCALE_OP(
    __half,
    bilinear_upscale2d_fwd_f16, bilinear_upscale2d_bwd_f16,
    bilinear_upscale2d_fwd, bilinear_upscale2d_bwd
);

UPSCALE_OP(
    float,
    nearest_upscale2d_fwd_f32, nearest_upscale2d_bwd_f32,
    nearest_upscale2d_fwd, nearest_upscale2d_bwd
);
UPSCALE_OP(
    float,
    bilinear_upscale2d_fwd_f32, bilinear_upscale2d_bwd_f32,
    bilinear_upscale2d_fwd, bilinear_upscale2d_bwd
);
UPSCALE_OP(
    double,
    nearest_upscale2d_fwd_f64, nearest_upscale2d_bwd_f64,
    nearest_upscale2d_fwd, nearest_upscale2d_bwd
);
UPSCALE_OP(
    double,
    bilinear_upscale2d_fwd_f64, bilinear_upscale2d_bwd_f64,
    bilinear_upscale2d_fwd, bilinear_upscale2d_bwd
);