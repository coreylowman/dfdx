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
    if (i >= op.batch * op.chan * op.h_out * op.w_out) {
        return;
    }

    float h_scale = ((float)inp_sizes[2])/((float)out_sizes[2]);
    float w_scale = ((float)inp_sizes[3])/((float)out_sizes[3]);

    unsigned int idx = i;
    const size_t ow = idx % op.w_out;
    idx /= op.w_out;
    const size_t oh = idx % op.h_out;
    idx /= op.h_out;
    const size_t c = idx % op.chan;
    idx /= op.chan;
    const size_t b = idx % op.batch;

    size_t ih = min(h_scale * oh, out_sizes[2] - 1);
    size_t iw = min(w_scale * ow, out_sizes[3] - 1);

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
    if (i >= op.batch * op.chan * op.h_out * op.w_out) {
        return;
    }

    float h_scale = ((float)inp_sizes[2])/((float)out_sizes[2]);
    float w_scale = ((float)inp_sizes[3])/((float)out_sizes[3]);

    unsigned int idx = i;
    const size_t ow = idx % op.w_out;
    idx /= op.w_out;
    const size_t oh = idx % op.h_out;
    idx /= op.h_out;
    const size_t c = idx % op.chan;
    idx /= op.chan;
    const size_t b = idx % op.batch;

    size_t ih = min(h_scale * oh, out_sizes[2] - 1);
    size_t iw = min(w_scale * ow, out_sizes[3] - 1);

    size_t inp_i = b * inp_strides[0] + c * inp_strides[1] + ih * inp_strides[2] + iw * inp_strides[3];
    atomicAdd(grad_inp + inp_i, grad_out[i]);
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
    if (i >= op.batch * op.chan * op.h_out * op.w_out) {
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

    size_t y0 = min(h_scale * oh, out_sizes[2] - 1);
    size_t y1 = min(y0 + 1, out_sizes[2] - 1);
    size_t x0 = min(w_scale * ow, out_sizes[3] - 1);
    size_t x1 = min(x0 + 1, out_sizes[3] - 1);

    T hs = h_scale * oh - y0;
    T ws = w_scale * ow - x0;

    inp += b * inp_strides[0] + c * inp_strides[1];

    T ll = inp[y0 * inp_strides[2] + x0 * inp_strides[3]] * (1-hs) * (1-ws);
    T lh = inp[y0 * inp_strides[2] + x1 * inp_strides[3]] * (1-hs) * ws;
    T hl = inp[y1 * inp_strides[2] + x0 * inp_strides[3]] * hs * (1-ws);
    T hh = inp[y1 * inp_strides[2] + x1 * inp_strides[3]] * hs * ws;

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
    if (i >= op.batch * op.chan * op.h_out * op.w_out) {
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

    size_t y0 = min(h_scale * oh, out_sizes[2] - 1);
    size_t y1 = min(y0 + 1, out_sizes[2] - 1);
    size_t x0 = min(w_scale * ow, out_sizes[3] - 1);
    size_t x1 = min(x0 + 1, out_sizes[3] - 1);

    T hs = h_scale * oh - y0;
    T ws = w_scale * ow - x0;

    T go = grad_out[i];

    inp += b * inp_strides[0] + c * inp_strides[1];

    atomicAdd(inp + y0 * inp_strides[2] + x0 * inp_strides[3]], go * (1-hs) * (1-ws));
    atomicAdd(inp + y0 * inp_strides[2] + x1 * inp_strides[3]], go * (1-hs) * ws);
    atomicAdd(inp + y1 * inp_strides[2] + x0 * inp_strides[3]], go * hs * (1-ws));
    atomicAdd(inp + y1 * inp_strides[2] + x1 * inp_strides[3]], go * hs * ws);
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