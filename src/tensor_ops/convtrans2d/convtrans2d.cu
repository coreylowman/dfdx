#include "cuda_fp16.h"

struct Conv2DOp {
    size_t kernel;
    size_t stride;
    size_t padding;
    size_t dilation;
    size_t groups;
    size_t batch;
    size_t chan_in;
    size_t chan_out;
    size_t h_in;
    size_t h_out;
    size_t w_in;
    size_t w_out;
};

template<typename T>
__device__ void unfold_input_into_patches(
    const Conv2DOp op,
    const T *image, // 4d (Batch, Groups * Channels, Height, Width)
    const size_t *strides, // 4d image strides
    T *patches // 6d (Batch, Groups * Channels, KernelSize, KernelSize, HeightOut, WidthOut)
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= op.batch * op.chan_in * op.h_out * op.w_out) {
        return;
    }

    unsigned int idx = i;
    const size_t ow = idx % op.w_out;
    idx /= op.w_out;
    const size_t oh = idx % op.h_out;
    idx /= op.h_out;
    const size_t c = idx % op.chan_in;
    idx /= op.chan_in;
    const size_t b = idx % op.batch;
    idx /= op.batch;

    image += b * strides[0] + c * strides[1];
    patches += oh * op.w_out + ow;
    patches += c * (op.kernel * op.kernel * op.h_out * op.w_out);
    patches += b * (op.chan_in * op.kernel * op.kernel * op.h_out * op.w_out);

    T zero = 0.0;

    for (int k1 = 0;k1 < op.kernel;k1++) {
        const size_t y_ks = oh + op.padding;
        const size_t y_s = y_ks - op.dilation * k1;
        const size_t y = y_s / op.stride;
        const bool k1_invalid = (y_ks < op.dilation * k1 || y_s % op.stride != 0 || y >= op.h_in);
        for (int k2 = 0;k2 < op.kernel;k2++) {
            const size_t x_ks = ow + op.padding;
            const size_t x_s = x_ks - op.dilation * k2;
            const size_t x = x_s / op.stride;
        
            const bool invalid = k1_invalid || (x_ks < op.dilation * k2 || x_s % op.stride != 0 || x >= op.w_in);
            *patches = invalid ? zero : image[y * strides[2] + x * strides[3]];
            patches += op.h_out * op.w_out;
        }
    }
}

template<typename T>
__device__ void unfold_output_into_patches(
    const Conv2DOp op,
    const T *image_out, // 4d (Batch, ChanOut, HeightOut, WidthOut)
    T *patches // 6d (Batch, ChanOut, KernelSize, KernelSize, HeightIn, WidthIn)
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= op.batch * op.chan_out * op.h_in * op.w_in) {
        return;
    }

    unsigned int idx = i;
    const size_t x = idx % op.w_in;
    idx /= op.w_in;
    const size_t y = idx % op.h_in;
    idx /= op.h_in;
    const size_t o = idx % op.chan_out;
    idx /= op.chan_out;
    const size_t b = idx % op.batch;

    image_out += b * (op.chan_out * op.h_out * op.w_out) + o * (op.h_out * op.w_out);
    patches += y * op.w_in + x;
    patches += o * (op.kernel * op.kernel * op.h_in * op.w_in);
    patches += b * (op.chan_out * op.kernel * op.kernel * op.h_in * op.w_in);

    T zero = 0.0;

    for (int k1 = 0;k1 < op.kernel;k1++) {
        const size_t oh = y * op.stride + op.dilation * k1 - op.padding;
        for (int k2 = 0;k2 < op.kernel;k2++) {
            const size_t ow = x * op.stride + op.dilation * k2 - op.padding;
            *patches = (oh >= op.h_out || ow >= op.w_out) ? zero : image_out[oh * op.w_out + ow];
            patches += op.h_in * op.w_in;
        }
    }
}

template<typename T>
__device__ void transpose_filters(
    const Conv2DOp op,
    const T *filters, // 4d (ChanIn, ChanOut/Groups, KernelSize, KernelSize)
    const size_t *strides, // 4d filters strides
    T *filters_tr // 5d (Groups, ChanOut/Groups, ChanIn/Groups, KernelSize, KernelSize)
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t o_per_g = op.chan_out / op.groups;
    const size_t c_per_g = op.chan_in / op.groups;
    if (i >= op.groups * o_per_g * c_per_g * op.kernel * op.kernel) {
        return;
    }


    unsigned int idx = i;
    const size_t k2 = idx % op.kernel;
    idx /= op.kernel;
    const size_t k1 = idx % op.kernel;
    idx /= op.kernel;
    const size_t og = idx % o_per_g;
    idx /= o_per_g;
    const size_t c = idx % op.chan_in;
    const size_t cg = c % c_per_g;
    const size_t g = c / c_per_g;

    auto i_no = c * strides[0] + og * strides[1] + k1 * strides[2] + k2 * strides[3];
    filters_tr += k2;
    filters_tr += k1 * op.kernel;
    filters_tr += cg * (op.kernel * op.kernel);
    filters_tr += og * (c_per_g * op.kernel * op.kernel);
    filters_tr += g * (o_per_g * * c_per_g * op.kernel * op.kernel);
    *filters_tr = filters[i_no];
}

#define CONV_OP(TYPENAME, UNFOLD_INPUT, UNFOLD_OUTPUT, TR_FILTERS, SUM_TR_FILTERS) \
extern "C" __global__ void UNFOLD_INPUT( \
    const Conv2DOp op, \
    const TYPENAME *image, \
    const size_t *strides, \
    TYPENAME *patches \
) { \
    unfold_input_into_patches(op, image, strides, patches); \
} \
extern "C" __global__ void UNFOLD_OUTPUT( \
    const Conv2DOp op, \
    const TYPENAME *image_out, \
    TYPENAME *patches \
) { \
    unfold_output_into_patches(op, image_out, patches); \
} \
extern "C" __global__ void TR_FILTERS( \
    const Conv2DOp op, \
    const TYPENAME *filters, \
    const size_t *strides, \
    TYPENAME *filters_tr \
) { \
    transpose_filters(op, filters, strides, filters_tr); \
}

CONV_OP(
    __half,
    unfold_input_into_patches_f16,
    unfold_output_into_patches_f16,
    transpose_filters_f16
);
CONV_OP(
    float,
    unfold_input_into_patches_f32,
    unfold_output_into_patches_f32,
    transpose_filters_f32
);
CONV_OP(
    double,
    unfold_input_into_patches_f64,
    unfold_output_into_patches_f64,
    transpose_filters_f64
);