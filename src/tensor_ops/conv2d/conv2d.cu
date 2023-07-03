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
    const size_t n = op.batch * op.groups * op.chan_in * op.h_out * op.w_out;
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        unsigned int idx = i;
        const size_t ow = idx % op.w_out;
        idx /= op.w_out;
        const size_t oh = idx % op.h_out;
        idx /= op.h_out;
        const size_t c = idx % (op.chan_in * op.groups);
        idx /= (op.chan_in * op.groups);
        const size_t b = idx % op.batch;
    
        const T *image_i = image + b * strides[0] + c * strides[1];
        T *patches_i = patches + oh * op.w_out + ow;
        patches_i += c * (op.kernel * op.kernel * op.h_out * op.w_out);
        patches_i += b * (op.groups * op.chan_in * op.kernel * op.kernel * op.h_out * op.w_out);
    
        T zero = 0.0;
    
        for (int k1 = 0;k1 < op.kernel;k1++) {
            const size_t y = oh * op.stride + op.dilation * k1 - op.padding;
            for (int k2 = 0;k2 < op.kernel;k2++) {
                const size_t x = ow * op.stride + op.dilation * k2 - op.padding;
                *patches_i = (y >= op.h_in || x >= op.w_in) ? zero : image[y * strides[2] + x * strides[3]];
                patches_i += op.h_out * op.w_out;
            }
        }
    }
}

template<typename T>
__device__ void unfold_output_into_patches(
    const Conv2DOp op,
    const T *image_out, // 4d (Batch, ChanOut, HeightOut, WidthOut)
    T *patches // 6d (Batch, ChanOut, KernelSize, KernelSize, HeightIn, WidthIn)
) {
    const size_t n = op.batch * op.chan_out * op.h_in * op.w_in;
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        unsigned int idx = i;
        const size_t x = idx % op.w_in;
        idx /= op.w_in;
        const size_t y = idx % op.h_in;
        idx /= op.h_in;
        const size_t o = idx % op.chan_out;
        idx /= op.chan_out;
        const size_t b = idx % op.batch;
    
        const T *image_i = image_out + b * (op.chan_out * op.h_out * op.w_out) + o * (op.h_out * op.w_out);
        T *patches_i = patches + y * op.w_in + x;
        patches_i += o * (op.kernel * op.kernel * op.h_in * op.w_in);
        patches_i += b * (op.chan_out * op.kernel * op.kernel * op.h_in * op.w_in);
    
        T zero = 0.0;
    
        for (int k1 = 0;k1 < op.kernel;k1++) {
            const size_t oh_ks = y + op.padding;
            const size_t oh_s = oh_ks - op.dilation * k1;
            const size_t oh = oh_s / op.stride;
            const bool k1_invalid = (oh_ks < op.dilation * k1 || oh_s % op.stride != 0 || oh >= op.h_out);
            for (int k2 = 0;k2 < op.kernel;k2++) {
                const size_t ow_ks = x + op.padding;
                const size_t ow_s = ow_ks - op.dilation * k2;
                const size_t ow = ow_s / op.stride;
            
                const bool invalid = k1_invalid || (ow_ks < op.dilation * k2 || ow_s % op.stride != 0 || ow >= op.w_out);
                *patches_i = invalid ? zero : image_out[oh * op.w_out + ow];
                patches_i += op.h_in * op.w_in;
            }
        }
    }
}

template<typename T>
__device__ void transpose_filters(
    const Conv2DOp op,
    const T *filters, // 4d (ChanOut, ChanIn, KernelSize, KernelSize)
    const size_t *strides, // 4d filters strides
    T *filters_tr // 5d (Groups, ChanIn, ChanOut/Groups, KernelSize, KernelSize)
) {
    const size_t n = op.chan_in * op.chan_out * op.kernel * op.kernel;
    const size_t o_per_g = op.chan_out / op.groups;

    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        unsigned int idx = i;
        const size_t k2 = idx % op.kernel;
        idx /= op.kernel;
        const size_t k1 = idx % op.kernel;
        idx /= op.kernel;
        const size_t c = idx % op.chan_in;
        idx /= op.chan_in;
        const size_t o = idx % op.chan_out;
        const size_t og = o % o_per_g;
        const size_t g = o / o_per_g;
    
        auto i_no = o * strides[0] + c * strides[1] + k1 * strides[2] + k2 * strides[3];
        T *filters_tr_i = filters_tr + k2;
        filters_tr_i += k1 * op.kernel;
        filters_tr_i += og * (op.kernel * op.kernel);
        filters_tr_i += c * (o_per_g * op.kernel * op.kernel);
        filters_tr_i += g * (op.chan_in * o_per_g * op.kernel * op.kernel);
        *filters_tr_i = filters[i_no];
    }
}

template<typename T>
__device__ void sum_transposed_filters(
    const Conv2DOp op,
    const T *filters_tr, // 6d (Batch, Groups, ChanIn, ChanOut/Groups, KernelSize, KernelSize)
    T *filters, // 4d (ChanOut, ChanIn, KernelSize, KernelSize)
    const size_t *strides // 4d filter strides
) {
    const size_t n = op.chan_out * op.chan_in * op.kernel * op.kernel;
    const size_t o_per_g = op.chan_out / op.groups;

    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        unsigned int idx = i;
        const size_t k2 = idx % op.kernel;
        idx /= op.kernel;
        const size_t k1 = idx % op.kernel;
        idx /= op.kernel;
        const size_t c = idx % op.chan_in;
        idx /= op.chan_in;
        const size_t o = idx % op.chan_out;
        const size_t og = o % o_per_g;
        const size_t g = o / o_per_g;
    
        auto i_no = o * strides[0] + c * strides[1] + k1 * strides[2] + k2 * strides[3];
    
        const T *filters_tr_i = filters_tr + k2;
        filters_tr_i += k1 * op.kernel;
        filters_tr_i += og * (op.kernel * op.kernel);
        filters_tr_i += c * (o_per_g * op.kernel * op.kernel);
        filters_tr_i += g * (op.chan_in * o_per_g * op.kernel * op.kernel);
    
        T tmp = 0.0;
        for (int b = 0; b < op.batch; b++) {
            tmp += *filters_tr_i;
            filters_tr_i += n;
        }
    
        filters[i_no] += tmp;
    }
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
} \
extern "C" __global__ void SUM_TR_FILTERS( \
    const Conv2DOp op, \
    const TYPENAME *filters_tr, \
    TYPENAME *filters, \
    const size_t *strides \
) { \
    sum_transposed_filters(op, filters_tr, filters, strides); \
}

CONV_OP(
    __half,
    unfold_input_into_patches_f16,
    unfold_output_into_patches_f16,
    transpose_filters_f16,
    sum_transposed_filters_f16
);
CONV_OP(
    float,
    unfold_input_into_patches_f32,
    unfold_output_into_patches_f32,
    transpose_filters_f32,
    sum_transposed_filters_f32
);
CONV_OP(
    double,
    unfold_input_into_patches_f64,
    unfold_output_into_patches_f64,
    transpose_filters_f64,
    sum_transposed_filters_f64
);