struct Conv2DOp {
    size_t stride;
    size_t padding;
    size_t kernel;
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
    const T *image, // 4d (Batch, Channels, Height, Width)
    const size_t *strides, // 4d image strides
    T *patches // 6d (Batch, Channels, KernelSize, KernelSize, HeightOut, WidthOut)
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t item_numel = op.batch * op.chan_in * op.kernel * op.kernel * op.h_out * op.w_out;
    if (i >= item_numel) {
        return;
    }

    // patches shape is (B, C, K, K, h_out, w_out)
    unsigned int idx = i;
    const size_t ow = idx % op.w_out;
    idx /= op.w_out;
    const size_t oh = idx % op.h_out;
    idx /= op.h_out;
    const size_t k2 = idx % op.kernel;
    idx /= op.kernel;
    const size_t k1 = idx % op.kernel;
    idx /= op.kernel;
    const size_t c = idx % op.chan_in;
    idx /= op.chan_in;
    const size_t b = idx % op.batch;

    const size_t y_plus_p = oh * op.stride + k1;
    const size_t y = y_plus_p - op.padding;
    const size_t x_plus_p = ow * op.stride + k2;
    const size_t x = x_plus_p - op.padding;

    if (y >= op.h_in || x >= op.w_in) {
        patches[i] = 0.0;
    } else {
        const size_t i_image = b * strides[0] + c * strides[1] + y * strides[2] + x * strides[3];
        patches[i] = image[i_image];
    }
}

template<typename T>
__device__ void unfold_output_into_patches(
    const Conv2DOp op,
    const T *image_out, // 4d (Batch, ChanOut, HeightOut, WidthOut)
    T *patches // 6d (Batch, ChanOut, KernelSize, KernelSize, HeightIn, WidthIn)
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t item_numel = op.chan_out * op.kernel * op.kernel * op.h_in * op.w_in;
    if (i >= item_numel) {
        return;
    }

    unsigned int idx = i;
    const size_t x = idx % op.w_in;
    idx /= op.w_in;
    const size_t y = idx % op.h_in;
    idx /= op.h_in;
    const size_t k2 = idx % op.kernel;
    idx /= op.kernel;
    const size_t k1 = idx % op.kernel;
    idx /= op.kernel;
    const size_t o = idx % op.chan_out;

    const size_t oh_ks = y + op.padding;
    const size_t oh_s = oh_ks - k1;
    const size_t oh = oh_s / op.stride;
    const size_t ow_ks = x + op.padding;
    const size_t ow_s = ow_ks - k2;
    const size_t ow = ow_s / op.stride;

    if (
        (oh_ks < k1 || oh_s % op.stride != 0 || oh >= op.h_out)
        || (ow_ks < k2 || ow_s % op.stride != 0 || ow >= op.w_out)
    ) {
        for (auto b = 0; b < op.batch; b++) {
            patches[b * item_numel + i] = 0.0;
        }
        return;
    }

    for (auto b = 0; b < op.batch; b++) {
        size_t image_i = b * (op.chan_out * op.h_out * op.w_out) + o * (op.h_out * op.w_out) + oh * (op.w_out)  + ow;
        patches[b * item_numel + i] = image_out[image_i];
    }
}

template<typename T>
__device__ void transpose_and_broadcast_filters(
    const Conv2DOp op,
    const T *filters, // 4d (ChanOut, ChanIn, KernelSize, KernelSize)
    const size_t *strides, // 4d filters strides
    T *filters_tr // 5d (Batch, ChanIn, ChanOut, KernelSize, KernelSize)
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    auto numel = op.chan_in * op.chan_out * op.kernel * op.kernel;
    if (i >= numel) {
        return;
    }

    unsigned int idx = i;
    const size_t k2 = idx % op.kernel;
    idx /= op.kernel;
    const size_t k1 = idx % op.kernel;
    idx /= op.kernel;
    const size_t c = idx % op.chan_in;
    idx /= op.chan_in;
    const size_t o = idx % op.chan_out;

    auto i_tr = c * (op.chan_out * op.kernel * op.kernel) + o * (op.kernel * op.kernel) + k1 * (op.kernel) + k2;
    auto i_no = o * strides[0] + c * strides[1] + k1 * strides[2] + k2 * strides[3];

    const T f = filters[i_no];
    for (auto b = 0; b < op.batch; b++) {
        filters_tr[b * numel + i_tr] = f;
    }
}

template<typename T>
__device__ void sum_transposed_filters(
    const Conv2DOp op,
    const T *filters_tr, // 5d (Batch, ChanIn, ChanOut, KernelSize, KernelSize)
    T *filters, // 4d (ChanOut, ChanIn, KernelSize, KernelSize)
    const size_t *strides // 4d filter strides
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    auto numel = op.chan_out * op.chan_in * op.kernel * op.kernel;
    if (i >= numel) {
        return;
    }

    unsigned int idx = i;
    const size_t k2 = idx % op.kernel;
    idx /= op.kernel;
    const size_t k1 = idx % op.kernel;
    idx /= op.kernel;
    const size_t c = idx % op.chan_in;
    idx /= op.chan_in;
    const size_t o = idx % op.chan_out;
    idx /= op.chan_out;

    auto i_tr = c * (op.chan_out * op.kernel * op.kernel) + o * (op.kernel * op.kernel) + k1 * (op.kernel) + k2;
    auto i_no = o * strides[0] + c * strides[1] + k1 * strides[2] + k2 * strides[3];

    T tmp = 0.0;
    for (auto b = 0; b < op.batch; b++) {
        tmp += filters_tr[b * numel + i_tr];
    }

    filters[i_no] += tmp;
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
    transpose_and_broadcast_filters(op, filters, strides, filters_tr); \
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
    float,
    unfold_input_into_patches_f32,
    unfold_output_into_patches_f32,
    transpose_and_broadcast_filters_f32,
    sum_transposed_filters_f32
);
CONV_OP(
    double,
    unfold_input_into_patches_f64,
    unfold_output_into_patches_f64,
    transpose_and_broadcast_filters_f64,
    sum_transposed_filters_f64
);