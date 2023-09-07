#include "cuda_fp16.h"

struct Conv1DOp {
  size_t kernel;
  size_t stride;
  size_t padding;
  size_t dilation;
  size_t groups;
  size_t batch;
  size_t chan_in;
  size_t chan_out;
  size_t l_in;
  size_t l_out;
};

template <typename T>
__device__ void unfold_input_into_patches(
    const Conv1DOp op,
    const T *image,        // 3d (Batch, Groups * Channels, Length)
    const size_t *strides, // 3d image strides
    T *patches // 4d (Batch, Groups * Channels, KernelSize, LengthOut)
) {
  const size_t n = op.batch * op.chan_in * op.l_out;
  for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    unsigned int idx = i;
    const size_t ol = idx % op.l_out;
    idx /= op.l_out;
    const size_t c = idx % op.chan_in;
    idx /= op.chan_in;
    const size_t b = idx % op.batch;

    const T *image_i = image + b * strides[0] + c * strides[1];
    T *patches_i = patches + ol;
    patches_i += c * (op.kernel * op.l_out);
    patches_i += b * (op.chan_in * op.kernel * op.l_out);

    T zero = 0.0;

    for (int k1 = 0; k1 < op.kernel; k1++) {
      const size_t y = ol * op.stride + op.dilation * k1 - op.padding;
      *patches_i = (y >= op.l_in) ? zero : image_i[y * strides[2]];
      patches_i += op.l_out;
    }
  }
}

template <typename T>
__device__ void unfold_output_into_patches(
    const Conv1DOp op,
    const T *image_out, // 3d (Batch, ChanOut, LengthOut)
    T *patches          // 4d (Batch, ChanOut, KernelSize, LengthIn)
) {
  const size_t n = op.batch * op.chan_out * op.l_in;
  for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    unsigned int idx = i;
    const size_t y = idx % op.l_in;
    idx /= op.l_in;
    const size_t o = idx % op.chan_out;
    idx /= op.chan_out;
    const size_t b = idx % op.batch;

    const T *image_i =
        image_out + b * (op.chan_out * op.l_out) + o * op.l_out;
    T *patches_i = patches + y;
    patches_i += o * (op.kernel * op.l_in);
    patches_i += b * (op.chan_out * op.kernel * op.l_in);

    T zero = 0.0;

    for (int k1 = 0; k1 < op.kernel; k1++) {
      const size_t ol_ks = y + op.padding;
      const size_t ol_s = ol_ks - op.dilation * k1;
      const size_t ol = ol_s / op.stride;
      const bool invalid =
          (ol_ks < op.dilation * k1 || ol_s % op.stride != 0 || ol >= op.l_out);

      *patches_i = invalid ? zero : image_i[ol];
      patches_i += op.l_in;
    }
  }
}

template <typename T>
__device__ void transpose_filters(
    const Conv1DOp op,
    const T *filters,      // 3d (ChanOut, ChanIn/Groups, KernelSize)
    const size_t *strides, // 4d filters strides
    T *filters_tr // 4d (Groups, ChanIn/Groups, ChanOut/Groups, KernelSize)
) {
  const size_t c_per_g = op.chan_in / op.groups;
  const size_t o_per_g = op.chan_out / op.groups;
  const size_t n = c_per_g * op.chan_out * op.kernel;

  for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    unsigned int idx = i;
    const size_t k1 = idx % op.kernel;
    idx /= op.kernel;
    const size_t cg = idx % c_per_g;
    idx /= c_per_g;
    const size_t o = idx % op.chan_out;
    const size_t og = o % o_per_g;
    const size_t g = o / o_per_g;

    auto i_no = o * strides[0] + cg * strides[1] + k1 * strides[2];
    T *filters_tr_i = filters_tr + k1;
    filters_tr_i += og * op.kernel;
    filters_tr_i += cg * (o_per_g * op.kernel);
    filters_tr_i += g * (c_per_g * o_per_g * op.kernel);
    *filters_tr_i = filters[i_no];
  }
}

template <typename T>
__device__ void
sum_transposed_filters(const Conv1DOp op,
                       const T *filters_tr, // 5d (Batch, Groups, ChanIn/Groups,
                                            // ChanOut/Groups, KernelSize)
                       T *filters, // 3d (ChanOut, ChanIn/Groups, KernelSize)
                       const size_t *strides // 3d filter strides
) {
  const size_t o_per_g = op.chan_out / op.groups;
  const size_t c_per_g = op.chan_in / op.groups;
  const size_t n = op.chan_out * c_per_g * op.kernel;

  for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    unsigned int idx = i;
    const size_t k1 = idx % op.kernel;
    idx /= op.kernel;
    const size_t cg = idx % c_per_g;
    idx /= c_per_g;
    const size_t o = idx % op.chan_out;
    const size_t og = o % o_per_g;
    const size_t g = o / o_per_g;

    auto i_no = o * strides[0] + cg * strides[1] + k1 * strides[2];

    const T *filters_tr_i = filters_tr + k1;
    filters_tr_i += og * op.kernel;
    filters_tr_i += cg * (o_per_g * op.kernel);
    filters_tr_i += g * (c_per_g * o_per_g * op.kernel);

    T tmp = 0.0;
    for (int b = 0; b < op.batch; b++) {
      tmp += *filters_tr_i;
      filters_tr_i += n;
    }

    filters[i_no] += tmp;
  }
}

#define CONV_OP(TYPENAME, UNFOLD_INPUT, UNFOLD_OUTPUT, TR_FILTERS,             \
                SUM_TR_FILTERS)                                                \
  extern "C" __global__ void UNFOLD_INPUT(                                     \
      const Conv1DOp op, const TYPENAME *image, const size_t *strides,         \
      TYPENAME *patches) {                                                     \
    unfold_input_into_patches(op, image, strides, patches);                    \
  }                                                                            \
  extern "C" __global__ void UNFOLD_OUTPUT(                                    \
      const Conv1DOp op, const TYPENAME *image_out, TYPENAME *patches) {       \
    unfold_output_into_patches(op, image_out, patches);                        \
  }                                                                            \
  extern "C" __global__ void TR_FILTERS(                                       \
      const Conv1DOp op, const TYPENAME *filters, const size_t *strides,       \
      TYPENAME *filters_tr) {                                                  \
    transpose_filters(op, filters, strides, filters_tr);                       \
  }                                                                            \
  extern "C" __global__ void SUM_TR_FILTERS(                                   \
      const Conv1DOp op, const TYPENAME *filters_tr, TYPENAME *filters,        \
      const size_t *strides) {                                                 \
    sum_transposed_filters(op, filters_tr, filters, strides);                  \
  }

CONV_OP(__half, unfold_input_into_patches_f16, unfold_output_into_patches_f16,
        transpose_filters_f16, sum_transposed_filters_f16);
CONV_OP(float, unfold_input_into_patches_f32, unfold_output_into_patches_f32,
        transpose_filters_f32, sum_transposed_filters_f32);
CONV_OP(double, unfold_input_into_patches_f64, unfold_output_into_patches_f64,
        transpose_filters_f64, sum_transposed_filters_f64);
