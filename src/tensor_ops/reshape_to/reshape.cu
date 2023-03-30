#include <cstdint>
#include "cuda_utils.cuh"

template<typename T>
__device__ void reshape_fwd(
    const size_t numel,
    const size_t inp_num_dims,
    const size_t out_num_dims,
    const size_t *info,
    const T *inp,
    T *out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }

    const size_t *inp_dims = info;
    const size_t *inp_strides = info + inp_num_dims;
    const size_t *out_dims = info + 2 * inp_num_dims;
    const size_t *out_strides = info + 2 * inp_num_dims + out_num_dims;

    unsigned int inp_i = get_strided_index(i, inp_num_dims, inp_dims, inp_strides);
    unsigned int out_i = get_strided_index(i, out_num_dims, out_dims, out_strides);

    out[out_i] = inp[inp_i];
}

template<typename T>
__device__ void reshape_bwd(
    const size_t numel,
    const size_t inp_num_dims,
    const size_t out_num_dims,
    const size_t *info,
    T *grad_inp,
    const T *grad_out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }

    const size_t *inp_dims = info;
    const size_t *inp_strides = info + inp_num_dims;
    const size_t *out_dims = info + 2 * inp_num_dims;
    const size_t *out_strides = info + 2 * inp_num_dims + out_num_dims;

    unsigned int inp_i = get_strided_index(i, inp_num_dims, inp_dims, inp_strides);
    unsigned int out_i = get_strided_index(i, out_num_dims, out_dims, out_strides);

    atomicAdd(grad_inp + inp_i, grad_out[out_i]);
}

#define RESHAPE_FWD(TYPENAME, FN) \
extern "C" __global__ void FN( \
    const size_t numel, \
    const size_t inp_num_dims, \
    const size_t out_num_dims, \
    const size_t *info, \
    const TYPENAME *inp, \
    TYPENAME *out \
) { \
    reshape_fwd(numel, inp_num_dims, out_num_dims, info, inp, out); \
}

#define RESHAPE(TYPENAME, FWD, BWD) \
RESHAPE_FWD(TYPENAME, FWD) \
\
extern "C" __global__ void BWD( \
    const size_t numel, \
    const size_t inp_num_dims, \
    const size_t out_num_dims, \
    const size_t *info, \
    TYPENAME *grad_inp, \
    const TYPENAME *grad_out \
) { \
    reshape_bwd(numel, inp_num_dims, out_num_dims, info, grad_inp, grad_out); \
}

RESHAPE(float, reshape_fwd_f32, reshape_bwd_f32);
RESHAPE(double, reshape_fwd_f64, reshape_bwd_f64);
RESHAPE_FWD(uint8_t, reshape_fwd_u8);
RESHAPE_FWD(uint16_t, reshape_fwd_u16);
RESHAPE_FWD(uint32_t, reshape_fwd_u32);
RESHAPE_FWD(uint64_t, reshape_fwd_u64);
RESHAPE_FWD(uintptr_t, reshape_fwd_usize);
RESHAPE_FWD(int8_t, reshape_fwd_i8);
RESHAPE_FWD(int16_t, reshape_fwd_i16);
RESHAPE_FWD(int32_t, reshape_fwd_i32);
RESHAPE_FWD(int64_t, reshape_fwd_i64);
RESHAPE_FWD(intptr_t, reshape_fwd_isize);
RESHAPE_FWD(bool, reshape_fwd_bool);
