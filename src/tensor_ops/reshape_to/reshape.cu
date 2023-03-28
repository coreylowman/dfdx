#include <cstdint>
#include "cuda_utils.cuh"

template<typename T>
__device__ void reshape_fwd(
    const size_t numel,
    const T *inp,
    const size_t inp_num_dims,
    const size_t *inp_dims,
    const size_t *inp_strides,
    T *out,
    const size_t out_num_dims,
    const size_t *out_dims,
    const size_t *out_strides
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }

    unsigned int inp_i = get_strided_index(i, inp_num_dims, inp_dims, inp_strides);
    unsigned int out_i = get_strided_index(i, out_num_dims, out_dims, out_strides);

    out[out_i] = inp[inp_i];
}

template<typename T>
__device__ void reshape_bwd(
    const size_t numel,
    T *grad_inp,
    const size_t inp_num_dims,
    const size_t *inp_dims,
    const size_t *inp_strides,
    const T *grad_out,
    const size_t out_num_dims,
    const size_t *out_dims,
    const size_t *out_strides
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }

    unsigned int inp_i = get_strided_index(i, inp_num_dims, inp_dims, inp_strides);
    unsigned int out_i = get_strided_index(i, out_num_dims, out_dims, out_strides);

    atomicAdd(grad_inp + inp_i, grad_out[out_i]);
}

#define RESHAPE_FWD(TYPENAME, FN) \
extern "C" __global__ void FN( \
    const size_t numel, \
    const TYPENAME *inp, \
    const size_t inp_num_dims, \
    const size_t *inp_dims, \
    const size_t *inp_strides, \
    TYPENAME *out, \
    const size_t out_num_dims, \
    const size_t *out_dims, \
    const size_t *out_strides \
) { \
    reshape_fwd(numel, inp, inp_num_dims, inp_dims, inp_strides, out, out_num_dims, out_dims, out_strides); \
}

#define RESHAPE(TYPENAME, FWD, BWD) \
RESHAPE_FWD(TYPENAME, FWD) \
\
extern "C" __global__ void BWD( \
    const size_t numel, \
    TYPENAME *grad_inp, \
    const size_t inp_num_dims, \
    const size_t *inp_dims, \
    const size_t *inp_strides, \
    const TYPENAME *grad_out, \
    const size_t out_num_dims, \
    const size_t *out_dims, \
    const size_t *out_strides \
) { \
    reshape_bwd(numel, grad_inp, inp_num_dims, inp_dims, inp_strides, grad_out, out_num_dims, out_dims, out_strides); \
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
