#include <cstdint>
#include "cuda_utils.cuh"

template<typename T>
__device__ void slice_fwd(
    const size_t numel,
    const size_t num_dims,
    const size_t *dims,
    const size_t *strides,
    const size_t offset,
    const T *inp,
    T *out
) {
    unsigned int out_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_i >= numel) {
        return;
    }

    unsigned int inp_i = offset + get_strided_index(out_i, num_dims, dims, strides);
    out[out_i] = inp[inp_i];
}

template<typename T>
__device__ void slice_bwd(
    const size_t numel,
    const size_t num_dims,
    const size_t *dims,
    const size_t *strides,
    const size_t offset,
    T *grad_inp,
    const T *out
) {
    unsigned int out_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_i >= numel) {
        return;
    }

    unsigned int inp_i = offset + get_strided_index(out_i, num_dims, dims, strides);
    // TODO (maybe): use chunk_sum to speed this up 
    atomicAdd(grad_inp + inp_i, out[out_i]);
}

#define SLICE_FWD(TYPENAME, FN) \
extern "C" __global__ void FN( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *dims, \
    const size_t *strides, \
    const size_t offset, \
    const TYPENAME *inp, \
    TYPENAME *out \
) { \
    slice_fwd(numel, num_dims, dims, strides, offset, inp, out); \
}

#define SLICE(TYPENAME, FWD, BWD) \
SLICE_FWD(TYPENAME, FWD) \
\
extern "C" __global__ void BWD( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *dims, \
    const size_t *strides, \
    const size_t offset, \
    TYPENAME *grad_inp, \
    const TYPENAME *grad_out \
) { \
    slice_bwd(numel, num_dims, dims, strides, offset, grad_inp, grad_out); \
}

SLICE(float, slice_fwd_f32, slice_bwd_f32);
SLICE(double, slice_fwd_f64, slice_bwd_f64);
SLICE_FWD(uint8_t, slice_fwd_u8);
SLICE_FWD(uint16_t, slice_fwd_u16);
SLICE_FWD(uint32_t, slice_fwd_u32);
SLICE_FWD(uint64_t, slice_fwd_u64);
SLICE_FWD(uintptr_t, slice_fwd_usize);
SLICE_FWD(int8_t, slice_fwd_i8);
SLICE_FWD(int16_t, slice_fwd_i16);
SLICE_FWD(int32_t, slice_fwd_i32);
SLICE_FWD(int64_t, slice_fwd_i64);
SLICE_FWD(intptr_t, slice_fwd_isize);
SLICE_FWD(bool, slice_fwd_bool);
