#include "cuda_utils.cuh"

struct RollOp {
    size_t axis;
    size_t amount;
};

template<typename T>
__device__ void roll_fwd(
    const RollOp op,
    const size_t num_dims,
    const size_t numel,
    const size_t *dims,
    const size_t *inp_strides,
    const size_t *out_strides,
    const T *inp,
    T *out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }

    const T item = inp[get_strided_index(i, num_dims, dims, inp_strides)];

    size_t out_i = 0;
    for (int d = num_dims - 1; d > op.axis; d--) {
        size_t dim_i = i % dims[d];
        out_i += dim_i * out_strides[d];
        i /= dims[d];
    }

    size_t dim_i = i % dims[op.axis];
    size_t new_dim_i = (dim_i + op.amount) % dims[op.axis];
    out_i += new_dim_i * out_strides[op.axis];
    i /= dims[op.axis];

    for (int d = op.axis - 1; d >= 0;d--) {
        size_t dim_i = i % dims[d];
        out_i += dim_i * out_strides[d];
        i /= dims[d];
    }

    out[out_i] = item;
}

template<typename T>
__device__ void roll_bwd(
    const RollOp op,
    const size_t num_dims,
    const size_t numel,
    const size_t *dims,
    const size_t *inp_strides,
    const size_t *out_strides,
    T *grad_inp,
    const T *grad_out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }

    const size_t inp_i = get_strided_index(i, num_dims, dims, inp_strides);

    size_t out_i = 0;
    for (int d = num_dims - 1; d > op.axis; d--) {
        size_t dim_i = i % dims[d];
        out_i += dim_i * out_strides[d];
        i /= dims[d];
    }

    size_t dim_i = i % dims[op.axis];
    size_t new_dim_i = (dim_i + op.amount) % dims[op.axis];
    out_i += new_dim_i * out_strides[op.axis];
    i /= dims[op.axis];

    for (int d = op.axis - 1; d >= 0;d--) {
        size_t dim_i = i % dims[d];
        out_i += dim_i * out_strides[d];
        i /= dims[d];
    }

    atomicAdd(grad_inp + inp_i, grad_out[out_i]);
}

#define ROLL(TY, FWD, BWD) \
extern "C" __global__ void FWD( \
    const RollOp op, \
    const size_t num_dims, \
    const size_t numel, \
    const size_t *dims, \
    const size_t *inp_strides, \
    const size_t *out_strides, \
    const TY *inp, \
    TY *out \
) { roll_fwd(op, num_dims, numel, dims, inp_strides, out_strides, inp, out); } \
extern "C" __global__ void BWD( \
    const RollOp op, \
    const size_t num_dims, \
    const size_t numel, \
    const size_t *dims, \
    const size_t *inp_strides, \
    const size_t *out_strides, \
    TY *grad_inp, \
    const TY *grad_out \
) { roll_bwd(op, num_dims, numel, dims, inp_strides, out_strides, grad_inp, grad_out); }

ROLL(__half, roll_fwd_f16, roll_bwd_f16);
ROLL(float, roll_fwd_f32, roll_bwd_f32);
ROLL(double, roll_fwd_f64, roll_bwd_f64);
