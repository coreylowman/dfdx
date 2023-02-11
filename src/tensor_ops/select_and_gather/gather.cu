#include "cuda_utils.cuh"

__device__ unsigned int get_gathered_index(
    const unsigned int index,
    const size_t inp_num_dims,
    const size_t *inp_dims,
    const size_t *inp_strides,
    const size_t *idx,
    const size_t idx_num_dims,
    const size_t *idx_dims,
    const size_t *idx_strides,
    const size_t out_num_dims
) {
    unsigned int ax;

    if (out_num_dims > inp_num_dims) {
        ax = 0;
    } else {
        ax = idx_num_dims - 1;
    }

    unsigned int elem_size = 1; // the size of each indexed element
    unsigned int row_len = inp_dims[ax]; // the size of the indexed dimension

    for (unsigned int d = 0; d < inp_num_dims - ax - 1; d++) {
        unsigned int dim_idx = inp_num_dims - 1 - d;
        elem_size *= inp_dims[dim_idx];
    }

    // location to find the index for the replaced dimension in "idx"
    unsigned int idx_idx = get_strided_index(index / elem_size, idx_num_dims, idx_dims, idx_strides);

    // indices for dimensions before, at, and after the indexed dimension
    unsigned int idx_before = index / (elem_size * row_len);
    unsigned int idx_mid = idx[idx_idx];
    unsigned int idx_after = index % elem_size;

    // recombine
    unsigned int new_idx = (idx_before * row_len + idx_mid) * elem_size + idx_after;
    return get_strided_index(new_idx, inp_num_dims, inp_dims, inp_strides);
}

template<typename T>
__device__ void gather_forward(
    const size_t numel,
    const T *inp,
    const size_t inp_num_dims,
    const size_t *inp_dims,
    const size_t *inp_strides,
    const size_t *idx,
    const size_t idx_num_dims,
    const size_t *idx_dims,
    const size_t *idx_strides,
    T *out,
    const size_t out_num_dims
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }

    unsigned int out_i = i;
    unsigned int inp_i =
        get_gathered_index(i, inp_num_dims, inp_dims, inp_strides, idx, idx_num_dims, idx_dims, idx_strides, out_num_dims);

    out[out_i] = inp[inp_i];
}

template<typename T>
__device__ void gather_backward(
    const size_t numel,
    T *grad_inp,
    const size_t inp_num_dims,
    const size_t *inp_dims,
    const size_t *inp_strides,
    const size_t *idx,
    const size_t idx_num_dims,
    const size_t *idx_dims,
    const size_t *idx_strides,
    const T *grad_out,
    const size_t out_num_dims
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }

    unsigned int out_i = i;
    unsigned int inp_i =
        get_gathered_index(i, inp_num_dims, inp_dims, inp_strides, idx, idx_num_dims, idx_dims, idx_strides, out_num_dims);

    atomicAdd(grad_inp + inp_i, grad_out[out_i]);
}

extern "C" __global__ void gather_forward_f32(
    const size_t numel,
    const float *inp,
    const size_t inp_num_dims,
    const size_t *inp_dims,
    const size_t *inp_strides,
    const size_t *idx,
    const size_t idx_num_dims,
    const size_t *idx_dims,
    const size_t *idx_strides,
    float *out,
    const size_t out_num_dims
) {
    gather_forward(numel, inp, inp_num_dims, inp_dims, inp_strides, idx, idx_num_dims, idx_dims, idx_strides, out, out_num_dims);
}

extern "C" __global__ void gather_backward_f32(
    const size_t numel,
    float *grad_inp,
    const size_t inp_num_dims,
    const size_t *inp_dims,
    const size_t *inp_strides,
    const size_t *idx,
    const size_t idx_num_dims,
    const size_t *idx_dims,
    const size_t *idx_strides,
    const float *grad_out,
    const size_t out_num_dims
) {
    gather_backward(numel, grad_inp, inp_num_dims, inp_dims, inp_strides, idx, idx_num_dims, idx_dims, idx_strides, grad_out, out_num_dims);
}

extern "C" __global__ void gather_forward_f64(
    const size_t numel,
    const double *inp,
    const size_t inp_num_dims,
    const size_t *inp_dims,
    const size_t *inp_strides,
    const size_t *idx,
    const size_t idx_num_dims,
    const size_t *idx_dims,
    const size_t *idx_strides,
    double *out,
    const size_t out_num_dims
) {
    gather_forward(numel, inp, inp_num_dims, inp_dims, inp_strides, idx, idx_num_dims, idx_dims, idx_strides, out, out_num_dims);
}

extern "C" __global__ void gather_backward_f64(
    const size_t numel,
    double *grad_inp,
    const size_t inp_num_dims,
    const size_t *inp_dims,
    const size_t *inp_strides,
    const size_t *idx,
    const size_t idx_num_dims,
    const size_t *idx_dims,
    const size_t *idx_strides,
    const double *grad_out,
    const size_t out_num_dims
) {
    gather_backward(numel, grad_inp, inp_num_dims, inp_dims, inp_strides, idx, idx_num_dims, idx_dims, idx_strides, grad_out, out_num_dims);
}
