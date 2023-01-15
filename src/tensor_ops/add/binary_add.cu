struct BinaryAddOp {};

__device__ unsigned int get_strided_index(
    unsigned int idx,
    const size_t num_dims,
    const size_t *dims,
    const size_t *strides
) {
    unsigned int strided_i = 0;
    for (unsigned int d = 0; d < num_dims; d++) {
        unsigned int dim_idx = num_dims - 1 - d;
        strided_i += (idx % dims[dim_idx]) * strides[dim_idx];
        idx /= dims[dim_idx];
    }
    return strided_i;
}

__device__ unsigned int get_unstrided_index(
    const unsigned int strided_i,
    const size_t num_dims,
    const size_t *dims,
    const size_t *strides
) {
    unsigned int idx = 0;
    for (unsigned int d = 0; d < num_dims; d++) {
        idx *= dims[d];
        idx += strides[d] == 0 ? 0 : (strided_i / strides[d]) % dims[d];
    }
    return idx;
}

extern "C" __global__ void binary_add_forward(
    const BinaryAddOp op,
    const size_t numel,
    const size_t num_dims,
    const size_t *dims,
    const float *lhs,
    const size_t *lhs_strides,
    const float *rhs,
    const size_t *rhs_strides,
    float *out,
    const size_t *out_strides
) {
    unsigned int out_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_i >= numel) {
        return;
    }

    unsigned int i = get_unstrided_index(out_i, num_dims, dims, out_strides);
    unsigned int lhs_i = get_strided_index(i, num_dims, dims, lhs_strides);
    unsigned int rhs_i = get_strided_index(i, num_dims, dims, rhs_strides);

    out[out_i] = lhs[lhs_i] + rhs[rhs_i];
}

extern "C" __global__ void binary_add_backward(
    const BinaryAddOp op,
    const size_t numel,
    const size_t num_dims,
    const size_t *dims,
    const float *lhs,
    float *grad_lhs,
    const size_t *lhs_strides,
    const float *rhs,
    float *grad_rhs,
    const size_t *rhs_strides,
    const float *grad_out,
    const size_t *out_strides
) {
    unsigned int out_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_i >= numel) {
        return;
    }

    unsigned int virtual_numel = 1;
    for (unsigned int d = 0; d < num_dims; d++) {
        virtual_numel *= dims[d];
    }
    float mul = virtual_numel / numel;

    unsigned int i = get_unstrided_index(out_i, num_dims, dims, out_strides);
    unsigned int lhs_i = get_strided_index(i, num_dims, dims, lhs_strides);
    unsigned int rhs_i = get_strided_index(i, num_dims, dims, rhs_strides);

    auto x = lhs[lhs_i];
    auto y = rhs[rhs_i];
    auto go = grad_out[out_i];

    atomicAdd(grad_lhs + lhs_i, go * mul);
    atomicAdd(grad_rhs + rhs_i, go * mul);
}
