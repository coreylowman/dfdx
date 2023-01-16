__device__ unsigned int get_strided_index(
    unsigned int idx,
    size_t num_dims,
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

// Accepts pre-broadcasted strides for both input & output.
// So both inp & out are expected to be broadcasted to the same size.
extern "C" __global__ void sum_to_forward(
    const size_t numel,
    const size_t num_dims,
    const float mul,
    const size_t *dims,
    const float *inp,
    const size_t *inp_strides,
    float *out,
    const size_t *out_strides
) {
    unsigned int inp_i = blockIdx.x * blockDim.x + threadIdx.x;

    if (inp_i >= numel) {
        return;
    }

    auto tmp = inp[inp_i];

    unsigned int i = get_unstrided_index(inp_i, num_dims, dims, inp_strides);
    unsigned int out_i = get_strided_index(i, num_dims, dims, out_strides);
    atomicAdd(out + out_i, tmp * mul);
}

// Accepts pre-broadcasted strides for both input & output.
// So both inp & out are expected to be broadcasted to the same size.
extern "C" __global__ void sum_to_backward(
    const size_t numel,
    const size_t num_dims,
    const float mul,
    const size_t *dims,
    float *grad_inp,
    const size_t *inp_strides,
    const float *grad_out,
    const size_t *out_strides
) {
    unsigned int inp_i = blockIdx.x * blockDim.x + threadIdx.x;

    if (inp_i >= numel) {
        return;
    }

    unsigned int i = get_unstrided_index(inp_i, num_dims, dims, inp_strides);
    unsigned int out_i = get_strided_index(i, num_dims, dims, out_strides);
    auto tmp = grad_out[out_i];

    // NOTE: since size of output is less than input, only 1 thread will be writing to inp
    // at a time. this means we don't have to worry about multiple concurrent writes
    // like we do with forward.
    grad_inp[inp_i] += tmp * mul;
}
