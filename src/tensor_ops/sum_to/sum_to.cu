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

// Accepts pre-broadcasted strides for both input & output.
// So both inp & out are expected to be broadcasted to the same size.
extern "C" __global__ void sum_to_forward(
    const size_t numel,
    const size_t num_dims,
    const size_t *dims,
    const float *inp,
    const size_t *inp_strides,
    float *out,
    const size_t *out_strides
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= numel) {
        return;
    }

    unsigned int inp_strided_i = get_strided_index(i, num_dims, dims, inp_strides);
    auto tmp = inp[inp_strided_i];

    unsigned int out_strided_i = get_strided_index(i, num_dims, dims, out_strides);
    atomicAdd(out + out_strided_i, tmp);
}

// Accepts pre-broadcasted strides for both input & output.
// So both inp & out are expected to be broadcasted to the same size.
extern "C" __global__ void sum_to_backward(
    const size_t numel,
    const size_t num_dims,
    const size_t *dims,
    float *grad_inp,
    const size_t *inp_strides,
    const float *grad_out,
    const size_t *out_strides
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= numel) {
        return;
    }

    unsigned int out_strided_i = get_strided_index(i, num_dims, dims, out_strides);
    auto tmp = grad_out[out_strided_i];

    // NOTE: since size of output is less than input, only 1 thread will be writing to inp
    // at a time. this means we don't have to worry about multiple concurrent writes
    // like we do with forward.
    unsigned int inp_strided_i = get_strided_index(i, num_dims, dims, inp_strides);
    atomicAdd(grad_inp + inp_strided_i, tmp);
}
