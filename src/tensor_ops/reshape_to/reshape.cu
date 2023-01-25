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

extern "C" __global__ void reshape_forward(
    const size_t numel,
    const float *inp,
    const size_t inp_num_dims,
    const size_t *inp_dims,
    const size_t *inp_strides,
    float *out,
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

extern "C" __global__ void reshape_backward(
    const size_t numel,
    float *grad_inp,
    const size_t inp_num_dims,
    const size_t *inp_dims,
    const size_t *inp_strides,
    const float *grad_out,
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
