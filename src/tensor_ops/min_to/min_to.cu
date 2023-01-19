// atomicMax is not implemented for floats,
// solution copied https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda
__device__ __forceinline__ float atomicMinf(float * addr, float value) {
    if (signbit(value)) {
        return __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));
    } else {
        return __int_as_float(atomicMin((int *)addr, __float_as_int(value)));
    }
}

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

extern "C" __global__ void fill_with(float *buf, float value, const size_t numel) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    buf[i] = value;
}

// Accepts pre-broadcasted strides for both input & output.
// So both inp & out are expected to be broadcasted to the same size.
extern "C" __global__ void min_to_forward(
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
    unsigned int out_strided_i = get_strided_index(i, num_dims, dims, out_strides);

    atomicMinf(out + out_strided_i, inp[inp_strided_i]);
}

// Accepts pre-broadcasted strides for both input & output.
// So both inp & out are expected to be broadcasted to the same size.
extern "C" __global__ void min_to_backward(
    const size_t numel,
    const size_t num_dims,
    const size_t *dims,
    const float *inp,
    float *grad_inp,
    const size_t *inp_strides,
    const float *out,
    const float *grad_out,
    const size_t *out_strides
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= numel) {
        return;
    }

    unsigned int inp_strided_i = get_strided_index(i, num_dims, dims, inp_strides);
    unsigned int out_strided_i = get_strided_index(i, num_dims, dims, out_strides);

    auto tmp = inp[inp_strided_i] == out[out_strided_i] ? grad_out[out_strided_i] : 0.0;
    atomicAdd(grad_inp + inp_strided_i, tmp);
}
