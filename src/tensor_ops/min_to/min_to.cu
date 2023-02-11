#include "cuda_utils.cuh"

// atomicMax is not implemented for floats,
// solution copied https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda
__device__ __forceinline__ float atomicMinf(float * addr, float value) {
    if (signbit(value)) {
        return __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));
    } else {
        return __int_as_float(atomicMin((int *)addr, __float_as_int(value)));
    }
}

__device__ __forceinline__ double atomicMinf(double * addr, double value) {
    if (signbit(value)) {
        return __longlong_as_double(atomicMax((unsigned long long int *)addr, __double_as_longlong(value)));
    } else {
        return __longlong_as_double(atomicMin((long long int *)addr, __double_as_longlong(value)));
    }
}

__device__ __forceinline__ float fminNonAtomic(float a, float b) {
    return fminf(a, b);
}

__device__ __forceinline__ double fminNonAtomic(double a, double b) {
    return fmin(a, b);
}

// Efficiently computes the min of each chunk in "data" of size chunk_len, and
// stores the minimums in out[i / chunk_len]
template<typename T>
__device__ void chunk_min(
    const size_t numel,
    const size_t chunk_len,
    const T data,
    T* out
) {
    __shared__ T buf[1024];
    // assumes that threads where i >= numel have already exited
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int block_i = threadIdx.x;
    buf[block_i] = data;

    unsigned int chunk_i = i % chunk_len;
    unsigned int chunk_start = max((int)(block_i - chunk_i), 0);
    unsigned int chunk_end = min((unsigned int)(block_i + chunk_len - chunk_i), blockDim.x);

    chunk_i = block_i - chunk_start;

    size_t max_chunk_len = min(chunk_end - chunk_start, blockDim.x);
    size_t incr = next_power_of_two(max_chunk_len) >> 1;

    __syncthreads();

    // Uses sequential addressing as discussed in
    // https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    for (; incr > 0; incr >>= 1) {
        unsigned int block_i_2 = block_i + incr;

        if (block_i_2 < chunk_end && chunk_i < incr) {
            // This is sound because __syncthreads and the conditions above
            // ensure that no data races occur
            buf[block_i] = fminNonAtomic(buf[block_i], buf[block_i_2]);
        }

        __syncthreads();
    }

    if (block_i == chunk_start) {
        atomicMinf(out + i / chunk_len, buf[block_i]);
    }
}

// strides and dims specify how to index inp to put all summed elements next to
// each other, and chunk_len is len(inp) / len(out)
template<typename T>
__device__ void min_to_forward(
    const size_t numel,
    const size_t num_dims,
    const size_t chunk_len,
    const T *inp,
    const size_t *dims,
    const size_t *strides,
    T *out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= numel) {
        return;
    }

    unsigned int inp_i = get_strided_index(i, num_dims, dims, strides);
    chunk_min(numel, chunk_len, inp[inp_i], out);
}

// Accepts pre-broadcasted strides for both input & output.
// So both inp & out are expected to be broadcasted to the same size.
template<typename T>
__device__ void min_to_backward(
    const size_t numel,
    const size_t num_dims,
    const T elems_per_thread,
    const size_t *dims,
    const T *inp,
    T *grad_inp,
    const size_t *inp_strides,
    const T *out,
    const T *grad_out,
    const size_t *out_strides
) {
    unsigned int inp_i = blockIdx.x * blockDim.x + threadIdx.x;

    if (inp_i >= numel) {
        return;
    }

    unsigned int i = get_unstrided_index(inp_i, num_dims, dims, inp_strides);
    unsigned int out_i = get_strided_index(i, num_dims, dims, out_strides);

    auto tmp = inp[inp_i] == out[out_i] ? grad_out[out_i] : 0.0;
    grad_inp[inp_i] += tmp * elems_per_thread;
}

extern "C" __global__ void min_to_forward_f32(
    const size_t numel,
    const size_t num_dims,
    const size_t chunk_len,
    const float *inp,
    const size_t *dims,
    const size_t *strides,
    float *out
) {
    min_to_forward(numel, num_dims, chunk_len, inp, dims, strides, out);
}

extern "C" __global__ void min_to_backward_f32(
    const size_t numel,
    const size_t num_dims,
    const float elems_per_thread,
    const size_t *dims,
    const float *inp,
    float *grad_inp,
    const size_t *inp_strides,
    const float *out,
    const float *grad_out,
    const size_t *out_strides
) {
    min_to_backward(numel, num_dims, elems_per_thread, dims, inp, grad_inp, inp_strides, out, grad_out, out_strides);
}

extern "C" __global__ void min_to_forward_f64(
    const size_t numel,
    const size_t num_dims,
    const size_t chunk_len,
    const double *inp,
    const size_t *dims,
    const size_t *strides,
    double *out
) {
    min_to_forward(numel, num_dims, chunk_len, inp, dims, strides, out);
}

extern "C" __global__ void min_to_backward_f64(
    const size_t numel,
    const size_t num_dims,
    const double elems_per_thread,
    const size_t *dims,
    const double *inp,
    double *grad_inp,
    const size_t *inp_strides,
    const double *out,
    const double *grad_out,
    const size_t *out_strides
) {
    min_to_backward(numel, num_dims, elems_per_thread, dims, inp, grad_inp, inp_strides, out, grad_out, out_strides);
}
