template<typename T>
__device__ void ema(const size_t n, const T* src, T* dst, T decay) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    dst[i] = dst[i] * decay + src[i] * (1 - decay);
}

extern "C" __global__ void ema_f32(const size_t n, const float* src, float* dst, float decay) {
    ema(n, src, dst, decay);
}

extern "C" __global__ void ema_f64(const size_t n, const double* src, double* dst, double decay) {
    ema(n, src, dst, decay);
}
