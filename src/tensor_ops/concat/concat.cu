extern "C" __global__ void concat_bwd_f32(const size_t numel, const float *inp, float *out) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    out[i] += inp[i];
}
extern "C" __global__ void concat_bwd_f64(const size_t numel, const double *inp, double *out) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    out[i] += inp[i];
}
