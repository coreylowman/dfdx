extern "C" __global__ void prelu_fwd_f32(
    const size_t size,
    const float *lhs, // Any tensor
    const float rhs, // A 0d tensor
    float *out // same tensor
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("(%i %x %x)", i, lhs, out);
    out[i] = fmaxf(lhs[i], 0.0) + fminf(lhs[i],0.0)*rhs;
}

extern "C" __global__ void prelu_bwd_f32(
    const size_t size,
    const float *lhs, // floats
    float *lhs_grad,
    const float rhs, // A 0d tensor
    float *rhs_grad,
    const float *out_grad // floats
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    float size_f = (float)size;
    size_f = 1/size_f;

    if (lhs[i] >= 0) {
        lhs_grad[i] += out_grad[i];
    }
    else {
        lhs_grad[i] += rhs * out_grad[i];
        *rhs_grad += lhs[i] * size_f * out_grad[i];
    }
}