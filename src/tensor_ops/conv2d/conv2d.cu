struct ConvParams {
    size_t channels_in;
    size_t height_in;
    size_t width_in;
    size_t stride;
    size_t padding;
    size_t kernel;
    size_t channels_out;
    size_t height_out;
    size_t width_out;
};

extern "C" __global__ void unfold_input_into_patches(
    const ConvParams op,
    const float *image,
    const size_t *image_strides,
    float *patches,
    const size_t numel
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }

    // patches shape is (C, K, K, height_out, width_out)
    unsigned idx = i;
    const size_t ow = idx % op.width_out;
    idx /= op.width_out;
    const size_t oh = idx % op.height_out;
    idx /= op.height_out;
    const size_t k2 = idx % op.kernel;
    idx /= op.kernel;
    const size_t k1 = idx % op.kernel;
    idx /= op.kernel;
    const size_t c = idx % op.channels_in;

    const size_t y_plus_p = oh * op.stride + k1;
    if (y_plus_p < op.padding) {
        return;
    }
    const size_t y = y_plus_p - op.padding;

    const size_t x_plus_p = ow * op.stride + k2;
    if (x_plus_p < op.padding) {
        return;
    }
    const size_t x = x_plus_p - op.padding;

    if (y >= op.height_in || x >= op.width_in) {
        return;
    }

    patches[i] = image[c * image_strides[0] + y * image_strides[1] + x * image_strides[2]];
}

extern "C" __global__ void unfold_output_into_patches(
    const ConvParams op,
    const float *image,
    const size_t *image_strides,
    float *patches,
    const size_t numel
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }

    // patches shape is (channels_out, K, K, height_in, width_in)
    unsigned idx = i;
    const size_t y = idx % op.width_in;
    idx /= op.width_in;
    const size_t x = idx % op.height_in;
    idx /= op.height_in;
    const size_t k2 = idx % op.kernel;
    idx /= op.kernel;
    const size_t k1 = idx % op.kernel;
    idx /= op.kernel;
    const size_t o = idx % op.channels_out;

    if (y + op.padding) < k1 {
        return;
    }
    const size_t oh_mul_s = y + op.padding - k1;
    if oh_mul_s % op.stride != 0 {
        return;
    }
    const size_t oh = oh_mul_s / op.stride;

    if (x + op.padding) < k2 {
        return;
    }
    const size_t ow_mul_s = x + op.padding - k2;
    if ow_mul_s % op.stride != 0 {
        return;
    }
    const size_t ow = ow_mul_s / op.stride;

    patches[i] = image[o * image_strides[0] + oh * image_strides[1]  + ow * image_strides[2]];
}

extern "C" __global__ void transpose_filters(

) {

}

extern "C" __global__ void sum_transposed_filters() {

}