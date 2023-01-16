struct ConvParams {
    size_t stride;
    size_t padding;
};

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

extern "C" __global__ void unfold_input_into_patches(
    const ConvParams params,
    const float *image,
    const size_t *image_dims, // NOTE: assumes num_dims == 3
    const size_t *image_strides,
    float *patches,
    const size_t numel,
    const size_t *patches_dims, // NOTE: assumes num_dims == 5
    const size_t *patches_strides
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }

    const size_t height_in = image_dims[1];
    const size_t width_in = image_dims[2];

    unsigned idx = i;
    const size_t ow = idx % patches_dims[4];
    idx /= patches_dims[4];
    const size_t oh = idx % patches_dims[3];
    idx /= patches_dims[3];
    const size_t k2 = idx % patches_dims[2];
    idx /= patches_dims[2];
    const size_t k1 = idx % patches_dims[1];
    idx /= patches_dims[1];
    const size_t c = idx % patches_dims[0];

    const size_t y_plus_p = oh * params.stride + k1;
    if (y_plus_p < params.padding) {
        return;
    }
    const size_t y = y_plus_p - params.padding;

    const size_t x_plus_p = ow * params.stride + k2;
    if (x_plus_p < params.padding) {
        return;
    }
    const size_t x = x_plus_p - params.padding;

    if (y >= height_in || x >= width_in) {
        return;
    }

    patches[i] = image[c * image_strides[0] + y * image_strides[1] + x * image_strides[2]];
}