#include "cuda_utils.cuh"
extern "C" __global__ void attention_reshape_f32(
    const size_t numel,
    const size_t num_heads,
    const size_t head_dim,
    const size_t sequence_length,
    const size_t past_length,
    const float *qkv,
    const float *past_key,
    const float *past_value,
    float *query,
    float *key,
    float *value
) {
    size_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= numel) {
        return;
    }
    const size_t hidden_dim = num_heads * head_dim;
    const size_t total_length = sequence_length + past_length;
    const size_t q_length = hidden_dim * sequence_length;
    const size_t k_length = hidden_dim * total_length;
    if (n < q_length){
        const size_t k = n % head_dim;
        const size_t j = (n / head_dim) % sequence_length;
        const size_t i = n / head_dim / sequence_length;
        const size_t qkv_index = j * hidden_dim * 3 + i * head_dim + k;
        const size_t q_index = n;
        query[q_index] = qkv[qkv_index];
    } else if (n < q_length + k_length){
        const size_t k_index = n - q_length;
              size_t k = k_index % total_length;
        const size_t j = (k_index / total_length) % head_dim;
        const size_t i = k_index / head_dim / total_length;

        if (k < past_length){
            const size_t past_key_index = i * past_length * head_dim + j * past_length + k;
            key[k_index] = past_key[past_key_index];
            // key[k_index] = 0;
        }else{
            k -= past_length;
            const size_t qkv_index = k * hidden_dim * 3 + i * head_dim + j + hidden_dim;
            key[k_index] = qkv[qkv_index];
            // key[k_index] = 0;
        }
    } else{
        const size_t v_index = n - k_length - q_length;
        const size_t k = v_index % head_dim;
              size_t j = (v_index / head_dim) % total_length;
        const size_t i = v_index / head_dim / total_length;

        if (j < past_length){
            const size_t past_value_index = i * past_length * head_dim + j * head_dim + k;
            value[v_index] = past_value[past_value_index];
            // value[v_index] = 0;
        }else{
            j -= past_length;
            const size_t qkv_index = j * hidden_dim * 3 + i * head_dim + k + 2 * hidden_dim;
            value[v_index] = qkv[qkv_index];
            // value[v_index] = 0;
        }
    }
}
