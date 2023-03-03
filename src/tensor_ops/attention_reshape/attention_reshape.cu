#include "cuda_utils.cuh"

struct AttentionReshapeOp {
    size_t numel;
    size_t num_heads;
    size_t head_dim;
    size_t sequence_length;
    size_t past_length;
};

template<typename T>
__device__ void attention_reshape(
    const AttentionReshapeOp op,
    const T *qkv,
    const T *past_key,
    const T *past_value,
    T *query,
    T *key,
    T *value
) {
    size_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= op.numel) {
        return;
    }
    const size_t hidden_dim = op.num_heads * op.head_dim;
    const size_t total_length = op.sequence_length + op.past_length;
    const size_t q_length = hidden_dim * op.sequence_length;
    const size_t k_length = hidden_dim * total_length;
    if (n < q_length){
        const size_t k = n % op.head_dim;
        const size_t j = (n / op.head_dim) % op.sequence_length;
        const size_t i = n / op.head_dim / op.sequence_length;
        const size_t qkv_index = j * hidden_dim * 3 + i * op.head_dim + k;
        const size_t q_index = n;
        query[q_index] = qkv[qkv_index];
    } else if (n < q_length + k_length){
        const size_t k_index = n - q_length;
              size_t k = k_index % total_length;
        const size_t j = (k_index / total_length) % op.head_dim;
        const size_t i = k_index / op.head_dim / total_length;

        if (k < op.past_length){
            const size_t past_key_index = i * op.past_length * op.head_dim + j * op.past_length + k;
            key[k_index] = past_key[past_key_index];
        }else{
            k -= op.past_length;
            const size_t qkv_index = k * hidden_dim * 3 + i * op.head_dim + j + hidden_dim;
            key[k_index] = qkv[qkv_index];
        }
    } else{
        const size_t v_index = n - k_length - q_length;
        const size_t k = v_index % op.head_dim;
              size_t j = (v_index / op.head_dim) % total_length;
        const size_t i = v_index / op.head_dim / total_length;

        if (j < op.past_length){
            const size_t past_value_index = i * op.past_length * op.head_dim + j * op.head_dim + k;
            value[v_index] = past_value[past_value_index];
        }else{
            j -= op.past_length;
            const size_t qkv_index = j * hidden_dim * 3 + i * op.head_dim + k + 2 * hidden_dim;
            value[v_index] = qkv[qkv_index];
        }
    }
}


extern "C" __global__ void attention_reshape_f32(
    const AttentionReshapeOp op,
    const float *qkv,
    const float *past_key,
    const float *past_value,
    float *query,
    float *key,
    float *value
) {
    attention_reshape(op, qkv, past_key, past_value, query, key, value);
}

extern "C" __global__ void attention_reshape_f64(
    const AttentionReshapeOp op,
    const double *qkv,
    const double *past_key,
    const double *past_value,
    double *query,
    double *key,
    double *value
) {
    attention_reshape(op, qkv, past_key, past_value, query, key, value);
}
