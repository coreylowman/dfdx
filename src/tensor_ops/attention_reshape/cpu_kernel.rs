use super::*;
use crate::{gradients::NoneTape, tensor::cpu::Cpu};
use std::vec;

impl super::AttentionReshapeKernel<f32> for Cpu {
    fn forward<const THREE_HIDDEN_DIM: usize, const NUM_HEADS: usize, const HEAD_DIM: usize>(
        &self,
        qkv: &Tensor<(usize, Const<THREE_HIDDEN_DIM>), f32, Self>,
        past_key: &Tensor<(Const<NUM_HEADS>, Const<HEAD_DIM>, usize), f32, Self>,
        past_value: &Tensor<(Const<NUM_HEADS>, usize, Const<HEAD_DIM>), f32, Self>,
    ) -> Result<
        (
            Tensor<(Const<NUM_HEADS>, usize, Const<HEAD_DIM>), f32, Self>,
            Tensor<(Const<NUM_HEADS>, Const<HEAD_DIM>, usize), f32, Self>,
            Tensor<(Const<NUM_HEADS>, usize, Const<HEAD_DIM>), f32, Self>,
        ),
        Self::Err,
    > {
        let sequence_length = qkv.shape().0;
        let past_sequence_length = past_key.shape().2;
        let total_length = sequence_length.size() + past_sequence_length.size();
        let dev = qkv.device.clone();

        let mut q: Tensor<(Const<NUM_HEADS>, usize, Const<HEAD_DIM>), f32, Self, NoneTape> =
            dev.zeros_like(&(Const, sequence_length, Const));
        let mut k: Tensor<(Const<NUM_HEADS>, Const<HEAD_DIM>, usize), f32, Self, NoneTape> =
            dev.zeros_like(&(Const, Const, total_length));
        let mut v: Tensor<(Const<NUM_HEADS>, usize, Const<HEAD_DIM>), f32, Self, NoneTape> =
            dev.zeros_like(&(Const, total_length, Const));
        let mut q_vec = vec![0.0; q.shape().num_elements()];
        let mut k_vec = vec![0.0; k.shape().num_elements()];
        let mut v_vec = vec![0.0; v.shape().num_elements()];
        let mut past_key_vec = vec![0.0; past_key.shape().num_elements()];
        let mut past_value_vec = vec![0.0; past_value.shape().num_elements()];
        let mut qkv_vec = vec![0.0; qkv.shape().num_elements()];
        past_key.copy_into(past_key_vec.as_mut_slice());
        past_value.copy_into(&mut past_value_vec);
        qkv.copy_into(&mut qkv_vec);

        let head_dim = HEAD_DIM;
        let hidden_dim = THREE_HIDDEN_DIM / 3;
        let num_heads = NUM_HEADS;
        (0..num_heads).for_each(|i| {
            (0..sequence_length.size()).for_each(|j| {
                (0..head_dim).for_each(|k| {
                    let index = j * hidden_dim * 3 + i * head_dim + k;
                    let out_index = i * sequence_length.size() * head_dim + j * head_dim + k;
                    let value = qkv_vec[index];
                    q_vec[out_index] = value;
                });
            });
        });
        (0..num_heads).for_each(|i| {
            (0..past_sequence_length.size() + sequence_length.size()).for_each(|j| {
                (0..head_dim).for_each(|k| {
                    let in_index_k =
                        i * (past_sequence_length.size() + sequence_length.size()) * head_dim
                            + k * (past_sequence_length.size() + sequence_length.size())
                            + j;

                    let in_index_v =
                        i * (past_sequence_length.size() + sequence_length.size()) * head_dim
                            + j * head_dim
                            + k;
                    if j < past_sequence_length.size() {
                        let k_index = i * past_sequence_length.size() * head_dim
                            + k * past_sequence_length.size()
                            + j;
                        let k_value = past_key_vec[k_index];
                        k_vec[in_index_k] = k_value;

                        let v_index = i * past_sequence_length.size() * head_dim + j * head_dim + k;
                        let v_value = past_value_vec[v_index];
                        v_vec[in_index_v] = v_value;
                    } else {
                        let sj = j - past_sequence_length.size();
                        let k_index = sj * hidden_dim * 3 + i * head_dim + hidden_dim + k;
                        let k_value = qkv_vec[k_index];
                        k_vec[in_index_k] = k_value;

                        let v_index = sj * hidden_dim * 3 + i * head_dim + hidden_dim * 2 + k;
                        let v_value = qkv_vec[v_index];
                        v_vec[in_index_v] = v_value;
                    }
                });
            });
        });
        q.copy_from(&q_vec);
        k.copy_from(&k_vec);
        v.copy_from(&v_vec);
        Ok((q, k, v))
    }
}
