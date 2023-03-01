use super::*;
use crate::tensor::cuda::{Cuda, CudaArray};
use cudarc::driver::{LaunchAsync, LaunchConfig};
use std::sync::Arc;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/attention_reshape.ptx"));

impl super::AttentionReshapeKernel<f32> for Cuda {
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
        let mod_ = "attention_reshape_f32";
        let fns = "attention_reshape_f32";
        if !self.dev.has_func(mod_, fns) {
            self.dev.load_ptx(PTX.into(), mod_, &[fns])?;
        }
        let f = self.dev.get_func(mod_, fns).unwrap();
        let seq = qkv.shape().0;
        let sequence_length = seq.size();
        let past_length = past_key.shape().2;
        let total_length = sequence_length + past_length;
        let head_dim = HEAD_DIM;
        let num_heads = NUM_HEADS;

        let q_shape = (Const, seq, Const);
        let mut q_storage = self.dev.alloc_zeros_async::<f32>(q_shape.num_elements())?;

        let k_shape = (Const, Const, total_length);
        let mut k_storage = self.dev.alloc_zeros_async::<f32>(k_shape.num_elements())?;

        let v_shape = (Const, total_length, Const);
        let mut v_storage = self.dev.alloc_zeros_async::<f32>(v_shape.num_elements())?;

        let numel = q_shape.num_elements() + k_shape.num_elements() + v_shape.num_elements();
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let params = (
            numel,                            // const size_t numel,
            num_heads,                        // const size_t num_heads,
            head_dim,                         // const size_t head_dim,
            sequence_length,                  // const size_t sequence_length,
            past_length,                      // const size_t past_length,
            qkv.storage.data.as_ref(),        // const float *qkv,
            past_key.storage.data.as_ref(),   // const float *past_key,
            past_value.storage.data.as_ref(), // const float *past_value,
            &mut q_storage,                   // float *q,
            &mut k_storage,                   // float *k,
            &mut v_storage,                   // float *v
        );

        unsafe { f.launch_async(cfg, params) }?;
        let device = qkv.device.clone();
        let q: Tensor<(Const<NUM_HEADS>, usize, Const<HEAD_DIM>), f32, Self> =
            device.upgrade(CudaArray {
                data: Arc::new(q_storage),
                shape: q_shape,
                strides: q_shape.strides(),
            });
        let k: Tensor<(Const<NUM_HEADS>, Const<HEAD_DIM>, usize), f32, Self> =
            device.upgrade(CudaArray {
                data: Arc::new(k_storage),
                shape: k_shape,
                strides: k_shape.strides(),
            });
        let v: Tensor<(Const<NUM_HEADS>, usize, Const<HEAD_DIM>), f32, Self> =
            device.upgrade(CudaArray {
                data: Arc::new(v_storage),
                shape: v_shape,
                strides: v_shape.strides(),
            });
        Ok((q, k, v))
    }
}
