use super::*;
use crate::tensor::cuda::Cuda;
use cudarc::driver::{DeviceRepr, LaunchAsync};

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/attention_reshape.ptx"));

#[repr(C)]
struct AttentionReshapeOp {
    numel: usize,
    num_heads: usize,
    head_dim: usize,
    sequence_length: usize,
    past_length: usize,
}

unsafe impl DeviceRepr for AttentionReshapeOp {}

trait HasCudaKernel<E: Unit> {
    const FN: &'static str;
}

impl HasCudaKernel<f32> for Cuda {
    const FN: &'static str = "attention_reshape_f32";
}

impl HasCudaKernel<f64> for Cuda {
    const FN: &'static str = "attention_reshape_f64";
}

impl<E: Dtype> super::AttentionReshapeKernel<E> for Cuda
where
    Self: HasCudaKernel<E>,
{
    fn forward<const THREE_HIDDEN_DIM: usize, const NUM_HEADS: usize, const HEAD_DIM: usize>(
        &self,
        qkv: &Tensor<(usize, Const<THREE_HIDDEN_DIM>), E, Self>,
        past_key: &Tensor<(Const<NUM_HEADS>, Const<HEAD_DIM>, usize), E, Self>,
        past_value: &Tensor<(Const<NUM_HEADS>, usize, Const<HEAD_DIM>), E, Self>,
    ) -> Result<
        (
            Tensor<(Const<NUM_HEADS>, usize, Const<HEAD_DIM>), E, Self>,
            Tensor<(Const<NUM_HEADS>, Const<HEAD_DIM>, usize), E, Self>,
            Tensor<(Const<NUM_HEADS>, usize, Const<HEAD_DIM>), E, Self>,
        ),
        Self::Err,
    > {
        if !self.dev.has_func(Self::FN, Self::FN) {
            self.dev.load_ptx(PTX.into(), Self::FN, &[Self::FN])?;
        }
        let f = self.dev.get_func(Self::FN, Self::FN).unwrap();
        let seq = qkv.shape().0;
        let sequence_length = seq.size();
        let past_length = past_key.shape().2;
        let total_length = sequence_length + past_length;
        let head_dim = HEAD_DIM;
        let num_heads = NUM_HEADS;

        let q_shape = (Const, seq, Const);
        let mut q_storage = self.dev.alloc_zeros::<E>(q_shape.num_elements())?;

        let k_shape = (Const, Const, total_length);
        let mut k_storage = self.dev.alloc_zeros::<E>(k_shape.num_elements())?;

        let v_shape = (Const, total_length, Const);
        let mut v_storage = self.dev.alloc_zeros::<E>(v_shape.num_elements())?;

        let numel = q_shape.num_elements() + k_shape.num_elements() + v_shape.num_elements();
        let op = AttentionReshapeOp {
            numel,
            num_heads,
            head_dim,
            sequence_length,
            past_length,
        };
        let cfg = launch_cfg(numel as u32);
        let params = (
            op,
            qkv.data.as_ref(),
            past_key.data.as_ref(),
            past_value.data.as_ref(),
            &mut q_storage,
            &mut k_storage,
            &mut v_storage,
        );

        unsafe { f.launch(cfg, params) }?;
        let q = self.build_tensor(q_shape, q_shape.strides(), q_storage);
        let k = self.build_tensor(k_shape, k_shape.strides(), k_storage);
        let v = self.build_tensor(v_shape, v_shape.strides(), v_storage);
        Ok((q, k, v))
    }
}
