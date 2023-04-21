mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use crate::{
    shapes::*,
    tensor::{DeviceStorage, PutTape, SplitTape, Tape, Tensor},
};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DropoutKernelOp<F> {
    pub seed: u64,
    pub prob: F,
}

pub trait DropoutKernel<E: Dtype>: DeviceStorage {
    fn forward<S: Shape>(
        &self,
        op: DropoutKernelOp<E>,
        inp: &Tensor<S, E, Self>,
    ) -> Result<Tensor<S, E, Self>, Self::Err>;
    fn backward<S: Shape>(
        &self,
        op: DropoutKernelOp<E>,
        inp: &Tensor<S, E, Self>,
        grad_inp: &mut Self::Vec<E>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err>;
}

/// Zeros elements with probability `p` and scales all elements by `1 / (1 - p)`.
///
/// Described in paper: [Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/abs/1207.0580)
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let t = dev.tensor([1.0f32, 2.0, 3.0, 4.0]);
/// let r = t.dropout(0.5);
/// assert_eq!(r.array(), [2.0, 4.0, 6.0, 0.0]);
/// ```
///
/// ### Implementation details:
///
/// To reduce memory usage, this function first samples a u64 seed from `rng`,
/// and then instantiates two identical [rand::rngs::StdRng] with that seed. These rngs
/// are used in both the forward pass and backward pass to generate identical
/// random numbers, so the masking is the same for both.
pub fn dropout<S: Shape, E: Dtype, D: DropoutKernel<E>, T: Tape<E, D>>(
    t: Tensor<S, E, D, T>,
    prob: impl Into<E>,
) -> Tensor<S, E, D, T> {
    t.dropout(prob)
}

impl<S: Shape, E: Dtype, D: DropoutKernel<E>, T: Tape<E, D>> Tensor<S, E, D, T> {
    /// See [dropout]
    pub fn dropout(self, prob: impl Into<E>) -> Self {
        self.try_dropout(prob).unwrap()
    }
    /// See [dropout]
    pub fn try_dropout(self, prob: impl Into<E>) -> Result<Self, D::Err> {
        let seed = self.device.random_u64();
        let prob = prob.into();
        let op = DropoutKernelOp { seed, prob };
        let (inp, mut tape) = self.split_tape();
        let out = inp.device.forward(op, &inp)?;
        let inp_ghost = inp.ghost();
        let out_ghost = out.ghost();
        tape.add_backward_op(move |grads| {
            grads.try_alloc_for(&inp_ghost)?;
            grads.try_alloc_for(&out_ghost)?;
            let (grad_inp, grad_out) = grads.mut_and_ref(&inp_ghost, &out_ghost);
            inp.device.backward(op, &inp, grad_inp, grad_out)
        });
        Ok(out.put_tape(tape))
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tensor_ops::*, tests::*};

    #[test]
    fn test_dropout_all_0d() {
        let dev: TestDevice = Default::default();
        let t: Tensor<_, f32, _> = dev.tensor(3.0);
        let r = t.leaky_trace().dropout(1.0);
        assert_close_to_literal!(r, 0.0);
        let g = r.backward();
        assert_close_to_literal!(g.get(&t), 0.0);
    }

    #[test]
    fn test_dropout_none_0d() {
        let dev: TestDevice = Default::default();
        let t: Tensor<_, f32, _> = dev.tensor(3.0);
        let r = t.leaky_trace().dropout(0.0);
        assert_close_to_literal!(r, 3.0);
        let g = r.backward();
        assert_close_to_literal!(g.get(&t), 1.0);
    }

    #[test]
    fn test_dropout_1d_with_non_positive_values() {
        let dev: TestDevice = Default::default();
        let t: Tensor<_, f32, _> = dev.tensor([0.0, 2.0, -3.0, -4.0, 0.0]);
        let r = t.leaky_trace().dropout(0.5);
        assert_close_to_literal!(r, [0.0, 4.0, -6.0, 0.0, 0.0]);
        let g = r.mean().backward();
        assert_close_to_literal!(g.get(&t), [0.4, 0.4, 0.4, 0.0, 0.0]);
    }

    #[test]
    fn test_dropout_2d() {
        let dev: TestDevice = Default::default();
        let t: Tensor<_, f32, _> = dev.tensor([[0.05, 0.1, -0.2], [0.3, -0.4, 0.5]]);
        let r = t.leaky_trace().dropout(0.6);
        assert_close_to_literal!(r, [[0.125, 0.25, -0.5], [0.0, 0.0, 1.25]]);
        // NOTE: .exp() so we ensure result grad is used properly
        let g = r.exp().mean().backward();
        assert_close_to_literal!(
            g.get(&t),
            [[0.47214523, 0.5350107, 0.2527211], [0.0, 0.0, 1.4543099]]
        );
    }
}
