mod cpu_kernel;

use super::ops::{try_unary_op, UnaryKernel};
use crate::{gradients::Tape, shapes::*, tensor::Tensor};

#[derive(Debug, Clone, Copy)]
pub struct DropoutKernelOp {
    pub seed: u64,
    pub prob: f32,
}

/// Zeros elements with probability `p` and scales all elements by `1 / (1 - p)`.
///
/// Described in paper: [Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/abs/1207.0580)
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let t = dev.tensor([1.0, 2.0, 3.0, 4.0]);
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
pub fn dropout<S: Shape, E: Dtype, D: UnaryKernel<DropoutKernelOp, E>, T: Tape<D>>(
    t: Tensor<S, E, D, T>,
    prob: f32,
) -> Tensor<S, E, D, T> {
    t.dropout(prob)
}

impl<S: Shape, E: Dtype, D: UnaryKernel<DropoutKernelOp, E>, T: Tape<D>> Tensor<S, E, D, T> {
    /// See [dropout]
    pub fn dropout(self, prob: f32) -> Self {
        self.try_dropout(prob).unwrap()
    }
    /// See [dropout]
    pub fn try_dropout(self, prob: f32) -> Result<Self, D::Err> {
        let seed = self.device.random_u64();
        try_unary_op(DropoutKernelOp { seed, prob }, self)
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::*;
    use crate::tensor_ops::*;
    use crate::tests::{assert_close, TestDevice};

    #[test]
    fn test_dropout_all_0d() {
        let dev: TestDevice = Default::default();
        let t = dev.tensor(3.0);
        let r = t.trace().dropout(1.0);
        assert_eq!(r.array(), 0.0);
        let g = r.backward();
        assert_eq!(g.get(&t).array(), 0.0);
    }

    #[test]
    fn test_dropout_none_0d() {
        let dev: TestDevice = Default::default();
        let t = dev.tensor(3.0);
        let r = t.trace().dropout(0.0);
        assert_eq!(r.array(), 3.0);
        let g = r.backward();
        assert_eq!(g.get(&t).array(), 1.0);
    }

    #[test]
    fn test_dropout_1d_with_non_positive_values() {
        let dev: TestDevice = Default::default();
        let t = dev.tensor([0.0, 2.0, -3.0, -4.0, 0.0]);
        let r = t.trace().dropout(0.5);
        assert_eq!(r.array(), [0.0, 4.0, -6.0, 0.0, 0.0]);
        let g = r.mean().backward();
        assert_eq!(g.get(&t).array(), [0.4, 0.4, 0.4, 0.0, 0.0]);
    }

    #[test]
    fn test_dropout_2d() {
        let dev: TestDevice = Default::default();
        let t = dev.tensor([[0.05, 0.1, -0.2], [0.3, -0.4, 0.5]]);
        let r = t.trace().dropout(0.6);
        assert_close(&r.array(), &[[0.125, 0.25, -0.5], [0.0, 0.0, 1.25]]);
        // NOTE: .exp() so we ensure result grad is used properly
        let g = r.exp().mean().backward();
        assert_eq!(
            g.get(&t).array(),
            [[0.47214523, 0.5350107, 0.2527211], [0.0, 0.0, 1.4543099]]
        );
    }
}
