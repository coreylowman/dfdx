mod cpu_kernel;

use super::ops::{try_unary_op, UnaryKernel};
use crate::{gradients::Tape, shapes::*, tensor::Tensor};

#[derive(Debug, Default, Copy, Clone)]
pub struct SqrtKernelOp;

/// `√t` or `t^0.5`
///
/// The derivative is `0.5 / (t ^ 0.5)`.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let t = dev.tensor([-1.0, 0.0, 1.0, 2.0]);
/// let r = t.sqrt();
/// ```
pub fn sqrt<S: Shape, E: Dtype, D: UnaryKernel<SqrtKernelOp, E>, T: Tape<D>>(
    t: Tensor<S, E, D, T>,
) -> Tensor<S, E, D, T> {
    t.sqrt()
}

impl<S: Shape, E: Dtype, D: UnaryKernel<SqrtKernelOp, E>, T: Tape<D>> Tensor<S, E, D, T> {
    /// See [sqrt]
    pub fn sqrt(self) -> Self {
        self.try_sqrt().unwrap()
    }
    /// See [sqrt]
    pub fn try_sqrt(self) -> Result<Self, D::Err> {
        try_unary_op(SqrtKernelOp, self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tensor_ops::*, tests::build_test_device};

    #[test]
    fn test_sqrt() {
        let dev = build_test_device!();
        let x = dev.tensor([-1.0, 0.0, 1.0, 4.0]);
        let r = x.trace().sqrt();
        assert!(r.array()[0].is_nan());
        assert_eq!(r.array()[1..], [0.0, 1.0, 2.0]);
        let g = r.mean().backward();
        let g = g.get(&x).array();
        assert!(g[0].is_nan());
        assert_eq!(g[1..], [f32::INFINITY, 0.5 / 4.0, 0.25 / 4.0]);
    }
}
