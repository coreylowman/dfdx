mod cpu_kernel;

use super::ops::{try_unary_op, UnaryKernel};
use crate::{gradients::Tape, shapes::*, tensor::Tensor};

#[derive(Debug, Clone, Copy)]
pub struct NansToKernelOp<E>(E);

/// Replaces any [std::f32::NAN] with `value`.
///
/// **Pytorch equivalent**: `t.nan_to_num(value)`
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let t = dev.tensor([1.0, f32::NAN, f32::NAN, 4.0]);
/// let r = t.nans_to(0.0);
/// assert_eq!(r.array(), [1.0, 0.0, 0.0, 4.0]);
/// ```
pub fn nans_to<S: Shape, E: Dtype, D: UnaryKernel<NansToKernelOp<E>, E>, T: Tape<D>>(
    t: Tensor<S, E, D, T>,
    value: E,
) -> Tensor<S, E, D, T> {
    t.nans_to(value)
}

impl<S: Shape, E: Dtype, D: UnaryKernel<NansToKernelOp<E>, E>, T: Tape<D>> Tensor<S, E, D, T> {
    /// See [nans_to]
    pub fn nans_to(self, value: E) -> Self {
        self.try_nans_to(value).unwrap()
    }
    /// See [nans_to]
    pub fn try_nans_to(self, value: E) -> Result<Self, D::Err> {
        try_unary_op(NansToKernelOp(value), self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tensor_ops::*, tests::TestDevice};

    #[test]
    fn test_nans_1d() {
        let dev: TestDevice = Default::default();
        let t = dev.tensor([1.0, f32::NAN, -f32::NAN, 4.0]);
        let r = t.trace().nans_to(0.0);
        assert_eq!(r.array(), [1.0, 0.0, 0.0, 4.0]);
        // NOTE: .exp() so we cover case where nans_to() needs to use result grad
        let g = r.exp().mean().backward();
        assert_eq!(g.get(&t).array(), [0.67957044, 0.0, 0.0, 13.649537]);
    }
}
