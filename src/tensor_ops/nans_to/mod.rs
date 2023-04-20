mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use super::ops::{try_unary_op, UnaryKernel};
use crate::{shapes::*, tensor::*};

#[repr(C)]
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
pub fn nans_to<S: Shape, E: Dtype, D: UnaryKernel<NansToKernelOp<E>, E>, T: Tape<E, D>>(
    t: Tensor<S, E, D, T>,
    value: impl Into<E>,
) -> Tensor<S, E, D, T> {
    t.nans_to(value)
}

impl<S: Shape, E: Dtype, D: UnaryKernel<NansToKernelOp<E>, E>, T: Tape<E, D>> Tensor<S, E, D, T> {
    /// See [nans_to]
    pub fn nans_to(self, value: impl Into<E>) -> Self {
        self.try_nans_to(value).unwrap()
    }
    /// See [nans_to]
    pub fn try_nans_to(self, value: impl Into<E>) -> Result<Self, D::Err> {
        let value = value.into();
        try_unary_op(NansToKernelOp(value), self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tensor_ops::*, tests::*};

    #[test]
    fn test_nans_1d() {
        let dev: TestDevice = Default::default();
        let t: Tensor<_, TestDtype, _> = dev.tensor([1.0, TestDtype::NAN, -TestDtype::NAN, 4.0]);
        let r = t.leaky_trace().nans_to(0.0);
        assert_close_to_literal!(r, [1.0, 0.0, 0.0, 4.0]);
        // NOTE: .exp() so we cover case where nans_to() needs to use result grad
        let g = r.exp().mean().backward();
        assert_close_to_literal!(g.get(&t), [0.67957044, 0.0, 0.0, 13.649537]);
    }
}
