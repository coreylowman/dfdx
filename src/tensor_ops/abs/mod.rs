mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use super::ops::{try_unary_op, UnaryKernel};
use crate::{shapes::*, tensor::*};

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct AbsKernelOp;

/// [Absolute value (abs)](https://en.wikipedia.org/wiki/Absolute_value). `|t|`
///
/// The derivative is -1.0 for t < 0, 0 for t == 0, and 1.0 for t > 0.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let t = dev.tensor([-1.0, 0.0, 1.0, 2.0]);
/// let r = t.abs();
/// assert_eq!(r.array(), [1.0, 0.0, 1.0, 2.0]);
/// ```
pub fn abs<S: Shape, E: Dtype, D: UnaryKernel<AbsKernelOp, E>, T: Tape<E, D>>(
    t: Tensor<S, E, D, T>,
) -> Tensor<S, E, D, T> {
    t.abs()
}

impl<S: Shape, E: Dtype, D: UnaryKernel<AbsKernelOp, E>, T: Tape<E, D>> Tensor<S, E, D, T> {
    /// See [abs]
    pub fn abs(self) -> Self {
        self.try_abs().unwrap()
    }
    /// See [abs]
    pub fn try_abs(self) -> Result<Self, D::Err> {
        try_unary_op(AbsKernelOp, self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tensor_ops::*, tests::*};

    #[test]
    fn test_abs() {
        let dev: TestDevice = Default::default();
        let x: Tensor<_, TestDtype, _> = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.leaky_trace().abs();
        assert_close_to_literal!(r, [2.0, 1.0, 0.0, 1.0, 2.0]);
        let g = r.mean().backward();
        assert_close_to_literal!(g.get(&x), [-0.2, -0.2, 0.0, 0.2, 0.2]);
    }
}
