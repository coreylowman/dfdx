mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use super::ops::{try_unary_op, UnaryKernel};
use crate::{
    shapes::*,
    tensor::{Tape, Tensor},
};

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct SquareKernelOp;

/// `t^2`
///
/// The derivative is `2 * t`.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let t = dev.tensor([-1.0, 0.0, 1.0, 2.0]);
/// let r = t.square();
/// ```
pub fn square<S: Shape, E: Dtype, D: UnaryKernel<SquareKernelOp, E>, T: Tape<E, D>>(
    t: Tensor<S, E, D, T>,
) -> Tensor<S, E, D, T> {
    t.square()
}

impl<S: Shape, E: Dtype, D: UnaryKernel<SquareKernelOp, E>, T: Tape<E, D>> Tensor<S, E, D, T> {
    /// See [square]
    pub fn square(self) -> Self {
        self.try_square().unwrap()
    }
    /// See [square]
    pub fn try_square(self) -> Result<Self, D::Err> {
        try_unary_op(SquareKernelOp, self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tensor_ops::*, tests::*};

    #[test]
    fn test_square() {
        let dev: TestDevice = Default::default();
        let x: Tensor<_, TestDtype, _> = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.trace().square();
        assert_eq!(r.array(), [4.0, 1.0, 0.0, 1.0, 4.0]);
        let g = r.mean().backward();
        assert_eq!(g.get(&x).array(), [-0.8, -0.4, 0.0, 0.4, 0.8]);
    }
}
