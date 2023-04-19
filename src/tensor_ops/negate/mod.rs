mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use super::ops::{try_unary_op, UnaryKernel};
use crate::{shapes::*, tensor::*};

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct NegateKernelOp;

/// Negates all elements.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a = dev.tensor([-2.0, 0.0, 5.0]);
/// let r = -a;
/// assert_eq!(r.array(), [2.0, 0.0, -5.0]);
/// ```
pub fn negate<S: Shape, E: Dtype, D: UnaryKernel<NegateKernelOp, E>, T: Tape<E, D>>(
    t: Tensor<S, E, D, T>,
) -> Tensor<S, E, D, T> {
    t.negate()
}

impl<S: Shape, E: Dtype, D: UnaryKernel<NegateKernelOp, E>, T: Tape<E, D>> Tensor<S, E, D, T> {
    pub fn negate(self) -> Self {
        self.try_negate().unwrap()
    }
    pub fn try_negate(self) -> Result<Self, D::Err> {
        try_unary_op(NegateKernelOp, self)
    }
}

impl<S: Shape, E: Dtype, D: UnaryKernel<NegateKernelOp, E>, T: Tape<E, D>> std::ops::Neg
    for Tensor<S, E, D, T>
{
    type Output = Self;
    fn neg(self) -> Self::Output {
        self.negate()
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tensor_ops::*, tests::*};

    #[test]
    fn test_1d_neg() {
        let dev: TestDevice = Default::default();
        let a: Tensor<_, TestDtype, _> = dev.tensor([-2.0, 0.0, 5.0]);
        let r = -(a.leaky_trace());
        assert_close_to_literal!(r, [2.0, 0.0, -5.0]);
        // NOTE: .exp() so we can make sure neg is using result grad properly
        let g = r.exp().mean().backward();
        assert_close_to_literal!(g.get(&a), [-2.463019, -0.33333334, -0.0022459824]);
    }
}
