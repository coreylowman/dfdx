mod cpu_kernel;

use super::{ops::try_unary_op, Device};
use crate::{arrays::*, gradients::Tape, tensor::Tensor};

#[derive(Debug, Default, Copy, Clone)]
pub struct NegateKernelOp;

/// Negates all elements.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let a: Tensor1D<3> = tensor([-2.0, 0.0, 5.0]);
/// let r = -a; // or negate(a);
/// assert_eq!(r.as_array(), [2.0, 0.0, -5.0]);
/// ```
pub fn negate<S: Shape, E: Dtype, D: Device<E>, T: Tape<D>>(
    t: Tensor<S, E, D, T>,
) -> Tensor<S, E, D, T> {
    t.negate()
}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<D>> Tensor<S, E, D, T> {
    pub fn negate(self) -> Self {
        self.try_negate().unwrap()
    }
    pub fn try_negate(self) -> Result<Self, D::Err> {
        try_unary_op(NegateKernelOp, self)
    }
}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<D>> std::ops::Neg for Tensor<S, E, D, T> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        self.negate()
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::storage::AsArray, tensor::*, tensor_ops::*, tests::build_test_device};

    #[test]
    fn test_1d_neg() {
        let dev = build_test_device!();
        let a = dev.tensor([-2.0, 0.0, 5.0]);
        let r = -(a.trace());
        assert_eq!(r.as_array(), [2.0, 0.0, -5.0]);
        // NOTE: .exp() so we can make sure neg is using result grad properly
        let g = r.exp().mean().backward();
        assert_eq!(
            g.get(&a).as_array(),
            [-2.463019, -0.33333334, -0.0022459824]
        );
    }
}
