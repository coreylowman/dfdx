mod cpu_kernel;

use crate::{
    arrays::{Dtype, Shape},
    devices::{Device, HasErr},
    gradients::Tape,
    tensor::Tensor,
};

use super::utils::{try_unary_op, UnaryKernel};

/// Negates all elements.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let a: Tensor1D<3> = tensor([-2.0, 0.0, 5.0]);
/// let r = -a; // or negate(a);
/// assert_eq!(r.as_array(), [2.0, 0.0, -5.0]);
/// ```
pub trait TryNegate: HasErr {
    fn negate(self) -> Self {
        self.try_negate().unwrap()
    }
    fn try_negate(self) -> Result<Self, Self::Err>;
}

#[derive(Debug, Default, Copy, Clone)]
pub(super) struct NegateKernelOp;

impl<S: Shape, E: Dtype, D: Device, T: Tape<D>> TryNegate for Tensor<S, E, D, T>
where
    D: UnaryKernel<NegateKernelOp, S, S, E>,
{
    fn try_negate(self) -> Result<Self, Self::Err> {
        try_unary_op(Default::default(), self)
    }
}

impl<S: Shape, E: Dtype, D: Device, T: Tape<D>> std::ops::Neg for Tensor<S, E, D, T>
where
    Self: TryNegate,
{
    type Output = Self;
    fn neg(self) -> Self::Output {
        self.negate()
    }
}

#[cfg(test)]
mod tests {
    use crate::{devices::AsArray, tensor::*, tensor_ops::*, tests::build_test_device};

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
