mod cpu_kernel;

use crate::{
    arrays::{Dtype, Shape},
    devices::{DeviceStorage, HasErr},
    gradients::Tape,
    tensor::Tensor,
};

use super::ops::{try_unary_op, UnaryKernel};

/// `t^2`
///
/// The derivative is `2 * t`.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = tensor([-1.0, 0.0, 1.0, 2.0]);
///
/// // use function version
/// let r = square(t.clone());
///
/// // or the tensor method!
/// let r2 = t.square();
/// ```
pub trait TrySquare: HasErr {
    fn square(self) -> Self {
        self.try_square().unwrap()
    }
    fn try_square(self) -> Result<Self, Self::Err>;
}

#[derive(Debug, Default, Copy, Clone)]
pub(super) struct SquareKernelOp;

impl<S: Shape, E: Dtype, D: DeviceStorage, T: Tape<D>> TrySquare for Tensor<S, E, D, T>
where
    D: UnaryKernel<SquareKernelOp, S, S, E>,
{
    fn try_square(self) -> Result<Self, Self::Err> {
        try_unary_op(Default::default(), self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{devices::AsArray, tensor::*, tensor_ops::*, tests::build_test_device};

    #[test]
    fn test_square() {
        let dev = build_test_device!();
        let x = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.trace().square();
        assert_eq!(r.as_array(), [4.0, 1.0, 0.0, 1.0, 4.0]);
        let g = r.mean().backward();
        assert_eq!(g.get(&x).as_array(), [-0.8, -0.4, 0.0, 0.4, 0.8]);
    }
}
