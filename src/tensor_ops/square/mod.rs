mod cpu_kernel;

use super::{ops::try_unary_op, Device};
use crate::{arrays::*, gradients::Tape, tensor::Tensor};

#[derive(Debug, Default, Copy, Clone)]
pub struct SquareKernelOp;

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
pub fn square<S: Shape, E: Dtype, D: Device<E>, T: Tape<D>>(
    t: Tensor<S, E, D, T>,
) -> Tensor<S, E, D, T> {
    t.square()
}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<D>> Tensor<S, E, D, T> {
    pub fn square(self) -> Self {
        self.try_square().unwrap()
    }
    pub fn try_square(self) -> Result<Self, D::Err> {
        try_unary_op(SquareKernelOp, self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::storage::AsArray, tensor::*, tensor_ops::*, tests::build_test_device};

    #[test]
    fn test_square() {
        let dev = build_test_device!();
        let x = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.trace().square();
        assert_eq!(r.array(), [4.0, 1.0, 0.0, 1.0, 4.0]);
        let g = r.mean().backward();
        assert_eq!(g.get(&x).array(), [-0.8, -0.4, 0.0, 0.4, 0.8]);
    }
}
