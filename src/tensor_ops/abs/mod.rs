mod cpu_kernel;

use super::{ops::try_unary_op, Device};
use crate::{arrays::*, gradients::Tape, tensor::Tensor};

#[derive(Debug, Default, Copy, Clone)]
pub struct AbsKernelOp;

/// [Absolute value (abs)](https://en.wikipedia.org/wiki/Absolute_value). `|t|`
///
/// The derivative is -1.0 for t < 0, 0 for t == 0, and 1.0 for t > 0.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = tensor([-1.0, 0.0, 1.0, 2.0]);
///
/// // use function version
/// let r = abs(t.clone());
///
/// // or the tensor method!
/// let r2 = t.abs();
/// ```
pub fn abs<S: Shape, E: Dtype, D: Device<E>, T: Tape<D>>(
    t: Tensor<S, E, D, T>,
) -> Tensor<S, E, D, T> {
    t.abs()
}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<D>> Tensor<S, E, D, T> {
    pub fn abs(self) -> Self {
        self.try_abs().unwrap()
    }
    pub fn try_abs(self) -> Result<Self, D::Err> {
        try_unary_op(AbsKernelOp, self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tensor_ops::*, tests::build_test_device};

    #[test]
    fn test_abs() {
        let dev = build_test_device!();
        let x = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.trace().abs();
        assert_eq!(r.array(), [2.0, 1.0, 0.0, 1.0, 2.0]);
        let g = r.mean().backward();
        assert_eq!(g.get(&x).array(), [-0.2, -0.2, 0.0, 0.2, 0.2]);
    }
}
