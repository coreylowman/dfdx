mod cpu_kernel;

use crate::{
    arrays::{Dtype, Shape},
    devices::{Device, HasErr},
    gradients::Tape,
    tensor::Tensor,
};

use super::ops::{try_unary_op, UnaryKernel};

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
pub trait TryAbs: HasErr {
    fn abs(self) -> Self {
        self.try_abs().unwrap()
    }
    fn try_abs(self) -> Result<Self, Self::Err>;
}

#[derive(Debug, Default, Copy, Clone)]
pub(super) struct AbsKernelOp;

impl<S: Shape, E: Dtype, D: Device, T: Tape<D>> TryAbs for Tensor<S, E, D, T>
where
    D: UnaryKernel<AbsKernelOp, S, S, E>,
{
    fn try_abs(self) -> Result<Self, Self::Err> {
        try_unary_op(Default::default(), self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{devices::AsArray, tensor::*, tensor_ops::*, tests::build_test_device};

    #[test]
    fn test_abs() {
        let dev = build_test_device!();
        let x = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.trace().abs();
        assert_eq!(r.as_array(), [2.0, 1.0, 0.0, 1.0, 2.0]);
        let g = r.mean().backward();
        assert_eq!(g.get(&x).as_array(), [-0.2, -0.2, 0.0, 0.2, 0.2]);
    }
}
