mod cpu_kernel;

use crate::{
    arrays::{Dtype, Shape},
    devices::{Device, HasErr},
    gradients::Tape,
    tensor::Tensor,
};

use super::ops::{try_unary_op, UnaryKernel};

/// [Exponential function (exp)](https://en.wikipedia.org/wiki/Natural_logarithm). `e^t`
///
/// It's derivative is itself! `e^t`.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = tensor([-1.0, 0.0, 1.0, 2.0]);
///
/// // use function version
/// let r = exp(t.clone());
///
/// // or the tensor method!
/// let r2 = t.exp();
/// ```
pub trait TryExp: HasErr {
    fn exp(self) -> Self {
        self.try_exp().unwrap()
    }
    fn try_exp(self) -> Result<Self, Self::Err>;
}

#[derive(Debug, Default, Copy, Clone)]
pub(super) struct ExpKernelOp;

impl<S: Shape, E: Dtype, D: Device, T: Tape<D>> TryExp for Tensor<S, E, D, T>
where
    D: UnaryKernel<ExpKernelOp, S, S, E>,
{
    fn try_exp(self) -> Result<Self, Self::Err> {
        try_unary_op(Default::default(), self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{devices::AsArray, tensor::*, tensor_ops::*, tests::build_test_device};

    #[test]
    fn test_exp() {
        let dev = build_test_device!();
        let x = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.trace().exp();
        assert_eq!(
            r.as_array(),
            [0.13533528, 0.36787945, 1.0, std::f32::consts::E, 7.389056]
        );
        let g = r.mean().backward();
        assert_eq!(
            g.get(&x).as_array(),
            [0.027067056, 0.07357589, 0.2, 0.54365635, 1.4778112]
        );
    }
}
