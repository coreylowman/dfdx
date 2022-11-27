mod cpu_kernel;

use crate::{
    arrays::{Dtype, Shape},
    gradients::Tape,
    tensor::Tensor,
};

use super::{device::Device, ops::try_unary_op};

#[derive(Debug, Default, Copy, Clone)]
pub struct ExpKernelOp;

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
pub fn exp<S: Shape, E: Dtype, D: Device<E>, T: Tape<D>>(
    t: Tensor<S, E, D, T>,
) -> Tensor<S, E, D, T> {
    t.exp()
}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<D>> Tensor<S, E, D, T> {
    pub fn exp(self) -> Self {
        self.try_exp().unwrap()
    }
    pub fn try_exp(self) -> Result<Self, D::Err> {
        try_unary_op(ExpKernelOp, self)
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
