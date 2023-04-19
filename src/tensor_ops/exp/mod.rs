mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use super::ops::{try_unary_op, UnaryKernel};
use crate::{shapes::*, tensor::*};

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct ExpKernelOp;

/// [Exponential function (exp)](https://en.wikipedia.org/wiki/Natural_logarithm). `e^t`
///
/// It's derivative is itself! `e^t`.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let t = dev.tensor([-1.0, 0.0, 1.0, 2.0]);
/// let r = t.exp();
/// ```
pub fn exp<S: Shape, E: Dtype, D: UnaryKernel<ExpKernelOp, E>, T: Tape<E, D>>(
    t: Tensor<S, E, D, T>,
) -> Tensor<S, E, D, T> {
    t.exp()
}

impl<S: Shape, E: Dtype, D: UnaryKernel<ExpKernelOp, E>, T: Tape<E, D>> Tensor<S, E, D, T> {
    /// See [exp]
    pub fn exp(self) -> Self {
        self.try_exp().unwrap()
    }
    /// See [exp]
    pub fn try_exp(self) -> Result<Self, D::Err> {
        try_unary_op(ExpKernelOp, self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tensor_ops::*, tests::*};

    #[test]
    fn test_exp() {
        let dev: TestDevice = Default::default();
        let x: Tensor<_, TestDtype, _> = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.leaky_trace().exp();
        assert_close_to_literal!(r, [0.13533528, 0.36787945, 1.0, f64::exp(1.0), 7.389056]);
        let g = r.mean().backward();
        assert_close_to_literal!(
            g.get(&x),
            [0.027067056, 0.07357589, 0.2, 0.54365635, 1.4778112]
        );
    }
}
