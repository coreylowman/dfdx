mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

// #[cfg(feature = "webgpu")]
// mod webgpu_kernel;

use super::ops::{try_unary_op, UnaryKernel};
use crate::{shapes::*, tensor::*};

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct AccurateGeLUKernelOp;

/// [Accurate Gaussian Linear Unit (GeLU)](https://paperswithcode.com/method/gelu). This is defined as `x * Phi(x)` where `Phi(x)` is the cumulative
/// distribution function of a standard normal distribution. This can be calculated via the Error
/// Function `erf(x)` using
/// ```text
/// 0.5 * x * (1.0 + erf(x / 2.0.sqrt()))
/// ```
/// As an accurate error function is [computationally expensive](https://en.wikipedia.org/wiki/Error_function#Numerical_approximations) it is
/// possible to approximate the Gaussian Linear Unit with a hyperbolic tangent function `tanh`
///
/// ```text
/// GeLU(x) ~ 0.5 ∗ x ∗ (1.0 + tanh((sqrt(2.0/π) ∗ (x + 0.044715 ∗ x^3)))
/// ```
///
/// See [fast_gelu](super::fast_gelu::fast_gelu) to use this approximation
///
///
/// Examples:
/// ```rust
/// # use dfdx_core::prelude::*;
/// # let dev: Cpu = Default::default();
/// let t = dev.tensor([-1.0, 0.0, 1.0, 2.0]);
/// let r = t.accurate_gelu();
/// ```
pub fn accurate_gelu<S: Shape, E: Dtype, D: UnaryKernel<AccurateGeLUKernelOp, E>, T: Tape<E, D>>(
    t: Tensor<S, E, D, T>,
) -> Tensor<S, E, D, T> {
    t.accurate_gelu()
}

impl<S: Shape, E: Dtype, D: UnaryKernel<AccurateGeLUKernelOp, E>, T: Tape<E, D>>
    Tensor<S, E, D, T>
{
    /// See [accurate_gelu]
    pub fn accurate_gelu(self) -> Self {
        self.try_accurate_gelu().unwrap()
    }
    /// See [accurate_gelu]
    pub fn try_accurate_gelu(self) -> Result<Self, crate::tensor::Error> {
        try_unary_op(AccurateGeLUKernelOp, self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tensor_ops::*, tests::*};

    #[test]
    fn test_accurate_gelu() {
        let dev: TestDevice = Default::default();
        let x = dev
            .tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
            .to_dtype::<TestDtype>();
        let r = x.leaky_trace().accurate_gelu();

        assert_close_to_literal!(r, [-0.04550027, -0.15865525, 0.0, 0.84134471, 1.9544997,]);
        // NOTE: call .exp() to make sure we cover cases where .gelu() uses the result's gradient
        let g = r.exp().mean().backward();
        assert_close_to_literal!(
            g.get(&x),
            [-0.024835737, -0.03132311, 0.1, 0.5490418, 1.59559]
        );
    }
}
