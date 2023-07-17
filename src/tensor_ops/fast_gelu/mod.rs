mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use super::ops::{try_unary_op, UnaryKernel};
use crate::{shapes::*, tensor::*};

#[allow(unused)]
#[deprecated(since = "0.12.0", note = "use `FastGeLUKernelOp` instead")]
pub type GeLUKernelOp = FastGeLUKernelOp;

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct FastGeLUKernelOp;

/// [Fast Gaussian Linear Unit (GeLU)](https://paperswithcode.com/method/gelu). A fast version of the gaussiane linear unit
/// calculated by
/// ```text
/// 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
/// ````
/// See also [accurate_gelu](super::accurate_gelu::accurate_gelu) for the more accurate version.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let t = dev.tensor([-1.0, 0.0, 1.0, 2.0]);
/// let r = t.gelu();
/// ```
pub fn fast_gelu<S: Shape, E: Dtype, D: UnaryKernel<FastGeLUKernelOp, E>, T: Tape<E, D>>(
    t: Tensor<S, E, D, T>,
) -> Tensor<S, E, D, T> {
    t.fast_gelu()
}

/// Use [fast_gelu] instead
#[deprecated(since = "0.12.0", note = "Use `fast_gelu` instead")]
pub fn gelu<S: Shape, E: Dtype, D: UnaryKernel<FastGeLUKernelOp, E>, T: Tape<E, D>>(
    t: Tensor<S, E, D, T>,
) -> Tensor<S, E, D, T> {
    fast_gelu(t)
}

impl<S: Shape, E: Dtype, D: UnaryKernel<FastGeLUKernelOp, E>, T: Tape<E, D>> Tensor<S, E, D, T> {
    /// See [fast_gelu]
    pub fn fast_gelu(self) -> Self {
        self.try_fast_gelu().unwrap()
    }
    /// See [fast_gelu]
    pub fn try_fast_gelu(self) -> Result<Self, D::Err> {
        try_unary_op(FastGeLUKernelOp, self)
    }

    /// Use [fast_gelu] instead
    #[deprecated(since = "0.12.0", note = "Use [fast_gelu](#method.fast_gelu) instead")]
    pub fn gelu(self) -> Self {
        self.fast_gelu()
    }

    /// Use [try_fast_gelu] instead
    #[deprecated(since = "0.12.0", note = "Use `try_fast_gelu` instead")]
    pub fn try_gelu(self) -> Result<Self, D::Err> {
        self.try_fast_gelu()
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tensor_ops::*, tests::*};

    #[test]
    fn test_fast_gelu() {
        let dev: TestDevice = Default::default();
        let x = dev
            .tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
            .to_dtype::<TestDtype>();
        let r = x.leaky_trace().fast_gelu();
        assert_close_to_literal!(r, [-0.04540229, -0.158808, 0.0, 0.841192, 1.9545977]);
        // NOTE: call .exp() to make sure we cover cases where .gelu() uses the result's gradient
        let g = r.exp().mean().backward();
        assert_close_to_literal!(
            g.get(&x),
            [-0.016455507, -0.014156329, 0.1, 0.5023068, 1.5338063]
        );
    }
}
