mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use super::ops::{try_unary_op, UnaryKernel};
use crate::{shapes::*, tensor::*};

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct GeLUCorrectKernelOp;

/// [Gaussian Linear Unit (GeLU)](https://paperswithcode.com/method/gelu). `0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))`
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let t = dev.tensor([-1.0, 0.0, 1.0, 2.0]);
/// let r = t.gelu_correct();
/// ```
pub fn gelu_correct<S: Shape, E: Dtype, D: UnaryKernel<GeLUCorrectKernelOp, E>, T: Tape<E, D>>(
    t: Tensor<S, E, D, T>,
) -> Tensor<S, E, D, T> {
    t.gelu_correct()
}

impl<S: Shape, E: Dtype, D: UnaryKernel<GeLUCorrectKernelOp, E>, T: Tape<E, D>> Tensor<S, E, D, T> {
    /// See [gelu]
    pub fn gelu_correct(self) -> Self {
        self.try_gelu_correct().unwrap()
    }
    /// See [gelu]
    pub fn try_gelu_correct(self) -> Result<Self, D::Err> {
        try_unary_op(GeLUCorrectKernelOp, self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tensor_ops::*, tests::*};

    #[test]
    fn test_gelu_correct() {
        let dev: TestDevice = Default::default();
        let x = dev
            .tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
            .to_dtype::<TestDtype>();
        let r = x.leaky_trace().gelu_correct();

        assert_close_to_literal!(r, [-0.04550027, -0.15865525, 0.0, 0.84134471, 1.9544997,]);
        // NOTE: call .exp() to make sure we cover cases where .gelu() uses the result's gradient
        let g = r.exp().mean().backward();
        assert_close_to_literal!(
            g.get(&x),
            [-0.024835737, -0.03132311, 0.1, 0.5490418, 1.59559]
        );
    }
}
