mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use super::ops::{try_unary_op, UnaryKernel};
use crate::{shapes::*, tensor::*};

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct GeLUKernelOp;

/// [Gaussian Linear Unit (GeLU)](https://paperswithcode.com/method/gelu). `0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))`
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let t = dev.tensor([-1.0, 0.0, 1.0, 2.0]);
/// let r = t.gelu();
/// ```
pub fn gelu<S: Shape, E: Dtype, D: UnaryKernel<GeLUKernelOp, E>, T: Tape<E, D>>(
    t: Tensor<S, E, D, T>,
) -> Tensor<S, E, D, T> {
    t.gelu()
}

impl<S: Shape, E: Dtype, D: UnaryKernel<GeLUKernelOp, E>, T: Tape<E, D>> Tensor<S, E, D, T> {
    /// See [gelu]
    pub fn gelu(self) -> Self {
        self.try_gelu().unwrap()
    }
    /// See [gelu]
    pub fn try_gelu(self) -> Result<Self, D::Err> {
        try_unary_op(GeLUKernelOp, self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tensor_ops::*, tests::*};

    #[test]
    fn test_gelu() {
        let dev: TestDevice = Default::default();
        let x: Tensor<_, TestDtype, _> = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.leaky_trace().gelu();
        assert_close_to_literal!(r, [-0.04540229, -0.158808, 0.0, 0.841192, 1.9545977]);
        // NOTE: call .exp() to make sure we cover cases where .gelu() uses the result's gradient
        let g = r.exp().mean().backward();
        assert_close_to_literal!(
            g.get(&x),
            [-0.016455507, -0.014156329, 0.1, 0.5023068, 1.5338063]
        );
    }
}
