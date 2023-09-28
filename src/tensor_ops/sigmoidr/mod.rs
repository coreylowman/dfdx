mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use super::ops::{try_unary_op, UnaryKernel};
use crate::{shapes::*, tensor::*};

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct SigmoidrKernelOp;

/// [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function). `1 / (1 + exp(-t))`.
/// Basically the same as sigmoid but will always return non-zero gradients.
/// The derivative is `sigmoid(t) * (1.0 - sigmoid(t))`.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let t = dev.tensor([-1.0, 0.0, 1.0, 2.0]);
/// let r = t.sigmoid();
/// ```
pub fn sigmoidr<S: Shape, E: Dtype, D: UnaryKernel<SigmoidrKernelOp, E>, T: Tape<E, D>>(
    t: Tensor<S, E, D, T>,
) -> Tensor<S, E, D, T> {
    t.sigmoidr()
}

impl<S: Shape, E: Dtype, D: UnaryKernel<SigmoidrKernelOp, E>, T: Tape<E, D>> Tensor<S, E, D, T> {
    /// See [sigmoidr]
    pub fn sigmoidr(self) -> Self {
        self.try_sigmoidr().unwrap()
    }
    /// See [sigmoidr]
    pub fn try_sigmoidr(self) -> Result<Self, D::Err> {
        try_unary_op(SigmoidrKernelOp, self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tensor_ops::*, tests::*};

    #[test]
    fn test_sigmoidr() {
        let dev: TestDevice = Default::default();
        let x = dev
            .tensor([-2.0, -1.0, 0.0, 1.0, -TestDtype::INFINITY])
            .to_dtype::<TestDtype>();
        let r = x.leaky_trace().sigmoidr();
        assert_close_to_literal!(r, [0.11920292, 0.26894143, 0.5, 0.7310586, 0.0]);
        let g = r.mean().backward();
        assert_close_to_literal!(
            g.get(&x),
            [0.020998716, 0.039322387, 0.05, 0.039322387, 0.00000002]
        );
    }
}
