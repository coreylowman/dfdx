mod cpu_kernel;

use super::ops::{try_unary_op, UnaryKernel};
use crate::{gradients::Tape, shapes::*, tensor::Tensor};

#[derive(Debug, Default, Copy, Clone)]
pub struct SigmoidKernelOp;

/// [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function). `1 / (1 + exp(-t))`.
///
/// The derivative is `sigmoid(t) * (1.0 - sigmoid(t))`.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let t = dev.tensor([-1.0, 0.0, 1.0, 2.0]);
/// let r = t.sigmoid();
/// ```
pub fn sigmoid<S: Shape, E: Dtype, D: UnaryKernel<SigmoidKernelOp, E>, T: Tape<D>>(
    t: Tensor<S, E, D, T>,
) -> Tensor<S, E, D, T> {
    t.sigmoid()
}

impl<S: Shape, E: Dtype, D: UnaryKernel<SigmoidKernelOp, E>, T: Tape<D>> Tensor<S, E, D, T> {
    /// See [sigmoid]
    pub fn sigmoid(self) -> Self {
        self.try_sigmoid().unwrap()
    }
    /// See [sigmoid]
    pub fn try_sigmoid(self) -> Result<Self, D::Err> {
        try_unary_op(SigmoidKernelOp, self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tensor_ops::*, tests::TestDevice};

    #[test]
    fn test_sigmoid() {
        let dev: TestDevice = Default::default();
        let x = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.trace().sigmoid();
        assert_eq!(
            r.array(),
            [0.11920292, 0.26894143, 0.5, 0.7310586, 0.880797]
        );
        let g = r.mean().backward();
        assert_eq!(
            g.get(&x).array(),
            [0.020998716, 0.039322387, 0.05, 0.039322387, 0.020998726]
        );
    }
}
