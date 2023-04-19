mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use super::ops::{try_unary_op, UnaryKernel};
use crate::{shapes::*, tensor::*};

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct ReLUKernelOp;

/// [Rectified Linear Unit (ReLU)](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)). `max(0, t)`
///
/// The derivative is the [Heaviside](https://en.wikipedia.org/wiki/Heaviside_step_function) function.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let t = dev.tensor([-1.0, 0.0, 1.0, 2.0]);
/// let r = t.relu();
/// assert_eq!(r.array(), [0.0, 0.0, 1.0, 2.0]);
/// ```
pub fn relu<S: Shape, E: Dtype, D: UnaryKernel<ReLUKernelOp, E>, T: Tape<E, D>>(
    t: Tensor<S, E, D, T>,
) -> Tensor<S, E, D, T> {
    t.relu()
}

impl<S: Shape, E: Dtype, D: UnaryKernel<ReLUKernelOp, E>, T: Tape<E, D>> Tensor<S, E, D, T> {
    /// See [relu]
    pub fn relu(self) -> Self {
        self.try_relu().unwrap()
    }
    /// See [relu]
    pub fn try_relu(self) -> Result<Self, D::Err> {
        try_unary_op(ReLUKernelOp, self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tensor_ops::*, tests::*};

    #[test]
    fn test_relu() {
        let dev: TestDevice = Default::default();
        let x: Tensor<_, TestDtype, _> = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.leaky_trace().relu();
        assert_close_to_literal!(r, [0.0, 0.0, 0.0, 1.0, 2.0]);
        // NOTE: call .exp() to make sure we cover cases where .relu() uses the result's gradient
        let g = r.exp().mean().backward();
        assert_close_to_literal!(g.get(&x), [0.0, 0.0, 0.0, 0.54365635, 1.4778112]);
    }
}
