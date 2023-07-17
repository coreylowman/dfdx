mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use super::ops::{try_unary_op, UnaryKernel};
use crate::{shapes::*, tensor::*};

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct ReLU6KernelOp;

/// Modification of [Rectified Linear Unit (ReLU)](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)). `min(max(0, t), 6)`
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let t = dev.tensor([-1.0, 0.0, 1.0, 2.0, 7.0]);
/// let r = t.relu6();
/// assert_eq!(r.array(), [0.0, 0.0, 1.0, 2.0, 6.0]);
/// ```
pub fn relu6<S: Shape, E: Dtype, D: UnaryKernel<ReLU6KernelOp, E>, T: Tape<E, D>>(
    t: Tensor<S, E, D, T>,
) -> Tensor<S, E, D, T> {
    t.relu6()
}

impl<S: Shape, E: Dtype, D: UnaryKernel<ReLU6KernelOp, E>, T: Tape<E, D>> Tensor<S, E, D, T> {
    /// See [relu6]
    pub fn relu6(self) -> Self {
        self.try_relu6().unwrap()
    }
    /// See [relu6]
    pub fn try_relu6(self) -> Result<Self, D::Err> {
        try_unary_op(ReLU6KernelOp, self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tensor_ops::*, tests::*};

    #[test]
    fn test_relu6() {
        let dev: TestDevice = Default::default();
        let x = dev
            .tensor([-2.0, -1.0, 0.0, 1.0, 2.0, 7.0])
            .to_dtype::<TestDtype>();
        let r = x.leaky_trace().relu6();
        assert_close_to_literal!(r, [0.0, 0.0, 0.0, 1.0, 2.0, 6.0]);
        // NOTE: call .exp() to make sure we cover cases where .relu6() uses the result's gradient
        let g = r.exp().mean().backward();
        assert_close_to_literal!(g.get(&x), [0.0, 0.0, 0.0, 0.45304698, 1.2315094, 0.0]);
    }
}
