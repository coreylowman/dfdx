mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

#[cfg(feature = "webgpu")]
mod webgpu_kernel;

use super::ops::{try_unary_op, UnaryKernel};
use crate::{shapes::*, tensor::*};

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct SiLUKernelOp;

/// [Sigmoid-Weighted Linear Unit (SiLU)](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)). `x * x.sigmoid()`
///
/// The derivative is `x * sigmoid'(x) + sigmoid(x)`.
///
/// Examples:
/// ```rust
/// # use dfdx_core::prelude::*;
/// # let dev: Cpu = Default::default();
/// let t = dev.tensor([-1.0, 0.0, 1.0, 2.0]);
/// let r = t.silu();
/// ```
pub fn silu<S: Shape, E: Dtype, D: UnaryKernel<SiLUKernelOp, E>, T: Tape<E, D>>(
    t: Tensor<S, E, D, T>,
) -> Tensor<S, E, D, T> {
    t.silu()
}

impl<S: Shape, E: Dtype, D: UnaryKernel<SiLUKernelOp, E>, T: Tape<E, D>> Tensor<S, E, D, T> {
    /// See [silu]
    pub fn silu(self) -> Self {
        self.try_silu().unwrap()
    }
    /// See [silu]
    pub fn try_silu(self) -> Result<Self, crate::tensor::Error> {
        try_unary_op(SiLUKernelOp, self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tensor_ops::*, tests::*};

    #[test]
    fn test_silu() {
        let dev: TestDevice = Default::default();
        let x = dev
            .tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
            .to_dtype::<TestDtype>();
        let r = x.leaky_trace().silu();
        assert_close_to_literal!(r, [-0.23840584, -0.26894143, 0.0, 0.7310586, 1.761594]);
        let g = r.mean().backward();
        assert_close_to_literal!(
            g.get(&x),
            [-0.018156849, 0.014465898, 0.1, 0.1855341, 0.21815684]
        );
    }
}
