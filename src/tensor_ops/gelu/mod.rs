mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use super::ops::{try_unary_op, UnaryKernel};
use crate::{gradients::Tape, shapes::*, tensor::Tensor};

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
/// assert_eq!(r.array(), [-0.158808, 0.0, 0.841192, 1.9545977]);
/// ```
pub fn gelu<S: Shape, E: Dtype, D: UnaryKernel<GeLUKernelOp, E>, T: Tape<D>>(
    t: Tensor<S, E, D, T>,
) -> Tensor<S, E, D, T> {
    t.gelu()
}

impl<S: Shape, E: Dtype, D: UnaryKernel<GeLUKernelOp, E>, T: Tape<D>> Tensor<S, E, D, T> {
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
    use crate::{tensor::*, tensor_ops::*, tests::TestDevice};

    fn simplify(data: &[f32]) -> no_std_compat::vec::Vec<f32> {
        let precision = 3;
        let m = 10.0 * 10.0f32.powf(precision as f32);
        data.iter().map(|x| (x * m).round() / m).collect()
    }

    #[test]
    fn test_gelu() {
        let dev: TestDevice = Default::default();
        let x = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.trace().gelu();
        assert_eq!(
            simplify(&r.array()),
            [-0.0454, -0.1588, 0.0, 0.8412, 1.9546]
        );

        // NOTE: call .exp() to make sure we cover cases where .gelu() uses the result's gradient
        let g = r.exp().mean().backward();
        assert_eq!(
            simplify(&g.get(&x).array()),
            [-0.0165, -0.0142, 0.1, 0.5023, 1.5338]
        );
    }
}
