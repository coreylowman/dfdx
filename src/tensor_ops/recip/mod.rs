mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use super::ops::{try_unary_op, UnaryKernel};
use crate::{shapes::*, tensor::*};

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct RecipKernelOp;

/// `1 / x`
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let t = dev.tensor([-1.0, 0.0, 1.0, 2.0]);
/// let r = t.recip();
/// ```
pub fn recip<S: Shape, E: Dtype, D: UnaryKernel<RecipKernelOp, E>, T: Tape<E, D>>(
    t: Tensor<S, E, D, T>,
) -> Tensor<S, E, D, T> {
    t.recip()
}

impl<S: Shape, E: Dtype, D: UnaryKernel<RecipKernelOp, E>, T: Tape<E, D>> Tensor<S, E, D, T> {
    /// See [recip]
    pub fn recip(self) -> Self {
        self.try_recip().unwrap()
    }
    /// See [recip]
    pub fn try_recip(self) -> Result<Self, D::Err> {
        try_unary_op(RecipKernelOp, self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tensor_ops::*, tests::*};

    #[test]
    fn test_recip() {
        let dev: TestDevice = Default::default();
        let x: Tensor<_, TestDtype, _> = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.leaky_trace().recip();
        assert_close_to_literal!(r, [-0.5, -1.0, f64::INFINITY, 1.0, 0.5]);
        let g = r.mean().backward();
        assert_close_to_literal!(g.get(&x), [-0.05, -0.2, f64::NEG_INFINITY, -0.2, -0.05]);
    }
}
