mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use super::ops::{try_unary_op, UnaryKernel};
use crate::{shapes::*, tensor::*};

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct LnKernelOp;

/// [Natural Logarithm (ln)](https://en.wikipedia.org/wiki/Natural_logarithm). `log_e(t)`.
///
/// It's derivative is `1 / t`.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let t = dev.tensor([-1.0, 0.0, 1.0, 2.0]);
/// let r = t.ln();
/// ```
pub fn ln<S: Shape, E: Dtype, D: UnaryKernel<LnKernelOp, E>, T: Tape<E, D>>(
    t: Tensor<S, E, D, T>,
) -> Tensor<S, E, D, T> {
    t.ln()
}

impl<S: Shape, E: Dtype, D: UnaryKernel<LnKernelOp, E>, T: Tape<E, D>> Tensor<S, E, D, T> {
    /// See [ln]
    pub fn ln(self) -> Self {
        self.try_ln().unwrap()
    }
    /// See [ln]
    pub fn try_ln(self) -> Result<Self, D::Err> {
        try_unary_op(LnKernelOp, self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tensor_ops::*, tests::*};

    #[test]
    fn test_ln() {
        let dev: TestDevice = Default::default();
        let x: Tensor<_, TestDtype, _> = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.leaky_trace().ln();
        let r_array = r.array();
        assert!(r_array[0].is_nan());
        assert!(r_array[1].is_nan());
        assert!(r_array[2..] == [TestDtype::NEG_INFINITY, 0.0, TestDtype::ln(2.0)]);
        let g = r.mean().backward();
        assert_close_to_literal!(g.get(&x), [-0.1, -0.2, f64::INFINITY, 0.2, 0.1]);
    }
}
