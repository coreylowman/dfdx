mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use super::ops::{try_unary_op, UnaryKernel};
use crate::{shapes::*, tensor::*};

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct TanhKernelOp;

/// [Hyperbolic Tangent (Tanh)](https://en.wikipedia.org/wiki/Hyperbolic_functions).
///
/// The derivative is `1.0 - square(tanh(t))`.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let t = dev.tensor([-1.0, 0.0, 1.0, 2.0]);
/// let r = t.tanh();
/// ```
pub fn tanh<S: Shape, E: Dtype, D: UnaryKernel<TanhKernelOp, E>, T: Tape<E, D>>(
    t: Tensor<S, E, D, T>,
) -> Tensor<S, E, D, T> {
    t.tanh()
}

impl<S: Shape, E: Dtype, D: UnaryKernel<TanhKernelOp, E>, T: Tape<E, D>> Tensor<S, E, D, T> {
    /// See [tanh]
    pub fn tanh(self) -> Self {
        self.try_tanh().unwrap()
    }
    /// See [tanh]
    pub fn try_tanh(self) -> Result<Self, D::Err> {
        try_unary_op(TanhKernelOp, self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tensor_ops::*, tests::*};

    #[test]
    fn test_tanh() {
        let dev: TestDevice = Default::default();
        let x: Tensor<_, TestDtype, _> = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.leaky_trace().tanh();
        assert_close_to_literal!(r, [-0.9640276, -0.7615942, 0., 0.7615942, 0.9640276]);
        let g = r.mean().backward();
        assert_close_to_literal!(
            g.get(&x),
            [0.014130163, 0.083994865, 0.2, 0.083994865, 0.014130163]
        );
    }
}
