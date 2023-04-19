mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use super::ops::{try_unary_op, UnaryKernel};
use crate::{shapes::*, tensor::*};

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct CosKernelOp;

/// [Cosine function](https://en.wikipedia.org/wiki/Sine_and_cosine).
///
/// It's derivative is `-sin(t)`
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let t = dev.tensor([-1.0, 0.0, 1.0, 2.0]);
/// let r = t.cos();
/// ```
pub fn cos<S: Shape, E: Dtype, D: UnaryKernel<CosKernelOp, E>, T: Tape<E, D>>(
    t: Tensor<S, E, D, T>,
) -> Tensor<S, E, D, T> {
    t.cos()
}

impl<S: Shape, E: Dtype, D: UnaryKernel<CosKernelOp, E>, T: Tape<E, D>> Tensor<S, E, D, T> {
    /// See [cos]
    pub fn cos(self) -> Self {
        self.try_cos().unwrap()
    }
    /// See [cos]
    pub fn try_cos(self) -> Result<Self, D::Err> {
        try_unary_op(CosKernelOp, self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tensor_ops::*, tests::*};

    #[test]
    fn test_cos() {
        let dev: TestDevice = Default::default();
        let x: Tensor<_, TestDtype, _> = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.leaky_trace().cos();
        assert_close_to_literal!(r, [-0.41614684, 0.5403023, 1.0, 0.5403023, -0.41614684]);
        let g = r.mean().backward();
        assert_close_to_literal!(
            g.get(&x),
            [0.18185948, 0.16829419, -0.0, -0.16829419, -0.18185948]
        );
    }
}
