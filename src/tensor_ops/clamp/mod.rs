mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use super::ops::{try_unary_op, UnaryKernel};
use crate::{shapes::*, tensor::*};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ClampKernelOp<E> {
    pub min: E,
    pub max: E,
}

/// Clamp all elements between the provided min and max values.
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let t = dev.tensor([-1.0, -0.5, 0.0, 0.5, 1.0]);
/// let r = t.clamp(-0.5, 0.5);
/// assert_eq!(r.array(), [-0.5, -0.5, 0.0, 0.5, 0.5]);
/// ```
pub fn clamp<S: Shape, E: Dtype, D: UnaryKernel<ClampKernelOp<E>, E>, T: Tape<E, D>>(
    t: Tensor<S, E, D, T>,
    min: impl Into<E>,
    max: impl Into<E>,
) -> Tensor<S, E, D, T> {
    t.clamp(min, max)
}

impl<S: Shape, E: Dtype, D: UnaryKernel<ClampKernelOp<E>, E>, T: Tape<E, D>> Tensor<S, E, D, T> {
    /// See [clamp]
    pub fn clamp(self, min: impl Into<E>, max: impl Into<E>) -> Self {
        self.try_clamp(min, max).unwrap()
    }
    /// See [clamp]
    pub fn try_clamp(self, min: impl Into<E>, max: impl Into<E>) -> Result<Self, D::Err> {
        try_unary_op(
            ClampKernelOp {
                min: min.into(),
                max: max.into(),
            },
            self,
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tensor_ops::*, tests::*};

    #[test]
    fn test_clamp() {
        let dev: TestDevice = Default::default();
        let t: Tensor<_, TestDtype, _> = dev.tensor([[-1.0, 0.0, 1.0], [-2.0, 2.0, 1.1]]);
        let r = t.leaky_trace().clamp(-1.0, 1.0);
        assert_close_to_literal!(r, [[-1.0, 0.0, 1.0], [-1.0, 1.0, 1.0]]);
        let g = r.exp().mean().backward();
        assert_close_to_literal!(g.get(&t), [[0.06131324, 0.16666667, 0.45304698], [0.0; 3]]);
    }
}
