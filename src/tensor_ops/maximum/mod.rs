mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use super::{ops::try_binary_op, Device};
use crate::{shapes::*, tensor::*};

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct MaximumKernelOp;

/// Element wise maximum.
///
/// **Pytorch equivalent**: `torch.maximum(a, b)`
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a = dev.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
/// let b = dev.tensor([[1.0, 0.5, 1.0], [-2.0, 2.0, -3.5]]);
/// let r = a.maximum(b);
/// assert_eq!(r.array(), [[1.0, 2.0, 3.0], [-1.0, 2.0, -3.0]]);
pub fn maximum<S: Shape, E: Dtype, D: Device<E>, LTape: Tape<E, D> + Merge<R>, R: Default>(
    lhs: Tensor<S, E, D, LTape>,
    rhs: Tensor<S, E, D, R>,
) -> Tensor<S, E, D, LTape> {
    lhs.maximum(rhs)
}

impl<S: Shape, E: Dtype, D: Device<E>, LTape: Tape<E, D>> Tensor<S, E, D, LTape> {
    /// See [maximum]
    pub fn maximum<R: Default>(self, rhs: Tensor<S, E, D, R>) -> Self
    where
        LTape: Merge<R>,
    {
        self.try_maximum(rhs).unwrap()
    }

    /// See [maximum]
    pub fn try_maximum<R: Default>(self, rhs: Tensor<S, E, D, R>) -> Result<Self, D::Err>
    where
        LTape: Merge<R>,
    {
        try_binary_op(MaximumKernelOp, self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tensor_ops::*, tests::*};

    #[test]
    fn test_maximum() {
        let dev: TestDevice = Default::default();
        let a: Tensor<_, TestDtype, _> = dev.tensor([[-1.0, 0.0, 1.0], [3.0, 4.0, -5.0]]);
        let b: Tensor<_, TestDtype, _> = dev.tensor([[0.0, 0.0, -1.0], [3.0, -4.0, 5.0]]);

        let result = a.leaky_trace().maximum(b.clone());
        assert_close_to_literal!(result, [[0.0, 0.0, 1.0], [3.0, 4.0, 5.0]]);

        let g = result.sum().backward();
        assert_close_to_literal!(g.get(&a), [[0.0, 0.5, 1.0], [0.5, 1.0, 0.0]]);
        assert_close_to_literal!(g.get(&b), [[1.0, 0.5, 0.0], [0.5, 0.0, 1.0]]);
    }
}
