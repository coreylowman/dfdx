mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use super::{ops::try_binary_op, Device};
use crate::{gradients::*, shapes::*, tensor::Tensor};

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
pub fn maximum<S: Shape, E: Dtype, D: Device<E>, LTape: Tape<D> + Merge<RTape>, RTape: Tape<D>>(
    lhs: Tensor<S, E, D, LTape>,
    rhs: Tensor<S, E, D, RTape>,
) -> Tensor<S, E, D, LTape> {
    lhs.maximum(rhs)
}

impl<S: Shape, E: Dtype, D: Device<E>, LTape: Tape<D>> Tensor<S, E, D, LTape> {
    /// See [maximum]
    pub fn maximum<RTape: Tape<D>>(self, rhs: Tensor<S, E, D, RTape>) -> Self
    where
        LTape: Merge<RTape>,
    {
        self.try_maximum(rhs).unwrap()
    }

    /// See [maximum]
    pub fn try_maximum<R: Tape<D>>(self, rhs: Tensor<S, E, D, R>) -> Result<Self, D::Err>
    where
        LTape: Merge<R>,
    {
        try_binary_op(MaximumKernelOp, self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tensor_ops::*, tests::TestDevice};

    #[test]
    fn test_maximum() {
        let dev: TestDevice = Default::default();
        let a = dev.tensor([[-1.0, 0.0, 1.0], [3.0, 4.0, -5.0]]);
        let b = dev.tensor([[0.0, 0.0, -1.0], [3.0, -4.0, 5.0]]);

        let result = a.trace().maximum(b.clone());
        assert_eq!(result.array(), [[0.0, 0.0, 1.0], [3.0, 4.0, 5.0]]);

        let g = result.sum().backward();
        assert_eq!(g.get(&a).array(), [[0.0, 0.5, 1.0], [0.5, 1.0, 0.0]]);
        assert_eq!(g.get(&b).array(), [[1.0, 0.5, 0.0], [0.5, 0.0, 1.0]]);
    }
}
