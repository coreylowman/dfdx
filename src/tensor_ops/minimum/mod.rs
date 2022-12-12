mod cpu_kernel;

use super::{ops::try_binary_op, Device};
use crate::{gradients::*, shapes::*, tensor::Tensor};

#[derive(Debug, Default, Clone, Copy)]
pub struct MinimumKernelOp;

/// Element wise minimum.
///
/// **Pytorch equivalent**: `torch.minimum(a, b)`
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a = dev.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
/// let b = dev.tensor([[1.0, 0.5, 1.0], [-2.0, 2.0, -3.5]]);
/// let r = a.minimum(b);
/// assert_eq!(r.array(), [[1.0, 0.5, 1.0], [-2.0, -2.0, -3.5]]);
pub fn minimum<S: Shape, E: Dtype, D: Device<E>, LTape: Tape<D> + Merge<RTape>, RTape: Tape<D>>(
    lhs: Tensor<S, E, D, LTape>,
    rhs: Tensor<S, E, D, RTape>,
) -> Tensor<S, E, D, LTape> {
    lhs.minimum(rhs)
}

impl<S: Shape, E: Dtype, D: Device<E>, LTape: Tape<D>> Tensor<S, E, D, LTape> {
    /// See [minimum]
    pub fn minimum<RTape: Tape<D>>(self, rhs: Tensor<S, E, D, RTape>) -> Self
    where
        LTape: Merge<RTape>,
    {
        self.try_minimum(rhs).unwrap()
    }

    /// See [minimum]
    pub fn try_minimum<RTape: Tape<D>>(self, rhs: Tensor<S, E, D, RTape>) -> Result<Self, D::Err>
    where
        LTape: Merge<RTape>,
    {
        try_binary_op(MinimumKernelOp, self, rhs)
    }
}
#[cfg(test)]
mod tests {
    use crate::{tensor::*, tensor_ops::*, tests::build_test_device};

    #[test]
    fn test_minimum() {
        let dev = build_test_device!();
        let a = dev.tensor([[-1.0, 0.0, 1.0], [3.0, 4.0, -5.0]]);
        let b = dev.tensor([[0.0, 0.0, -1.0], [3.0, -4.0, 5.0]]);

        let result = a.trace().minimum(b.clone());
        assert_eq!(result.array(), [[-1., 0., -1.], [3., -4., -5.]]);

        let g = result.sum().backward();
        assert_eq!(g.get(&a).array(), [[1.0, 0.5, 0.0], [0.5, 0.0, 1.0]]);
        assert_eq!(g.get(&b).array(), [[0.0, 0.5, 1.0], [0.5, 1.0, 0.0]]);
    }
}
