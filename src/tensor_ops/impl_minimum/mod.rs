mod cpu_kernel;

use crate::{
    arrays::{Dtype, Shape},
    devices::{BinaryKernel, Device, HasErr},
    gradients::{Merge, Tape},
    tensor::Tensor,
};

use super::utils::try_binary_op;

/// Element wise minimum.
///
/// **Pytorch equivalent**: `torch.minimum(a, b)`
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// let a = tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
/// let b = tensor([[1.0, 0.5, 1.0], [-2.0, 2.0, -3.5]]);
/// let r = a.minimum(b);
/// assert_eq!(r.data(), &[[1.0, 0.5, 1.0], [-2.0, -2.0, -3.5]]);
pub trait Minimum<Rhs = Self>: HasErr {
    fn minimum(self, rhs: Rhs) -> Self {
        self.try_minimum(rhs).unwrap()
    }
    fn try_minimum(self, rhs: Rhs) -> Result<Self, Self::Err>;
}

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct MinimumKernelOp;

impl<S: Shape, E: Dtype, D: Device, LhsTape: Tape<D>, RhsTape: Tape<D>>
    Minimum<Tensor<S, E, D, RhsTape>> for Tensor<S, E, D, LhsTape>
where
    D: BinaryKernel<MinimumKernelOp, S, S, S, E>,
    LhsTape: Merge<RhsTape>,
{
    fn try_minimum(self, rhs: Tensor<S, E, D, RhsTape>) -> Result<Self, Self::Err> {
        try_binary_op(Default::default(), self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        devices::AsArray,
        tensor::TensorSugar,
        tensor_ops::{impl_backward::TryBackward, impl_sum::SumTo},
        tests::build_test_device,
    };

    #[test]
    fn test_minimum() {
        let dev = build_test_device!();
        let a = dev.tensor([[-1.0, 0.0, 1.0], [3.0, 4.0, -5.0]]);
        let b = dev.tensor([[0.0, 0.0, -1.0], [3.0, -4.0, 5.0]]);

        let result = a.trace().minimum(b.clone());
        assert_eq!(result.as_array(), [[-1., 0., -1.], [3., -4., -5.]]);

        let g = result.sum().backward();
        assert_eq!(g.get(&a).as_array(), [[1.0, 0.5, 0.0], [0.5, 0.0, 1.0]]);
        assert_eq!(g.get(&b).as_array(), [[0.0, 0.5, 1.0], [0.5, 1.0, 0.0]]);
    }
}
