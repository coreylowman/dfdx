mod cpu_kernel;

use crate::{
    arrays::{Dtype, Shape},
    devices::{DeviceStorage, HasErr},
    gradients::{Merge, Tape},
    tensor::Tensor,
};

use super::ops::{try_binary_op, BinaryKernel};

/// Element wise maximum.
///
/// **Pytorch equivalent**: `torch.maximum(a, b)`
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// let a = tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
/// let b = tensor([[1.0, 0.5, 1.0], [-2.0, 2.0, -3.5]]);
/// let r = a.maximum(b);
/// assert_eq!(r.data(), &[[1.0, 2.0, 3.0], [-1.0, 2.0, -3.0]]);
pub trait TryMaximum<Rhs = Self>: HasErr {
    fn maximum(self, rhs: Rhs) -> Self {
        self.try_maximum(rhs).unwrap()
    }
    fn try_maximum(self, rhs: Rhs) -> Result<Self, Self::Err>;
}

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct MaximumKernelOp;

impl<S: Shape, E: Dtype, D: DeviceStorage, LhsTape: Tape<D>, RhsTape: Tape<D>>
    TryMaximum<Tensor<S, E, D, RhsTape>> for Tensor<S, E, D, LhsTape>
where
    D: BinaryKernel<MaximumKernelOp, S, S, S, E>,
    LhsTape: Merge<RhsTape>,
{
    fn try_maximum(self, rhs: Tensor<S, E, D, RhsTape>) -> Result<Self, Self::Err> {
        try_binary_op(Default::default(), self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        devices::AsArray, tensor::TensorFromArray, tensor_ops::*, tests::build_test_device,
    };

    #[test]
    fn test_maximum() {
        let dev = build_test_device!();
        let a = dev.tensor([[-1.0, 0.0, 1.0], [3.0, 4.0, -5.0]]);
        let b = dev.tensor([[0.0, 0.0, -1.0], [3.0, -4.0, 5.0]]);

        let result = a.trace().maximum(b.clone());
        assert_eq!(result.as_array(), [[0.0, 0.0, 1.0], [3.0, 4.0, 5.0]]);

        let g = result.sum().backward();
        assert_eq!(g.get(&a).as_array(), [[0.0, 0.5, 1.0], [0.5, 1.0, 0.0]]);
        assert_eq!(g.get(&b).as_array(), [[1.0, 0.5, 0.0], [0.5, 0.0, 1.0]]);
    }
}
