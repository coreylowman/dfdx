mod cpu_kernel;

use crate::{
    arrays::{Dtype, Shape},
    gradients::{Merge, Tape},
    tensor::Tensor,
};

use super::{ops::try_binary_op, Device};

#[derive(Debug, Default, Clone, Copy)]
pub struct MaximumKernelOp;

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
pub fn maximum<S: Shape, E: Dtype, D: Device<E>, T: Tape<D> + Merge<RhsTape>, RhsTape: Tape<D>>(
    lhs: Tensor<S, E, D, T>,
    rhs: Tensor<S, E, D, RhsTape>,
) -> Tensor<S, E, D, T> {
    lhs.maximum(rhs)
}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<D>> Tensor<S, E, D, T> {
    /// Calls [maximum]
    pub fn maximum<R: Tape<D>>(self, rhs: Tensor<S, E, D, R>) -> Self
    where
        T: Merge<R>,
    {
        self.try_maximum(rhs).unwrap()
    }

    /// Calls [try_maximum]
    pub fn try_maximum<R: Tape<D>>(self, rhs: Tensor<S, E, D, R>) -> Result<Self, D::Err>
    where
        T: Merge<R>,
    {
        try_binary_op(MaximumKernelOp, self, rhs)
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
