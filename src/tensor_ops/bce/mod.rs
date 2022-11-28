mod cpu_kernel;

use super::{ops::try_binary_op, Device};
use crate::{arrays::*, gradients::*, tensor::Tensor};

#[derive(Debug, Default, Clone, Copy)]
pub struct BCEKernelOp;

pub fn bce_with_logits<
    S: Shape,
    E: Dtype,
    D: Device<E>,
    T: Tape<D> + Merge<RhsTape>,
    RhsTape: Tape<D>,
>(
    lhs: Tensor<S, E, D, T>,
    rhs: Tensor<S, E, D, RhsTape>,
) -> Tensor<S, E, D, T> {
    lhs.bce_with_logits(rhs)
}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<D>> Tensor<S, E, D, T> {
    /// Calls [bce_with_logits]
    pub fn bce_with_logits<RhsTape: Tape<D>>(self, rhs: Tensor<S, E, D, RhsTape>) -> Self
    where
        T: Merge<RhsTape>,
    {
        self.try_bce_with_logits(rhs).unwrap()
    }

    /// Calls [try_bce_with_logits]
    pub fn try_bce_with_logits<RhsTape: Tape<D>>(
        self,
        rhs: Tensor<S, E, D, RhsTape>,
    ) -> Result<Self, D::Err>
    where
        T: Merge<RhsTape>,
    {
        try_binary_op(BCEKernelOp, self, rhs)
    }
}
