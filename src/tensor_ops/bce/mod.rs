mod cpu_kernel;

use super::{ops::try_binary_op, Device};
use crate::{gradients::*, shapes::*, tensor::Tensor};

#[derive(Debug, Default, Clone, Copy)]
pub struct BCEKernelOp;

pub fn bce_with_logits<S: Shape, E: Dtype, D: Device<E>, LhsTape, RhsTape>(
    lhs: Tensor<S, E, D, LhsTape>,
    rhs: Tensor<S, E, D, RhsTape>,
) -> Tensor<S, E, D, LhsTape>
where
    LhsTape: Tape<D> + Merge<RhsTape>,
    RhsTape: Tape<D>,
{
    lhs.bce_with_logits(rhs)
}

impl<S: Shape, E: Dtype, D: Device<E>, LhsTape: Tape<D>> Tensor<S, E, D, LhsTape> {
    /// See [bce_with_logits]
    pub fn bce_with_logits<RhsTape: Tape<D>>(self, rhs: Tensor<S, E, D, RhsTape>) -> Self
    where
        LhsTape: Merge<RhsTape>,
    {
        self.try_bce_with_logits(rhs).unwrap()
    }
    /// See [bce_with_logits]
    pub fn try_bce_with_logits<RhsTape>(self, rhs: Tensor<S, E, D, RhsTape>) -> Result<Self, D::Err>
    where
        RhsTape: Tape<D>,
        LhsTape: Merge<RhsTape>,
    {
        try_binary_op(BCEKernelOp, self, rhs)
    }
}
