use crate::{
    arrays::{Dtype, Shape},
    devices::{
        binary_ops,
        device::{BinaryKernel, HasErr},
        Device,
    },
    gradients::{Merge, Tape},
    tensor::Tensor,
};

use super::utils::try_binary_op;

pub trait TryBceWithLogits<Rhs = Self>: HasErr {
    fn bce_with_logits(self, rhs: Rhs) -> Self {
        self.try_bce_with_logits(rhs).unwrap()
    }
    fn try_bce_with_logits(self, rhs: Rhs) -> Result<Self, Self::Err>;
}

impl<S: Shape, E: Dtype, D: Device, LhsTape: Tape<D>, RhsTape: Tape<D>>
    TryBceWithLogits<Tensor<S, E, D, RhsTape>> for Tensor<S, E, D, LhsTape>
where
    D: BinaryKernel<binary_ops::BCEWithLogits, S, S, S, E>,
    LhsTape: Merge<RhsTape>,
{
    fn try_bce_with_logits(self, rhs: Tensor<S, E, D, RhsTape>) -> Result<Self, Self::Err> {
        try_binary_op(Default::default(), self, rhs)
    }
}
