mod cpu_kernel;

use crate::{
    arrays::{Dtype, Shape},
    devices::{DeviceStorage, HasErr},
    gradients::{Merge, Tape},
    tensor::Tensor,
};

use super::ops::{try_binary_op, BinaryKernel};

pub trait TryBceWithLogits<Rhs = Self>: HasErr {
    fn bce_with_logits(self, rhs: Rhs) -> Self {
        self.try_bce_with_logits(rhs).unwrap()
    }
    fn try_bce_with_logits(self, rhs: Rhs) -> Result<Self, Self::Err>;
}

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct BCEKernelOp;

impl<S: Shape, E: Dtype, D: DeviceStorage, LhsTape: Tape<D>, RhsTape: Tape<D>>
    TryBceWithLogits<Tensor<S, E, D, RhsTape>> for Tensor<S, E, D, LhsTape>
where
    D: BinaryKernel<BCEKernelOp, S, S, S, E>,
    LhsTape: Merge<RhsTape>,
{
    fn try_bce_with_logits(self, rhs: Tensor<S, E, D, RhsTape>) -> Result<Self, Self::Err> {
        try_binary_op(Default::default(), self, rhs)
    }
}
