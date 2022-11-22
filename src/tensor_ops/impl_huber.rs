use crate::{
    arrays::{Dtype, HasDtype, Shape},
    devices::{
        binary_ops,
        device::{BinaryKernel, HasErr},
        Device,
    },
    gradients::{Merge, Tape},
    tensor::Tensor,
};

use super::utils::try_binary_op;

pub trait TryHuberError<Rhs = Self>: HasErr + HasDtype {
    fn huber_error(self, rhs: Rhs, delta: Self::Dtype) -> Self {
        self.try_huber_error(rhs, delta).unwrap()
    }
    fn try_huber_error(self, rhs: Rhs, delta: Self::Dtype) -> Result<Self, Self::Err>;
}

impl<S: Shape, E: Dtype, D: Device, LhsTape: Tape<D>, RhsTape: Tape<D>>
    TryHuberError<Tensor<S, E, D, RhsTape>> for Tensor<S, E, D, LhsTape>
where
    D: BinaryKernel<binary_ops::HuberError<E>, S, S, S, E>,
    LhsTape: Merge<RhsTape>,
{
    fn try_huber_error(self, rhs: Tensor<S, E, D, RhsTape>, delta: E) -> Result<Self, Self::Err> {
        try_binary_op(binary_ops::HuberError { delta }, self, rhs)
    }
}
