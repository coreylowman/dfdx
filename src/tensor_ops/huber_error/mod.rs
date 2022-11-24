mod cpu_kernel;

use crate::{
    arrays::{Dtype, HasDtype, Shape},
    devices::{Device, HasErr},
    gradients::{Merge, Tape},
    tensor::Tensor,
};

use super::utils::{try_binary_op, BinaryKernel};

pub trait HuberError<Rhs = Self>: HasErr + HasDtype {
    fn huber_error(self, rhs: Rhs, delta: Self::Dtype) -> Self {
        self.try_huber_error(rhs, delta).unwrap()
    }
    fn try_huber_error(self, rhs: Rhs, delta: Self::Dtype) -> Result<Self, Self::Err>;
}

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct HuberErrorKernelOp<E: Dtype> {
    pub delta: E,
}

impl<S: Shape, E: Dtype, D: Device, LhsTape: Tape<D>, RhsTape: Tape<D>>
    HuberError<Tensor<S, E, D, RhsTape>> for Tensor<S, E, D, LhsTape>
where
    D: BinaryKernel<HuberErrorKernelOp<E>, S, S, S, E>,
    LhsTape: Merge<RhsTape>,
{
    fn try_huber_error(self, rhs: Tensor<S, E, D, RhsTape>, delta: E) -> Result<Self, Self::Err> {
        try_binary_op(HuberErrorKernelOp { delta }, self, rhs)
    }
}
