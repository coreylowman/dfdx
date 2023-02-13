use crate::{
    shapes::*,
    tensor::{
        cpu::{Cpu, StridedArray},
        Tensor,
    },
};

impl<E: Dtype> super::TryStackKernel<E> for Cpu {
    fn forward<S: Shape, Num: Dim>(
        &self,
        num: Num,
        inp: &[Self::Storage<S, E>],
    ) -> Result<Self::Storage<S::Larger, E>, Self::Err>
    where
        S: super::AddDim<Num>,
    {
        todo!()
    }
    fn backward<S: Shape, New: Dim>(
        &self,
        num: New,
        inp: &[Self::Storage<S, E>],
        grad_inp: &mut [Self::Storage<S, E>],
        grad_out: &[Self::Storage<S::Larger, E>],
    ) -> Result<(), Self::Err>
    where
        S: super::AddDim<New>,
    {
        todo!()
    }
}
