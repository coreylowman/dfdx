use crate::{
    prelude::NoneTape,
    shapes::*,
    tensor::{Cuda, Error, Tensor},
};
use cudarc::types::CudaTypeName;

impl<E: Dtype + CudaTypeName> super::UnstackKernel<E> for Cuda {
    fn forward<S: Shape, OptionalItems>(
        &self,
        _stack: Tensor<S, E, Self, NoneTape>,
    ) -> Result<OptionalItems, Error>
    where
        S: super::SubDim,
        OptionalItems: Array<Option<Tensor<S::Tail, E, Self, NoneTape>>, Dim = S::Head>,
    {
        todo!()
    }
    fn backward(
        &self,
        _grad_stack: &mut Self::Vec,
        _grad_unstack: &Self::Vec,
        _unstack_idx: usize,
    ) -> Result<(), Error> {
        todo!()
    }
}
