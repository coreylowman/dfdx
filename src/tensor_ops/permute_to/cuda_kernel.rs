use crate::shapes::*;
use crate::tensor::Cuda;

impl<E: Dtype> super::PermuteKernel<E> for Cuda {
    fn forward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        inp: &Self::Storage<Src, E>,
    ) -> Result<Self::Storage<Dst, E>, Self::Err>
    where
        Src: PermuteShapeTo<Dst, Ax>,
    {
        todo!()
    }
    fn backward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        grad_inp: &mut Self::Storage<Src, E>,
        grad_out: &Self::Storage<Dst, E>,
    ) -> Result<(), Self::Err>
    where
        Src: PermuteShapeTo<Dst, Ax>,
    {
        todo!()
    }
}
