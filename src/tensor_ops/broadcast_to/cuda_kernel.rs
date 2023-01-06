use crate::{shapes::*, tensor::Cuda};

impl<E: Dtype> super::BroadcastKernel<E> for Cuda {
    fn forward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        dst: Dst,
        inp: &Self::Storage<Src, E>,
    ) -> Result<Self::Storage<Dst, E>, Self::Err>
    where
        Src: BroadcastShapeTo<Dst, Ax>,
    {
        todo!()
    }
    fn backward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        grad_inp: &mut Self::Storage<Src, E>,
        grad_out: &Self::Storage<Dst, E>,
    ) -> Result<(), Self::Err>
    where
        Src: BroadcastShapeTo<Dst, Ax>,
    {
        todo!()
    }
}
