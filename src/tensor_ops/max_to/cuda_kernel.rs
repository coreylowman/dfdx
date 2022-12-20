use crate::{
    shapes::{Axes, ReduceShapeTo, Shape},
    tensor::Cuda,
};

impl super::MaxReduceKernel<f32> for Cuda {
    fn forward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        dst: Dst,
        inp: &Self::Storage<Src, f32>,
    ) -> Result<Self::Storage<Dst, f32>, Self::Err>
    where
        Src: ReduceShapeTo<Dst, Ax>,
    {
        todo!()
    }

    fn backward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        inp: &Self::Storage<Src, f32>,
        grad_inp: &mut Self::Storage<Src, f32>,
        out: &Self::Storage<Dst, f32>,
        grad_out: &Self::Storage<Dst, f32>,
    ) -> Result<(), Self::Err>
    where
        Src: ReduceShapeTo<Dst, Ax>,
    {
        todo!()
    }
}
