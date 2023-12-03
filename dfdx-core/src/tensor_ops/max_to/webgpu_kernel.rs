use crate::prelude::{Dtype, Webgpu};

impl<E: Dtype> super::MaxReduceKernel<E> for Webgpu {
    fn forward<Src: crate::prelude::Shape, Dst: crate::prelude::Shape, Ax: crate::prelude::Axes>(
        &self,
        dst: Dst,
        inp: &crate::prelude::Tensor<Src, E, Self>,
    ) -> Result<crate::prelude::Tensor<Dst, E, Self>, crate::prelude::Error>
    where
        Src: crate::prelude::ReduceShapeTo<Dst, Ax>,
    {
        todo!()
    }

    fn backward<Src: crate::prelude::Shape, Dst: crate::prelude::Shape, Ax: crate::prelude::Axes>(
        &self,
        inp: &crate::prelude::Tensor<Src, E, Self>,
        grad_inp: &mut Self::Vec,
        out: &crate::prelude::Tensor<Dst, E, Self>,
        grad_out: &Self::Vec,
    ) -> Result<(), crate::prelude::Error>
    where
        Src: crate::prelude::ReduceShapeTo<Dst, Ax>,
    {
        todo!()
    }
}
