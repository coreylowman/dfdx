use crate::prelude::{Dtype, Webgpu};

impl<E: Dtype> super::SumKernel<E> for Webgpu {
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
        dst: Dst,
        inp: &impl crate::prelude::Tensorlike<Src, E, Self>,
        grad_inp: &mut Self::Vec,
        grad_out: &Self::Vec,
    ) -> Result<(), crate::prelude::Error>
    where
        Src: crate::prelude::ReduceShapeTo<Dst, Ax>,
    {
        todo!()
    }
}
