use crate::prelude::{Dtype, Webgpu};

impl<E: Dtype> super::ReplaceDimKernel<E> for Webgpu {
    fn forward<Src: crate::prelude::Shape, Dst: crate::prelude::Shape, Idx: crate::prelude::Shape>(
        &self,
        inp: &crate::prelude::Tensor<Src, E, Self>,
        idx: &crate::prelude::Tensor<Idx, usize, Self>,
    ) -> Result<crate::prelude::Tensor<Dst, E, Self>, crate::prelude::Error>
    where
        Src: crate::prelude::ReplaceDimTo<Dst, Idx>,
    {
        todo!()
    }

    fn backward<
        Src: crate::prelude::Shape,
        Dst: crate::prelude::Shape,
        Idx: crate::prelude::Shape,
    >(
        &self,
        inp: &crate::prelude::Tensor<Src, E, Self>,
        grad_inp: &mut <Self as crate::prelude::Storage<E>>::Vec,
        idx: &crate::prelude::Tensor<Idx, usize, Self>,
        out: &crate::prelude::Tensor<Dst, E, Self>,
        grad_out: &<Self as crate::prelude::Storage<E>>::Vec,
    ) -> Result<(), crate::prelude::Error>
    where
        Src: crate::prelude::ReplaceDimTo<Dst, Idx>,
    {
        todo!()
    }
}

impl<E: Dtype> super::RemoveDimKernel<E> for Webgpu {
    fn forward<Src: crate::prelude::Shape, Dst: crate::prelude::Shape, Idx: crate::prelude::Shape>(
        &self,
        inp: &crate::prelude::Tensor<Src, E, Self>,
        idx: &crate::prelude::Tensor<Idx, usize, Self>,
    ) -> Result<crate::prelude::Tensor<Dst, E, Self>, crate::prelude::Error>
    where
        Src: crate::prelude::RemoveDimTo<Dst, Idx>,
    {
        todo!()
    }

    fn backward<
        Src: crate::prelude::Shape,
        Dst: crate::prelude::Shape,
        Idx: crate::prelude::Shape,
    >(
        &self,
        inp: &crate::prelude::Tensor<Src, E, Self>,
        grad_inp: &mut <Self as crate::prelude::Storage<E>>::Vec,
        idx: &crate::prelude::Tensor<Idx, usize, Self>,
        out: &crate::prelude::Tensor<Dst, E, Self>,
        grad_out: &<Self as crate::prelude::Storage<E>>::Vec,
    ) -> Result<(), crate::prelude::Error>
    where
        Src: crate::prelude::RemoveDimTo<Dst, Idx>,
    {
        todo!()
    }
}
