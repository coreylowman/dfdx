use crate::prelude::{Dtype, Webgpu};

impl<E: Dtype> super::SliceKernel<E> for Webgpu {
    fn forward<Src: crate::prelude::Shape + crate::prelude::SliceShape<Slice>, Slice>(
        &self,
        inp: &crate::prelude::Tensor<Src, E, Self>,
        slice: &Slice,
    ) -> Result<crate::prelude::Tensor<Src::Sliced, E, Self>, crate::prelude::Error> {
        todo!()
    }

    fn backward<Src: crate::prelude::Shape + crate::prelude::SliceShape<Slice>, Slice>(
        &self,
        inp: &crate::prelude::Tensor<Src, E, Self>,
        grad_inp: &mut Self::Vec,
        grad_out: &Self::Vec,
        slice: &Slice,
    ) -> Result<(), crate::prelude::Error> {
        todo!()
    }
}
