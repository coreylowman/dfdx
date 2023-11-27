use crate::prelude::{Dtype, Webgpu};

impl<E: Dtype> super::ReshapeKernel<E> for Webgpu {
    fn forward<Src: crate::prelude::Shape, Dst: crate::prelude::Shape>(
        &self,
        dst: &Dst,
        inp: &crate::prelude::Tensor<Src, E, Self>,
    ) -> Result<crate::prelude::Tensor<Dst, E, Self>, crate::prelude::Error> {
        todo!()
    }

    fn backward<Src: crate::prelude::Shape, Dst: crate::prelude::Shape>(
        &self,
        dst: &Dst,
        inp: &crate::prelude::Tensor<Src, E, Self>,
        grad_inp: &mut Self::Vec,
        grad_out: &Self::Vec,
    ) -> Result<(), crate::prelude::Error> {
        todo!()
    }
}
