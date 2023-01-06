use crate::shapes::{Dtype, HasSameNumelAs, Shape};
use crate::tensor::Cuda;

impl<E: Dtype> super::ReshapeKernel<E> for Cuda {
    fn forward<Src: Shape, Dst: Shape>(
        &self,
        dst: Dst,
        inp: &Self::Storage<Src, E>,
    ) -> Result<Self::Storage<Dst, E>, Self::Err>
    where
        Src: HasSameNumelAs<Dst>,
    {
        todo!()
    }

    fn backward<Src: Shape, Dst: Shape>(
        &self,
        grad_inp: &mut Self::Storage<Src, E>,
        grad_out: &Self::Storage<Dst, E>,
    ) -> Result<(), Self::Err>
    where
        Src: HasSameNumelAs<Dst>,
    {
        todo!()
    }
}
