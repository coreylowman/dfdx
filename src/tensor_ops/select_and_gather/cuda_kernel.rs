#![allow(clippy::needless_range_loop)]

use crate::shapes::{Axes, Dtype, RemoveDimTo, ReplaceDimTo, Shape};
use crate::tensor::Cuda;

impl<E: Dtype> super::ReplaceDimKernel<E> for Cuda {
    fn forward<Src: Shape, Dst: Shape, Idx: Shape>(
        &self,
        inp: &Self::Storage<Src, E>,
        idx: &Self::Storage<Idx, usize>,
    ) -> Result<Self::Storage<Dst, E>, Self::Err>
    where
        Src: ReplaceDimTo<Dst, Idx>,
    {
        todo!()
    }

    fn backward<Src: Shape, Dst: Shape, Idx: Shape>(
        &self,
        grad_inp: &mut Self::Storage<Src, E>,
        idx: &Self::Storage<Idx, usize>,
        grad_out: &Self::Storage<Dst, E>,
    ) -> Result<(), Self::Err>
    where
        Src: ReplaceDimTo<Dst, Idx>,
    {
        todo!()
    }
}

impl<E: Dtype> super::RemoveDimKernel<E> for Cuda {
    fn forward<Src: Shape, Dst: Shape, Idx: Shape>(
        &self,
        inp: &Self::Storage<Src, E>,
        idx: &Self::Storage<Idx, usize>,
    ) -> Result<Self::Storage<Dst, E>, Self::Err>
    where
        Src: RemoveDimTo<Dst, Idx>,
    {
        todo!()
    }

    fn backward<Src: Shape, Dst: Shape, Idx: Shape>(
        &self,
        grad_inp: &mut Self::Storage<Src, E>,
        idx: &Self::Storage<Idx, usize>,
        grad_out: &Self::Storage<Dst, E>,
    ) -> Result<(), Self::Err>
    where
        Src: RemoveDimTo<Dst, Idx>,
    {
        todo!()
    }
}
