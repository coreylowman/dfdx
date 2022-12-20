use crate::{shapes::*, tensor::Cuda};

impl<const K: usize, const S: usize, const P: usize, Kind> super::Pool2DKernel<f32, Kind, K, S, P>
    for Cuda
{
    #[rustfmt::skip]
    fn forward<C: Dim, const H: usize, const W: usize>(
        &self,
        inp: &Self::Storage<(C, Const<H>, Const<W>), f32>,
    ) -> Result<
        Self::Storage<(C, Const<{ (H + 2 * P - K) / S + 1 }>, Const<{ (W + 2 * P - K) / S + 1 }>), f32>,
        Self::Err,
    > {
        todo!()
    }

    #[rustfmt::skip]
    fn backward<C: Dim, const H: usize, const W: usize>(
        &self,
        inp: &Self::Storage<(C, Const<H>, Const<W>), f32>,
        grad_inp: &mut Self::Storage<(C, Const<H>, Const<W>), f32>,
        out: &Self::Storage<(C, Const<{ (H + 2 * P - K) / S + 1 }>, Const<{ (W + 2 * P - K) / S + 1 }>), f32>,
        grad_out: &Self::Storage<(C, Const<{ (H + 2 * P - K) / S + 1 }>, Const<{ (W + 2 * P - K) / S + 1 }>), f32>,
    ) -> Result<(), Self::Err> {
        todo!()
    }
}

impl<const K: usize, const S: usize, const P: usize, Kind>
    super::Pool2DBatchedKernel<f32, Kind, K, S, P> for Cuda
{
    #[rustfmt::skip]
    fn forward<B: Dim, C: Dim, const H: usize, const W: usize>(
        &self,
        inp: &Self::Storage<(B, C, Const<H>, Const<W>), f32>,
    ) -> Result<
        Self::Storage<(B, C, Const<{ (H + 2 * P - K) / S + 1 }>, Const<{ (W + 2 * P - K) / S + 1 }>), f32>,
        Self::Err,
    > {
        todo!()
    }

    #[rustfmt::skip]
    fn backward<B: Dim, C: Dim, const H: usize, const W: usize>(
        &self,
        inp: &Self::Storage<(B, C, Const<H>, Const<W>), f32>,
        grad_inp: &mut Self::Storage<(B, C, Const<H>, Const<W>), f32>,
        out: &Self::Storage<(B, C, Const<{ (H + 2 * P - K) / S + 1 }>, Const<{ (W + 2 * P - K) / S + 1 }>), f32>,
        grad_out: &Self::Storage<(B, C, Const<{ (H + 2 * P - K) / S + 1 }>, Const<{ (W + 2 * P - K) / S + 1 }>), f32>,
    ) -> Result<(), Self::Err> {
        todo!()
    }
}
