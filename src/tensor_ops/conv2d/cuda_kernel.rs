use crate::{shapes::*, tensor::Cuda};

impl<const K: usize, const S: usize, const P: usize, const C: usize, const O: usize>
    super::Conv2DKernel<f32, C, O, K, S, P> for Cuda
{
    fn forward<const H: usize, const W: usize>(
        &self,
        lhs: &Self::Storage<Rank3<C, H, W>, f32>,
        rhs: &Self::Storage<Rank4<O, C, K, K>, f32>,
    ) -> Result<
        Self::Storage<Rank3<O, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }>, f32>,
        Self::Err,
    > {
        todo!()
    }

    fn backward<const H: usize, const W: usize>(
        &self,
        lhs: &Self::Storage<Rank3<C, H, W>, f32>,
        grad_lhs: &mut Self::Storage<Rank3<C, H, W>, f32>,
        rhs: &Self::Storage<Rank4<O, C, K, K>, f32>,
        grad_rhs: &mut Self::Storage<Rank4<O, C, K, K>, f32>,
        grad_out: &Self::Storage<
            Rank3<O, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }>,
            f32,
        >,
    ) -> Result<(), Self::Err> {
        todo!()
    }
}

impl<const K: usize, const S: usize, const P: usize, const C: usize, const O: usize>
    super::Conv2DBatchedKernel<f32, C, O, K, S, P> for Cuda
{
    #[rustfmt::skip]
    fn forward<B: Dim, const H: usize, const W: usize>(
        &self,
        lhs: &Self::Storage<(B, Const<C>, Const<H>, Const<W>), f32>,
        rhs: &Self::Storage<Rank4<O, C, K, K>, f32>,
    ) -> Result<
        Self::Storage<
            (B, Const<O>, Const<{ (H + 2 * P - K) / S + 1 }>, Const<{ (W + 2 * P - K) / S + 1 }>),
            f32,
        >,
        Self::Err,
    > {
        todo!()
    }
    #[rustfmt::skip]
    fn backward<B: Dim, const H: usize, const W: usize>(
        &self,
        lhs: &Self::Storage<(B, Const<C>, Const<H>, Const<W>), f32>,
        grad_lhs: &mut Self::Storage<(B, Const<C>, Const<H>, Const<W>), f32>,
        rhs: &Self::Storage<Rank4<O, C, K, K>, f32>,
        grad_rhs: &mut Self::Storage<Rank4<O, C, K, K>, f32>,
        grad_out: &Self::Storage<
            (B, Const<O>, Const<{ (H + 2 * P - K) / S + 1 }>, Const<{ (W + 2 * P - K) / S + 1 }>),
            f32,
        >,
    ) -> Result<(), Self::Err> {
        todo!()
    }
}
