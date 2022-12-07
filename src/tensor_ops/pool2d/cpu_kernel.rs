use crate::shapes::{Const, Dim};
use crate::tensor::cpu::{Cpu, StridedArray, View, ViewMut};

use super::{pooling, Pool2DBatchedKernel, Pool2DKernel};

trait Fold {
    fn init() -> f32;
    fn fold(accum: &mut f32, item: &f32);
    fn finalize(accum: &mut f32, num: f32);
    fn filter(item: &f32, grad: &f32) -> bool;
}

impl Fold for pooling::Avg {
    fn init() -> f32 {
        0.0
    }
    fn fold(accum: &mut f32, item: &f32) {
        *accum += item;
    }
    fn finalize(accum: &mut f32, num: f32) {
        *accum /= num;
    }
    fn filter(_: &f32, _: &f32) -> bool {
        true
    }
}

impl Fold for pooling::Max {
    fn init() -> f32 {
        f32::NEG_INFINITY
    }
    fn fold(accum: &mut f32, item: &f32) {
        *accum = accum.max(*item);
    }
    fn finalize(_: &mut f32, _: f32) {}
    fn filter(a: &f32, b: &f32) -> bool {
        a == b
    }
}

impl Fold for pooling::Min {
    fn init() -> f32 {
        f32::INFINITY
    }
    fn fold(accum: &mut f32, item: &f32) {
        *accum = accum.min(*item);
    }
    fn finalize(_: &mut f32, _: f32) {}
    fn filter(a: &f32, b: &f32) -> bool {
        a == b
    }
}

trait Pooling<const K: usize, const S: usize, const P: usize> {
    #[rustfmt::skip]
    fn forward<C: Dim, const H: usize, const W: usize>(
        inp: View<(C, Const<H>, Const<W>), f32>,
        out: &mut ViewMut<(C, Const<{ (H + 2 * P - K) / S + 1 }>, Const<{ (W + 2 * P - K) / S + 1 }>), f32>,
    );
    #[rustfmt::skip]
    fn backward<C: Dim, const H: usize, const W: usize>(
        inp: View<(C, Const<H>, Const<W>), f32>,
        inp_grad: &mut ViewMut<(C, Const<H>, Const<W>), f32>,
        out: View<(C, Const<{ (H + 2 * P - K) / S + 1 }>, Const<{ (W + 2 * P - K) / S + 1 }>), f32>,
        out_grad: View<(C, Const<{ (H + 2 * P - K) / S + 1 }>, Const<{ (W + 2 * P - K) / S + 1 }>), f32>,
    );
}

impl<F: 'static + Fold, const K: usize, const S: usize, const P: usize> Pooling<K, S, P> for F {
    #[rustfmt::skip]
    fn forward<C: Dim, const H: usize, const W: usize>(
        inp: View<(C, Const<H>, Const<W>), f32>,
        out: &mut ViewMut<(C, Const<{ (H + 2 * P - K) / S + 1 }>, Const<{ (W + 2 * P - K) / S + 1 }>), f32>,
    ) {
        for c in 0..inp.shape.0.size() {
            for oh in 0..out.shape.1.size() {
                for ow in 0..out.shape.2.size() {
                    let mut tmp = Self::init();
                    for k1 in 0..K {
                        let y = (oh * S + k1).checked_sub(P);
                        for k2 in 0..K {
                            let x = (ow * S + k2).checked_sub(P);
                            if let Some((y, x)) = y.zip(x) {
                                if y < H && x < W {
                                    Self::fold(&mut tmp, inp.idx(c).idx(y).idx(x));
                                }
                            }
                        }
                    }
                    Self::finalize(&mut tmp, (K * K) as f32);
                    *out.idx_mut(c).idx_mut(oh).idx_mut(ow) = tmp;
                }
            }
        }
    }

    #[rustfmt::skip]
    fn backward<C: Dim, const H: usize, const W: usize>(
        inp: View<(C, Const<H>, Const<W>), f32>,
        inp_grad: &mut ViewMut<(C, Const<H>, Const<W>), f32>,
        out: View<(C, Const<{ (H + 2 * P - K) / S + 1 }>, Const<{ (W + 2 * P - K) / S + 1 }>), f32>,
        out_grad: View<(C, Const<{ (H + 2 * P - K) / S + 1 }>, Const<{ (W + 2 * P - K) / S + 1 }>), f32>,
    ) {
        for c in 0..inp.shape.0.size() {
            for oh in 0..out.shape.1.size() {
                for ow in 0..out.shape.2.size() {
                    let mut g = *out_grad.idx(c).idx(oh).idx(ow);
                    Self::finalize(&mut g, (K * K) as f32);
                    for k1 in 0..K {
                        let y = (oh * S + k1).checked_sub(P);
                        for k2 in 0..K {
                            let x = (ow * S + k2).checked_sub(P);
                            if let Some((y, x)) = y.zip(x) {
                                if x < W && y < H && Self::filter(inp.idx(c).idx(y).idx(x), out.idx(c).idx(oh).idx(ow)) {
                                    *inp_grad.idx_mut(c).idx_mut(y).idx_mut(x) += g;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

impl<const K: usize, const S: usize, const P: usize, Kind: Pooling<K, S, P>>
    Pool2DKernel<f32, Kind, K, S, P> for Cpu
{
    #[rustfmt::skip]
    fn forward<C: Dim, const H: usize, const W: usize>(
        &self,
        inp: &Self::Storage<(C, Const<H>, Const<W>), f32>,
    ) -> Result<
        Self::Storage<(C, Const<{ (H + 2 * P - K) / S + 1 }>, Const<{ (W + 2 * P - K) / S + 1 }>), f32>,
        Self::Err,
    > {
        let (c, _, _) = inp.shape;
        let mut out: StridedArray<_, f32> = StridedArray::new((c, Const, Const))?;
        Kind::forward(inp.view(), &mut out.view_mut());
        Ok(out)
    }

    #[rustfmt::skip]
    fn backward<C: Dim, const H: usize, const W: usize>(
        &self,
        inp: &Self::Storage<(C, Const<H>, Const<W>), f32>,
        grad_inp: &mut Self::Storage<(C, Const<H>, Const<W>), f32>,
        out: &Self::Storage<(C, Const<{ (H + 2 * P - K) / S + 1 }>, Const<{ (W + 2 * P - K) / S + 1 }>), f32>,
        grad_out: &Self::Storage<(C, Const<{ (H + 2 * P - K) / S + 1 }>, Const<{ (W + 2 * P - K) / S + 1 }>), f32>,
    ) -> Result<(), Self::Err> {
        Kind::backward(inp.view(), &mut grad_inp.view_mut(), out.view(), grad_out.view());
        Ok(())
    }
}

impl<const K: usize, const S: usize, const P: usize, Kind: Pooling<K, S, P>>
    Pool2DBatchedKernel<f32, Kind, K, S, P> for Cpu
{
    #[rustfmt::skip]
    fn forward<B: Dim, C: Dim, const H: usize, const W: usize>(
        &self,
        inp: &Self::Storage<(B, C, Const<H>, Const<W>), f32>,
    ) -> Result<
        Self::Storage<(B, C, Const<{ (H + 2 * P - K) / S + 1 }>, Const<{ (W + 2 * P - K) / S + 1 }>), f32>,
        Self::Err,
    > {
        let (batch, chan, _, _) = inp.shape;
        let mut out: StridedArray<_, f32> = StridedArray::new((batch, chan, Const, Const))?;
        let inp = inp.view();
        let mut out_view = out.view_mut();
        for b in 0..batch.size() {
            Kind::forward(inp.idx(b), &mut out_view.idx_mut(b));
        }
        Ok(out)
    }

    #[rustfmt::skip]
    fn backward<B: Dim, C: Dim, const H: usize, const W: usize>(
        &self,
        inp: &Self::Storage<(B, C, Const<H>, Const<W>), f32>,
        grad_inp: &mut Self::Storage<(B, C, Const<H>, Const<W>), f32>,
        out: &Self::Storage<(B, C, Const<{ (H + 2 * P - K) / S + 1 }>, Const<{ (W + 2 * P - K) / S + 1 }>), f32>,
        grad_out: &Self::Storage<(B, C, Const<{ (H + 2 * P - K) / S + 1 }>, Const<{ (W + 2 * P - K) / S + 1 }>), f32>,
    ) -> Result<(), Self::Err> {
        let (batch, _, _, _) = inp.shape;
        let inp = inp.view();
        let mut grad_inp = grad_inp.view_mut();
        let out = out.view();
        let grad_out = grad_out.view();
        for b in 0..batch.size() {
            Kind::backward(inp.idx(b), &mut grad_inp.idx_mut(b), out.idx(b), grad_out.idx(b));
        }
        Ok(())
    }
}
