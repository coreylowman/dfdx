use crate::arrays::{Const, Dim};
use crate::tensor::storage::cpu::{Cpu, StridedArray, View, ViewMut};
use crate::tensor::storage::ZerosLike;

use super::{pooling, Pool2DBatchedKernel, Pool2DKernel};

trait Pooling<const K: usize, const S: usize, const P: usize> {
    #[rustfmt::skip]
    fn forward<C: Dim, const H: usize, const W: usize>(
        inp: View<(C, Const<H>, Const<W>), f32>,
        out: ViewMut<(C, Const<{ (H + 2 * P - K) / S + 1 }>, Const<{ (W + 2 * P - K) / S + 1 }>), f32>,
    );
    #[rustfmt::skip]
    fn backward<C: Dim, const H: usize, const W: usize>(
        inp: View<(C, Const<H>, Const<W>), f32>,
        inp_grad: ViewMut<(C, Const<H>, Const<W>), f32>,
        out: View<(C, Const<{ (H + 2 * P - K) / S + 1 }>, Const<{ (W + 2 * P - K) / S + 1 }>), f32>,
        out_grad: View<(C, Const<{ (H + 2 * P - K) / S + 1 }>, Const<{ (W + 2 * P - K) / S + 1 }>), f32>,
    );
}

impl<const K: usize, const S: usize, const P: usize> Pooling<K, S, P> for pooling::Avg {
    #[rustfmt::skip]
    fn forward<C: Dim, const H: usize, const W: usize>(
        inp: View<(C, Const<H>, Const<W>), f32>,
        out: ViewMut<(C, Const<{ (H + 2 * P - K) / S + 1 }>, Const<{ (W + 2 * P - K) / S + 1 }>), f32>,
    ) {
        let inv_k2 = 1.0 / (K * K) as f32;
        for c in 0..inp.shape.0.size() {
            for oh in 0..out.shape.1.size() {
                for ow in 0..out.shape.2.size() {
                    let mut tmp = 0.0;
                    for k1 in 0..K {
                        let y = (oh * S + k1).checked_sub(P);
                        for k2 in 0..K {
                            let x = (ow * S + k2).checked_sub(P);
                            if let Some((y, x)) = y.zip(x) {
                                if y < H && x < W {
                                    tmp += inp.idx(c).idx(y).idx(x);
                                }
                            }
                        }
                    }
                    *out.idx(c).idx(oh).idx(ow) = tmp * inv_k2;
                }
            }
        }
    }

    #[rustfmt::skip]
    fn backward<C: Dim, const H: usize, const W: usize>(
        inp: View<(C, Const<H>, Const<W>), f32>,
        inp_grad: ViewMut<(C, Const<H>, Const<W>), f32>,
        out: View<(C, Const<{ (H + 2 * P - K) / S + 1 }>, Const<{ (W + 2 * P - K) / S + 1 }>), f32>,
        out_grad: View<(C, Const<{ (H + 2 * P - K) / S + 1 }>, Const<{ (W + 2 * P - K) / S + 1 }>), f32>,
    ) {
        let inv_k2 = 1.0 / (K * K) as f32;
        for c in 0..inp.shape.0.size() {
            for oh in 0..out.shape.1.size() {
                for ow in 0..out.shape.2.size() {
                    let g = out_grad.idx(c).idx(oh).idx(ow) * inv_k2;
                    for k1 in 0..K {
                        let y = (oh * S + k1).checked_sub(P);
                        for k2 in 0..K {
                            let x = (ow * S + k2).checked_sub(P);
                            if let Some((y, x)) = y.zip(x) {
                                if x < W && y < H {
                                    *inp_grad.idx(c).idx(y).idx(x) += g;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

impl<const K: usize, const S: usize, const P: usize> Pooling<K, S, P> for pooling::Max {
    #[rustfmt::skip]
    fn forward<C: Dim, const H: usize, const W: usize>(
        inp: View<(C, Const<H>, Const<W>), f32>,
        out: ViewMut<(C, Const<{ (H + 2 * P - K) / S + 1 }>, Const<{ (W + 2 * P - K) / S + 1 }>), f32>,
    ) {
        for c in 0..inp.shape.0.size() {
            for oh in 0..out.shape.1.size() {
                for ow in 0..out.shape.2.size() {
                    let mut tmp = f32::NEG_INFINITY;
                    for k1 in 0..K {
                        let y = (oh * S + k1).checked_sub(P);
                        for k2 in 0..K {
                            let x = (ow * S + k2).checked_sub(P);
                            if let Some((y, x)) = y.zip(x) {
                                if y < H && x < W {
                                    tmp = inp.idx(c).idx(y).idx(x).max(tmp);
                                }
                            }
                        }
                    }
                    *out.idx(c).idx(oh).idx(ow) = tmp;
                }
            }
        }
    }
    #[rustfmt::skip]
    fn backward<C: Dim, const H: usize, const W: usize>(
        inp: View<(C, Const<H>, Const<W>), f32>,
        inp_grad: ViewMut<(C, Const<H>, Const<W>), f32>,
        out: View<(C, Const<{ (H + 2 * P - K) / S + 1 }>, Const<{ (W + 2 * P - K) / S + 1 }>), f32>,
        out_grad: View<(C, Const<{ (H + 2 * P - K) / S + 1 }>, Const<{ (W + 2 * P - K) / S + 1 }>), f32>,
    ) {
        let out_height = (H + 2 * P - K) / S + 1;
        let out_width = (W + 2 * P - K) / S + 1;
        for c in 0..inp.shape.0.size() {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let o_g = *out_grad.idx(c).idx(oh).idx(ow);
                    let o = *out.idx(c).idx(oh).idx(ow);
                    for k1 in 0..K {
                        let y = (oh * S + k1).checked_sub(P);
                        for k2 in 0..K {
                            let x = (ow * S + k2).checked_sub(P);
                            if let Some((y, x)) = y.zip(x) {
                                if y < H && x < W && *inp.idx(c).idx(y).idx(x) == o {
                                    *inp_grad.idx(c).idx(y).idx(x) += o_g;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

impl<const K: usize, const S: usize, const P: usize> Pooling<K, S, P> for pooling::Min {
    #[rustfmt::skip]
    fn forward<C: Dim, const H: usize, const W: usize>(
        inp: View<(C, Const<H>, Const<W>), f32>,
        out: ViewMut<(C, Const<{ (H + 2 * P - K) / S + 1 }>, Const<{ (W + 2 * P - K) / S + 1 }>), f32>,
    ) {
        for c in 0..inp.shape.0.size() {
            for oh in 0..out.shape.1.size() {
                for ow in 0..out.shape.2.size() {
                    let mut tmp = f32::INFINITY;
                    for k1 in 0..K {
                        let y = (oh * S + k1).checked_sub(P);
                        for k2 in 0..K {
                            let x = (ow * S + k2).checked_sub(P);
                            if let Some((y, x)) = y.zip(x) {
                                if y < H && x < W {
                                    tmp = inp.idx(c).idx(y).idx(x).min(tmp);
                                }
                            }
                        }
                    }
                    *out.idx(c).idx(oh).idx(ow) = tmp;
                }
            }
        }
    }
    #[rustfmt::skip]
    fn backward<C: Dim, const H: usize, const W: usize>(
        inp: View<(C, Const<H>, Const<W>), f32>,
        inp_grad: ViewMut<(C, Const<H>, Const<W>), f32>,
        out: View<(C, Const<{ (H + 2 * P - K) / S + 1 }>, Const<{ (W + 2 * P - K) / S + 1 }>), f32>,
        out_grad: View<(C, Const<{ (H + 2 * P - K) / S + 1 }>, Const<{ (W + 2 * P - K) / S + 1 }>), f32>,
    ) {
        for c in 0..out.shape.0.size() {
            for oh in 0..out.shape.1.size() {
                for ow in 0..out.shape.2.size() {
                    let o_g = *out_grad.idx(c).idx(oh).idx(ow);
                    let o = *out.idx(c).idx(oh).idx(ow);
                    for k1 in 0..K {
                        let y = (oh * S + k1).checked_sub(P);
                        for k2 in 0..K {
                            let x = (ow * S + k2).checked_sub(P);
                            if let Some((y, x)) = y.zip(x) {
                                if y < H && x < W && *inp.idx(c).idx(y).idx(x) == o {
                                    *inp_grad.idx(c).idx(y).idx(x) += o_g;
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
        let mut out: StridedArray<_, f32> = self.try_zeros_like((c, Const, Const))?;
        Kind::forward(inp.view(), out.view_mut());
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
        Kind::backward(inp.view(), grad_inp.view_mut(), out.view(), grad_out.view());
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
        let mut out: StridedArray<_, f32> = self.try_zeros_like((batch, chan, Const, Const))?;
        let inp = inp.view();
        let out_view = out.view_mut();
        for b in 0..batch.size() {
            Kind::forward(inp.idx(b), out_view.idx(b));
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
        let grad_inp = grad_inp.view_mut();
        let out = out.view();
        let grad_out = grad_out.view();
        for b in 0..batch.size() {
            Kind::backward(inp.idx(b), grad_inp.idx(b), out.idx(b), grad_out.idx(b));
        }
        Ok(())
    }
}
