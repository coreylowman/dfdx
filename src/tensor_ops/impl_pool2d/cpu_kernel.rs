use crate::arrays::{Rank3, Rank4};
use crate::devices::{Zeros, cpu::{Cpu, StridedArray, View, ViewMut}};
use crate::tensor_ops::utils::UnaryKernel;

use super::{pooling, Pool2DKernelOp};

trait Pooling<const K: usize, const S: usize, const P: usize> {
    fn pool_forward<const C: usize, const H: usize, const W: usize>(
        &self,
        inp: View<Rank3<C, H, W>, f32>,
        out: ViewMut<Rank3<C, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }>, f32>,
    );
    fn pool_backward<const C: usize, const H: usize, const W: usize>(
        &self,
        inp: View<Rank3<C, H, W>, f32>,
        inp_grad: ViewMut<Rank3<C, H, W>, f32>,
        out_grad: View<Rank3<C, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }>, f32>,
    );
}

impl<const K: usize, const S: usize, const P: usize> Pooling<K, S, P>
    for Pool2DKernelOp<pooling::Avg, K, S, P>
{
    fn pool_forward<const C: usize, const H: usize, const W: usize>(
        &self,
        inp: View<Rank3<C, H, W>, f32>,
        out: ViewMut<Rank3<C, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }>, f32>,
    ) {
        let out_height = (H + 2 * P - K) / S + 1;
        let out_width = (W + 2 * P - K) / S + 1;
        let inv_k2 = 1.0 / (K * K) as f32;
        for c in 0..C {
            for oh in 0..out_height {
                for ow in 0..out_width {
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

    fn pool_backward<const C: usize, const H: usize, const W: usize>(
        &self,
        _inp: View<Rank3<C, H, W>, f32>,
        inp_grad: ViewMut<Rank3<C, H, W>, f32>,
        out_grad: View<Rank3<C, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }>, f32>,
    ) {
        let out_height = (H + 2 * P - K) / S + 1;
        let out_width = (W + 2 * P - K) / S + 1;
        let inv_k2 = 1.0 / (K * K) as f32;
        for c in 0..C {
            for oh in 0..out_height {
                for ow in 0..out_width {
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

impl<const K: usize, const S: usize, const P: usize> Pooling<K, S, P>
    for Pool2DKernelOp<pooling::Max, K, S, P>
{
    fn pool_forward<const C: usize, const H: usize, const W: usize>(
        &self,
        inp: View<Rank3<C, H, W>, f32>,
        out: ViewMut<Rank3<C, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }>, f32>,
    ) {
        let out_height = (H + 2 * P - K) / S + 1;
        let out_width = (W + 2 * P - K) / S + 1;
        for c in 0..C {
            for oh in 0..out_height {
                for ow in 0..out_width {
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

    fn pool_backward<const C: usize, const H: usize, const W: usize>(
        &self,
        inp: View<Rank3<C, H, W>, f32>,
        inp_grad: ViewMut<Rank3<C, H, W>, f32>,
        out_grad: View<Rank3<C, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }>, f32>,
    ) {
        let out_height = (H + 2 * P - K) / S + 1;
        let out_width = (W + 2 * P - K) / S + 1;
        for c in 0..C {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let o_g = *out_grad.idx(c).idx(oh).idx(ow);
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

                    for k1 in 0..K {
                        let y = (oh * S + k1).checked_sub(P);
                        for k2 in 0..K {
                            let x = (ow * S + k2).checked_sub(P);
                            if let Some((y, x)) = y.zip(x) {
                                if y < H && x < W && *inp.idx(c).idx(y).idx(x) == tmp {
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

impl<const K: usize, const S: usize, const P: usize> Pooling<K, S, P>
    for Pool2DKernelOp<pooling::Min, K, S, P>
{
    fn pool_forward<const C: usize, const H: usize, const W: usize>(
        &self,
        inp: View<Rank3<C, H, W>, f32>,
        out: ViewMut<Rank3<C, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }>, f32>,
    ) {
        let out_height = (H + 2 * P - K) / S + 1;
        let out_width = (W + 2 * P - K) / S + 1;
        for c in 0..C {
            for oh in 0..out_height {
                for ow in 0..out_width {
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

    fn pool_backward<const C: usize, const H: usize, const W: usize>(
        &self,
        inp: View<Rank3<C, H, W>, f32>,
        inp_grad: ViewMut<Rank3<C, H, W>, f32>,
        out_grad: View<Rank3<C, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }>, f32>,
    ) {
        let out_height = (H + 2 * P - K) / S + 1;
        let out_width = (W + 2 * P - K) / S + 1;
        for c in 0..C {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let o_g = *out_grad.idx(c).idx(oh).idx(ow);
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

                    for k1 in 0..K {
                        let y = (oh * S + k1).checked_sub(P);
                        for k2 in 0..K {
                            let x = (ow * S + k2).checked_sub(P);
                            if let Some((y, x)) = y.zip(x) {
                                if y < H && x < W && *inp.idx(c).idx(y).idx(x) == tmp {
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

impl<
        const K: usize,
        const S: usize,
        const P: usize,
        const C: usize,
        const H: usize,
        const W: usize,
        Kind: 'static,
    >
    UnaryKernel<
        Pool2DKernelOp<Kind, K, S, P>,
        Rank3<C, H, W>,
        Rank3<C, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }>,
        f32,
    > for Cpu
where
    Pool2DKernelOp<Kind, K, S, P>: Pooling<K, S, P>,
{
    fn unary_fwd(
        &self,
        op: Pool2DKernelOp<Kind, K, S, P>,
        inp: &Self::Storage<Rank3<C, H, W>, f32>,
    ) -> Result<
        Self::Storage<Rank3<C, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }>, f32>,
        Self::Err,
    > {
        let mut out: StridedArray<
            Rank3<C, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }>,
            f32,
        > = self.try_zeros()?;
        op.pool_forward(inp.view(), out.view_mut());
        Ok(out)
    }
    fn unary_bwd(
        &self,
        op: Pool2DKernelOp<Kind, K, S, P>,
        inp: &Self::Storage<Rank3<C, H, W>, f32>,
        grad_inp: &mut Self::Storage<Rank3<C, H, W>, f32>,
        grad_out: &Self::Storage<
            Rank3<C, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }>,
            f32,
        >,
    ) -> Result<(), Self::Err> {
        op.pool_backward(inp.view(), grad_inp.view_mut(), grad_out.view());
        Ok(())
    }
}

impl<
        const B: usize,
        const K: usize,
        const S: usize,
        const P: usize,
        const C: usize,
        const H: usize,
        const W: usize,
        Kind: 'static,
    >
    UnaryKernel<
        Pool2DKernelOp<Kind, K, S, P>,
        Rank4<B, C, H, W>,
        Rank4<B, C, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }>,
        f32,
    > for Cpu
where
    Pool2DKernelOp<Kind, K, S, P>: Pooling<K, S, P>,
{
    fn unary_fwd(
        &self,
        op: Pool2DKernelOp<Kind, K, S, P>,
        inp: &Self::Storage<Rank4<B, C, H, W>, f32>,
    ) -> Result<
        Self::Storage<Rank4<B, C, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }>, f32>,
        Self::Err,
    > {
        let mut out: StridedArray<
            Rank4<B, C, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }>,
            f32,
        > = self.try_zeros()?;
        let inp = inp.view();
        let out_view = out.view_mut();
        for b in 0..B {
            op.pool_forward(inp.idx(b), out_view.idx(b));
        }
        Ok(out)
    }
    fn unary_bwd(
        &self,
        op: Pool2DKernelOp<Kind, K, S, P>,
        inp: &Self::Storage<Rank4<B, C, H, W>, f32>,
        grad_inp: &mut Self::Storage<Rank4<B, C, H, W>, f32>,
        grad_out: &Self::Storage<
            Rank4<B, C, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }>,
            f32,
        >,
    ) -> Result<(), Self::Err> {
        let inp = inp.view();
        let grad_inp = grad_inp.view_mut();
        let grad_out = grad_out.view();
        for b in 0..B {
            op.pool_backward(inp.idx(b), grad_inp.idx(b), grad_out.idx(b));
        }
        Ok(())
    }
}
