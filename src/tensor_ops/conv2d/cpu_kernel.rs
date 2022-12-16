use crate::shapes::{Const, Dim, HasShape, Rank3, Rank4};
use crate::tensor::cpu::*;
use crate::tensor_ops::matmul::cpu_kernel::matmul;

use super::{Conv2DBatchedKernel, Conv2DKernel};

impl Cpu {
    fn conv2d_forward<C: Dim, H: Dim, W: Dim, O: Dim, K: Dim, OH: Dim, OW: Dim>(
        &self,
        stride: usize,
        padding: usize,
        img: View<(C, H, W), f32>,
        filters: View<(O, C, K, K), f32>,
        out: &mut ViewMut<(O, OH, OW), f32>,
        inp_patches_buf: &mut StridedArray<(C, K, K, OH, OW), f32>,
    ) -> Result<(), CpuError> {
        let (chan, height, width) = img.shape;
        let (out_chan, out_height, out_width) = out.shape;
        let kernel = filters.shape.3;

        let mut patch_iter = inp_patches_buf.iter_mut_with_index();
        while let Some((p, [c, k1, k2, oh, ow])) = patch_iter.next() {
            let y = (oh * stride + k1).wrapping_sub(padding);
            let x = (ow * stride + k2).wrapping_sub(padding);
            if y < height.size() && x < width.size() {
                *p = *img.idx(c).idx(y).idx(x);
            }
        }

        // (O, C * K * K) * (C * K * K, OH * OW) = (O, OH * OW)
        let m = out_chan;
        let k = chan.size() * kernel.size() * kernel.size();
        let n = out_width.size() * out_height.size();
        matmul(
            View::new(filters.data, (m, k)),
            View::new(inp_patches_buf.view().data, (k, n)),
            &mut ViewMut::new(out.data, (m, n)),
        );
        Ok(())
    }

    fn conv2d_backward<C: Dim, H: Dim, W: Dim, O: Dim, K: Dim, OH: Dim, OW: Dim>(
        &self,
        stride: usize,
        padding: usize,
        img: View<(C, H, W), f32>,
        grad_img: &mut ViewMut<(C, H, W), f32>,
        filters_tr: View<(C, O, K, K), f32>,
        grad_filters_tr: &mut ViewMut<(C, O, K, K), f32>,
        grad_out: View<(O, OH, OW), f32>,
        out_patches_buf: &mut StridedArray<(O, K, K, H, W), f32>,
    ) -> Result<(), CpuError> {
        let (chan, height, width) = img.shape;
        let (out_chan, out_height, out_width) = grad_out.shape;
        let kernel = filters_tr.shape.3;

        {
            let mut out_patches_buf = out_patches_buf.view_mut();
            for o in 0..out_chan.size() {
                for oh in 0..out_height.size() {
                    for ow in 0..out_width.size() {
                        let g = *grad_out.idx(o).idx(oh).idx(ow);
                        for k1 in 0..kernel.size() {
                            for k2 in 0..kernel.size() {
                                let y = (oh * stride + k1).wrapping_sub(padding);
                                let x = (ow * stride + k2).wrapping_sub(padding);
                                if y < height.size() && x < width.size() {
                                    *out_patches_buf
                                        .idx_mut(o)
                                        .idx_mut(k1)
                                        .idx_mut(k2)
                                        .idx_mut(y)
                                        .idx_mut(x) = g;
                                }
                            }
                        }
                    }
                }
            }
        }

        {
            // img_g += filters^T * unfold(grad_out)
            // (C, H * W) += (C, O * K * K) * (O * K * K, H * W)
            let m = chan;
            let k = out_chan.size() * kernel.size() * kernel.size();
            let n = height.size() * width.size();
            matmul(
                View::new(filters_tr.data, (m, k)),
                View::new(out_patches_buf.view().data, (k, n)),
                &mut ViewMut::new(grad_img.data, (m, n)),
            );
        }

        {
            // weight_g^T += img * patches^T
            // (C, O * K * K) += (C, H * W) * (H * W, O * K * K)
            let m = chan;
            let k = height.size() * width.size();
            let n = out_chan.size() * kernel.size() * kernel.size();
            matmul(
                View::new(img.data, (m, k)),
                View::new(out_patches_buf.view().data, (n, k)).tr(),
                &mut ViewMut::new(grad_filters_tr.data, (m, n)),
            );
        }

        Ok(())
    }
}

impl<const K: usize, const S: usize, const P: usize, const C: usize, const O: usize>
    Conv2DKernel<f32, C, O, K, S, P> for Cpu
{
    fn forward<const H: usize, const W: usize>(
        &self,
        lhs: &Self::Storage<Rank3<C, H, W>, f32>,
        rhs: &Self::Storage<Rank4<O, C, K, K>, f32>,
    ) -> Result<
        Self::Storage<Rank3<O, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }>, f32>,
        Self::Err,
    > {
        let mut out: StridedArray<_, f32> = StridedArray::new(Default::default())?;
        let mut patches: StridedArray<_, f32> = StridedArray::new(Default::default())?;
        self.conv2d_forward(
            S,
            P,
            lhs.view(),
            rhs.view(),
            &mut out.view_mut(),
            &mut patches,
        )?;
        Ok(out)
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
        let mut patches: StridedArray<_, f32> = StridedArray::new(Default::default())?;
        let mut f1023: StridedArray<Rank4<C, O, K, K>, f32> =
            StridedArray::new(Default::default())?;
        let mut grad_f1023: StridedArray<_, f32> = StridedArray::new(Default::default())?;

        {
            let rhs_view = rhs.view();
            let mut f_iter = f1023.iter_mut_with_index();
            while let Some((f, [c, o, k1, k2])) = f_iter.next() {
                *f = *rhs_view.idx(o).idx(c).idx(k1).idx(k2);
            }
        }

        self.conv2d_backward(
            S,
            P,
            lhs.view(),
            &mut grad_lhs.view_mut(),
            f1023.view(),
            &mut grad_f1023.view_mut(),
            grad_out.view(),
            &mut patches,
        )?;

        {
            let grad_f1023 = grad_f1023.view();
            let mut iter = grad_rhs.iter_mut_with_index();
            while let Some((f, [o, c, k1, k2])) = iter.next() {
                *f += *grad_f1023.idx(c).idx(o).idx(k1).idx(k2);
            }
        }

        Ok(())
    }
}

impl<const K: usize, const S: usize, const P: usize, const C: usize, const O: usize>
    Conv2DBatchedKernel<f32, C, O, K, S, P> for Cpu
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
        let batch = lhs.shape.0;
        let mut patches: StridedArray<_, f32> = StridedArray::new(Default::default())?;
        let mut out: StridedArray<_, f32> = StridedArray::new((batch, Const, Const, Const))?;
        {
            let lhs = lhs.view();
            let rhs = rhs.view();
            let mut out = out.view_mut();
            for b in 0..batch.size() {
                self.conv2d_forward(S, P, lhs.idx(b), rhs, &mut out.idx_mut(b), &mut patches)?;
            }
        }
        Ok(out)
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
        let (batch, _, _, _) = lhs.shape();
        let mut patches: StridedArray<_, f32> = StridedArray::new(Default::default())?;
        let mut filters_1023: StridedArray<Rank4<C, O, K, K>, f32> =
            StridedArray::new(Default::default())?;
        let mut grad_filters_1023: StridedArray<Rank4<C, O, K, K>, f32> =
            StridedArray::new(Default::default())?;

        {
            let rhs_view = rhs.view();
            let mut f_iter = filters_1023.iter_mut_with_index();
            while let Some((f, [c, o, k1, k2])) = f_iter.next() {
                *f = *rhs_view.idx(o).idx(c).idx(k1).idx(k2);
            }
        }

        let lhs = lhs.view();
        let mut grad_lhs = grad_lhs.view_mut();
        let grad_out = grad_out.view();
        for b in 0..batch.size() {
            self.conv2d_backward(
                S, P,
                lhs.idx(b), &mut grad_lhs.idx_mut(b),
                filters_1023.view(), &mut grad_filters_1023.view_mut(),
                grad_out.idx(b),
                &mut patches
            )?;
        }

        {
            let grad_filters_1023 = grad_filters_1023.view();
            let mut iter = grad_rhs.iter_mut_with_index();
            while let Some((f, [o, c, k1, k2])) = iter.next() {
                *f += *grad_filters_1023.idx(c).idx(o).idx(k1).idx(k2);
            }
        }

        Ok(())
    }
}
