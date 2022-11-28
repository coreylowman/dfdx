use crate::arrays::{Const, Dim, Dyn, HasShape, Rank3, Rank4, Rank5};
use crate::devices::ZerosLike;
use crate::devices::{
    cpu::{Cpu, CpuError, LendingIterator, StridedArray, View, ViewMut},
    Zeros,
};

use super::{Conv2DBatchedKernel, Conv2DKernel};
use crate::tensor_ops::matmul::cpu_kernel::matmul;

impl Cpu {
    fn conv2d_forward<
        const K: usize,
        const S: usize,
        const P: usize,
        const C: usize,
        const O: usize,
        const H: usize,
        const W: usize,
    >(
        &self,
        img: View<Rank3<C, H, W>, f32>,
        filters: View<Rank4<O, C, K, K>, f32>,
        out: ViewMut<Rank3<O, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }>, f32>,
    ) -> Result<(), CpuError> {
        let mut patches: StridedArray<
            Rank5<C, K, K, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }>,
            f32,
        > = self.try_zeros()?;

        let mut patch_iter = patches.iter_mut_with_index();
        while let Some((p, [c, k1, k2, oh, ow])) = patch_iter.next() {
            let y = (oh * S + k1).wrapping_sub(P);
            let x = (ow * S + k2).wrapping_sub(P);
            if y < H && x < W {
                *p = *img.idx(c).idx(y).idx(x);
            }
        }

        // (O, C * K * K) * (C * K * K, OH * OW) = (O, OH * OW)
        let m = Dyn(O);
        let k = Dyn(C * K * K);
        let n = Dyn(((H + 2 * P - K) / S + 1) * ((W + 2 * P - K) / S + 1));
        matmul(
            View::new(filters.ptr, (m, k)),
            View::new(patches.data.as_ptr(), (k, n)),
            ViewMut::new(out.ptr, (m, n)),
            1.0,
        );
        Ok(())
    }

    fn conv2d_backward<
        const K: usize,
        const S: usize,
        const P: usize,
        const C: usize,
        const O: usize,
        const H: usize,
        const W: usize,
    >(
        &self,
        img: View<Rank3<C, H, W>, f32>,
        grad_img: ViewMut<Rank3<C, H, W>, f32>,
        filters: View<Rank4<O, C, K, K>, f32>,
        grad_filters: ViewMut<Rank4<O, C, K, K>, f32>,
        grad_out: View<Rank3<O, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }>, f32>,
    ) -> Result<(), CpuError> {
        let mut tr_filters: StridedArray<Rank4<C, O, K, K>, f32> = self.try_zeros()?;

        {
            let tr_filters = tr_filters.view_mut();
            for c in 0..C {
                for o in 0..O {
                    for k1 in 0..K {
                        for k2 in 0..K {
                            *tr_filters.idx(c).idx(o).idx(k1).idx(k2) =
                                *filters.idx(o).idx(c).idx(k1).idx(k2);
                        }
                    }
                }
            }
        }

        let mut unfolded_grad_out: StridedArray<Rank5<O, K, K, H, W>, f32> = self.try_zeros()?;
        {
            let unfolded_grad_out = unfolded_grad_out.view_mut();
            for o in 0..O {
                for oh in 0..(H + 2 * P - K) / S + 1 {
                    for ow in 0..(W + 2 * P - K) / S + 1 {
                        let g = *grad_out.idx(o).idx(oh).idx(ow);
                        for k1 in 0..K {
                            for k2 in 0..K {
                                let y = (oh * S + k1).wrapping_sub(P);
                                let x = (ow * S + k2).wrapping_sub(P);
                                if y < H && x < W {
                                    *unfolded_grad_out.idx(o).idx(k1).idx(k2).idx(y).idx(x) = g;
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
            let m = Dyn(C);
            let k = Dyn(O * K * K);
            let n = Dyn(H * W);
            matmul(
                View::new(tr_filters.view().ptr, (m, k)),
                View::new(unfolded_grad_out.view().ptr, (k, n)),
                ViewMut::new(grad_img.ptr, (m, n)),
                1.0,
            );
        }

        {
            // weight_g^T += img * patches^T
            // (C, O * K * K) += (C, H * W) * (H * W, O * K * K)
            let m = Dyn(C);
            let k = Dyn(H * W);
            let n = Dyn(O * K * K);
            matmul(
                View::new(img.ptr, (m, k)),
                View::new(unfolded_grad_out.view().ptr, (n, k)).tr(),
                ViewMut::new(tr_filters.view_mut().ptr, (m, n)),
                0.0,
            );

            {
                let tr_filters = tr_filters.view();
                for o in 0..O {
                    for c in 0..C {
                        for k1 in 0..K {
                            for k2 in 0..K {
                                *grad_filters.idx(o).idx(c).idx(k1).idx(k2) +=
                                    tr_filters.idx(c).idx(o).idx(k1).idx(k2);
                            }
                        }
                    }
                }
            }
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
        let mut out: StridedArray<_, f32> = self.try_zeros()?;
        self.conv2d_forward::<K, S, P, C, O, H, W>(lhs.view(), rhs.view(), out.view_mut())?;
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
        self.conv2d_backward::<K, S, P, C, O, H, W>(
            lhs.view(),
            grad_lhs.view_mut(),
            rhs.view(),
            grad_rhs.view_mut(),
            grad_out.view(),
        )
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
        let (batch, _, _, _) = lhs.shape();
        let mut out: StridedArray<_, f32> = self.try_zeros_like((*batch, Const, Const, Const))?;
        {
            let lhs = lhs.view();
            let rhs = rhs.view();
            let out = out.view_mut();
            for b in 0..batch.size() {
                self.conv2d_forward::<K, S, P, C, O, H, W>(lhs.idx(b), rhs, out.idx(b))?;
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
        let lhs = lhs.view();
        let grad_lhs = grad_lhs.view_mut();
        let rhs = rhs.view();
        let grad_rhs = grad_rhs.view_mut();
        let grad_out = grad_out.view();
        for b in 0..batch.size() {
            self.conv2d_backward::<K, S, P, C, O, H, W>(
                lhs.idx(b),
                grad_lhs.idx(b),
                rhs,
                grad_rhs,
                grad_out.idx(b),
            )?;
        }
        Ok(())
    }
}
