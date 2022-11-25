use crate::arrays::{Rank3, Rank4, Rank5};
use crate::devices::{
    cpu::{Cpu, CpuError, LendingIterator, StridedArray, View, ViewMut},
    Zeros,
};
use crate::tensor_ops::ops::BinaryKernel;

use super::Conv2DKernelOp;

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
        let m = O;
        let k = C * K * K;
        let n = ((H + 2 * P - K) / S + 1) * ((W + 2 * P - K) / S + 1);
        let a = filters.ptr;
        let b = patches.data.as_ptr();
        let c = out.ptr;
        #[cfg(not(feature = "cblas"))]
        unsafe {
            matrixmultiply::sgemm(
                m, k, n, 1.0, a, k as isize, 1, b, n as isize, 1, 1.0, c, n as isize, 1,
            )
        }

        #[cfg(feature = "cblas")]
        unsafe {
            let (m, n, k) = (m as libc::c_int, n as libc::c_int, k as libc::c_int);
            sgemm(RowMajor, NoTr, NoTr, m, n, k, 1.0, a, k, b, n, 1.0, c, n)
        }

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

            let m = C;
            let k = O * K * K;
            let n = H * W;
            let a = tr_filters.view().ptr;
            let b = unfolded_grad_out.view().ptr;
            let c = grad_img.ptr;
            #[cfg(not(feature = "cblas"))]
            unsafe {
                matrixmultiply::sgemm(
                    m, k, n, 1.0, a, k as isize, 1, b, n as isize, 1, 1.0, c, n as isize, 1,
                )
            }

            #[cfg(feature = "cblas")]
            unsafe {
                let (m, n, k) = (m as libc::c_int, n as libc::c_int, k as libc::c_int);
                sgemm(RowMajor, NoTr, NoTr, m, n, k, 1.0, a, k, b, n, 1.0, c, n)
            }
        }

        {
            // weight_g^T += img * patches^T
            // (C, O * K * K) += (C, H * W) * (H * W, O * K * K)

            let m = C;
            let k = H * W;
            let n = O * K * K;
            let a = img.ptr;
            let b = unfolded_grad_out.view().ptr;
            let c = tr_filters.view_mut().ptr;
            #[cfg(not(feature = "cblas"))]
            unsafe {
                matrixmultiply::sgemm(
                    m, k, n, 1.0, a, k as isize, 1, b, 1, k as isize, 0.0, c, n as isize, 1,
                )
            }
            #[cfg(feature = "cblas")]
            unsafe {
                let (m, n, k) = (m as libc::c_int, n as libc::c_int, k as libc::c_int);
                sgemm(RowMajor, NoTr, Tr, m, n, k, 1.0, a, k, b, k, 0.0, c, n)
            }

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

impl<
        const K: usize,
        const S: usize,
        const P: usize,
        const C: usize,
        const O: usize,
        const H: usize,
        const W: usize,
    >
    BinaryKernel<
        Conv2DKernelOp<K, S, P>,
        Rank3<C, H, W>,
        Rank4<O, C, K, K>,
        Rank3<O, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }>,
        f32,
    > for Cpu
{
    fn binary_fwd(
        &self,
        _op: Conv2DKernelOp<K, S, P>,
        lhs: &Self::Storage<Rank3<C, H, W>, f32>,
        rhs: &Self::Storage<Rank4<O, C, K, K>, f32>,
    ) -> Result<
        Self::Storage<Rank3<O, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }>, f32>,
        Self::Err,
    > {
        let mut out: StridedArray<
            Rank3<O, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }>,
            f32,
        > = self.try_zeros()?;
        self.conv2d_forward::<K, S, P, C, O, H, W>(lhs.view(), rhs.view(), out.view_mut())?;
        Ok(out)
    }
    fn binary_bwd(
        &self,
        _op: Conv2DKernelOp<K, S, P>,
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

impl<
        const B: usize,
        const K: usize,
        const S: usize,
        const P: usize,
        const C: usize,
        const O: usize,
        const H: usize,
        const W: usize,
    >
    BinaryKernel<
        Conv2DKernelOp<K, S, P>,
        Rank4<B, C, H, W>,
        Rank4<O, C, K, K>,
        Rank4<B, O, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }>,
        f32,
    > for Cpu
{
    fn binary_fwd(
        &self,
        _op: Conv2DKernelOp<K, S, P>,
        lhs: &Self::Storage<Rank4<B, C, H, W>, f32>,
        rhs: &Self::Storage<Rank4<O, C, K, K>, f32>,
    ) -> Result<
        Self::Storage<Rank4<B, O, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }>, f32>,
        Self::Err,
    > {
        let mut out: StridedArray<
            Rank4<B, O, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }>,
            f32,
        > = self.try_zeros()?;
        {
            let lhs = lhs.view();
            let rhs = rhs.view();
            let out = out.view_mut();
            for b in 0..B {
                self.conv2d_forward::<K, S, P, C, O, H, W>(lhs.idx(b), rhs, out.idx(b))?;
            }
        }
        Ok(out)
    }
    fn binary_bwd(
        &self,
        _op: Conv2DKernelOp<K, S, P>,
        lhs: &Self::Storage<Rank4<B, C, H, W>, f32>,
        grad_lhs: &mut Self::Storage<Rank4<B, C, H, W>, f32>,
        rhs: &Self::Storage<Rank4<O, C, K, K>, f32>,
        grad_rhs: &mut Self::Storage<Rank4<O, C, K, K>, f32>,
        grad_out: &Self::Storage<
            Rank4<B, O, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }>,
            f32,
        >,
    ) -> Result<(), Self::Err> {
        let lhs = lhs.view();
        let grad_lhs = grad_lhs.view_mut();
        let rhs = rhs.view();
        let grad_rhs = grad_rhs.view_mut();
        let grad_out = grad_out.view();
        for b in 0..B {
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
