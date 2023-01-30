use crate::shapes::{Dtype, Shape};
use crate::tensor::cpu::*;
use crate::tensor_ops::matmul::cpu_kernel::MatMulImpl;

use super::{Conv2DKernel, Conv2DOp};

use std::sync::Arc;

impl Conv2DOp {
    #[inline(always)]
    fn unfold_idx(&self, [k1, k2, y, x]: [usize; 4]) -> Option<[usize; 2]> {
        let mut oh = y + self.padding;
        if oh < k1 {
            return None;
        }
        oh -= k1;
        if oh % self.stride != 0 {
            return None;
        }
        oh /= self.stride;
        if oh >= self.h_out {
            return None;
        }

        let mut ow = x + self.padding;
        if ow < k2 {
            return None;
        }
        ow -= k2;
        if ow % self.stride != 0 {
            return None;
        }
        ow /= self.stride;
        if ow >= self.w_out {
            return None;
        }

        Some([oh, ow])
    }
}

impl Cpu {
    #[inline]
    fn conv2d_forward<F: Dtype, P: Shape<Concrete = [usize; 5]>>(
        &self,
        op: &Conv2DOp,
        img: &[F],
        filters: &[F],
        out: &mut [F],
        inp_patches_buf: &mut StridedArray<P, F>,
    ) -> Result<(), CpuError>
    where
        Self: MatMulImpl<F>,
    {
        {
            let buf = Arc::make_mut(&mut inp_patches_buf.data);
            let mut i = 0;
            for c in 0..op.chan_in {
                for k1 in 0..op.kernel {
                    for k2 in 0..op.kernel {
                        for oh in 0..op.h_out {
                            for ow in 0..op.w_out {
                                let y = (oh * op.stride + k1).wrapping_sub(op.padding);
                                let x = (ow * op.stride + k2).wrapping_sub(op.padding);
                                if y < op.h_in && x < op.w_in {
                                    buf[i] = img[c * (op.w_in * op.h_in) + y * op.w_in + x];
                                }
                                i += 1;
                            }
                        }
                    }
                }
            }
        }

        // (O, C * K * K) * (C * K * K, OH * OW) = (O, OH * OW)
        let m = op.chan_out;
        let k = op.chan_in * op.kernel * op.kernel;
        let n = op.w_out * op.h_out;
        Self::matmul(
            View::new(filters, (m, k)),
            View::new(inp_patches_buf.view().data, (k, n)),
            &mut ViewMut::new(out, (m, n)),
        );
        Ok(())
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn conv2d_backward<F: Dtype, P: Shape<Concrete = [usize; 5]>>(
        &self,
        op: &Conv2DOp,
        img: &[F],
        grad_img: &mut [F],
        filters_tr: &[F],
        grad_filters_tr: &mut [F],
        grad_out: &[F],
        out_patches_buf: &mut StridedArray<P, F>,
    ) -> Result<(), CpuError>
    where
        Self: MatMulImpl<F>,
    {
        {
            let mut i = 0;
            let buf = Arc::make_mut(&mut out_patches_buf.data);
            for o in 0..op.chan_out {
                for k1 in 0..op.kernel {
                    for k2 in 0..op.kernel {
                        for y in 0..op.h_in {
                            for x in 0..op.w_in {
                                if let Some([oh, ow]) = op.unfold_idx([k1, k2, y, x]) {
                                    buf[i] =
                                        grad_out[o * (op.h_out * op.w_out) + oh * op.w_out + ow];
                                }
                                i += 1;
                            }
                        }
                    }
                }
            }
        }

        {
            // img_g += filters^T * unfold(grad_out)
            // (C, H * W) += (C, O * K * K) * (O * K * K, H * W)
            let m = op.chan_in;
            let k = op.chan_out * op.kernel * op.kernel;
            let n = op.h_in * op.w_in;
            Self::matmul(
                View::new(filters_tr, (m, k)),
                View::new(out_patches_buf.view().data, (k, n)),
                &mut ViewMut::new(grad_img, (m, n)),
            );
        }

        {
            // weight_g^T += img * patches^T
            // (C, O * K * K) += (C, H * W) * (H * W, O * K * K)
            let m = op.chan_in;
            let k = op.h_in * op.w_in;
            let n = op.chan_out * op.kernel * op.kernel;
            Self::matmul(
                View::new(img, (m, k)),
                View::new(out_patches_buf.view().data, (n, k)).tr(),
                &mut ViewMut::new(grad_filters_tr, (m, n)),
            );
        }
        Ok(())
    }
}

impl<F: Dtype> Conv2DKernel<F> for Cpu
where
    Self: MatMulImpl<F>,
{
    fn forward<L: Shape, R: Shape, O: Shape>(
        &self,
        op: Conv2DOp,
        lhs: &Self::Storage<L, F>,
        rhs: &Self::Storage<R, F>,
        out: &mut Self::Storage<O, F>,
    ) -> Result<(), Self::Err> {
        let mut patches: StridedArray<_, F> = StridedArray::new(op.inp_patches_shape())?;
        let [lstride, ostride] = match L::NUM_DIMS {
            3 => [0; 2],
            4 => [lhs.strides[0], out.strides[0]],
            _ => unreachable!(),
        };
        let lhs = lhs.data.as_ref();
        let rhs = rhs.data.as_ref();
        let out = Arc::make_mut(&mut out.data);
        for i_batch in 0..op.batch {
            self.conv2d_forward(
                &op,
                &lhs[i_batch * lstride..],
                rhs,
                &mut out[i_batch * ostride..],
                &mut patches,
            )?;
        }
        Ok(())
    }

    fn backward<L: Shape, R: Shape, O: Shape>(
        &self,
        op: Conv2DOp,
        lhs: &Self::Storage<L, F>,
        grad_lhs: &mut Self::Storage<L, F>,
        rhs: &Self::Storage<R, F>,
        grad_rhs: &mut Self::Storage<R, F>,
        grad_out: &Self::Storage<O, F>,
    ) -> Result<(), Self::Err> {
        let mut patches: StridedArray<_, F> = StridedArray::new(op.out_patches_shape())?;
        let mut f1023: StridedArray<_, F> = StridedArray::new(op.filters_tr_shape())?;
        let mut grad_f1023: StridedArray<_, F> = StridedArray::new(op.filters_tr_shape())?;

        {
            // transpose filters in f1023
            let buf = rhs.data.as_ref();
            let mut f_iter = f1023.iter_mut_with_index();
            while let Some((f, [c, o, k1, k2])) = f_iter.next() {
                let idx = o * rhs.strides[0]
                    + c * rhs.strides[1]
                    + k1 * rhs.strides[2]
                    + k2 * rhs.strides[3];
                *f = buf[idx];
            }
        }

        let [lstride, ostride] = match L::NUM_DIMS {
            3 => [0; 2],
            4 => [lhs.strides[0], grad_out.strides[0]],
            _ => unreachable!(),
        };
        let lhs = lhs.data.as_ref();
        let grad_lhs = Arc::make_mut(&mut grad_lhs.data);
        let f = f1023.data.as_ref();
        let grad_f = Arc::make_mut(&mut grad_f1023.data);
        let grad_out = grad_out.data.as_ref();

        for i_batch in 0..op.batch {
            self.conv2d_backward(
                &op,
                &lhs[i_batch * lstride..],
                &mut grad_lhs[i_batch * lstride..],
                f,
                grad_f,
                &grad_out[i_batch * ostride..],
                &mut patches,
            )?;
        }

        {
            // untranspose filters
            let buf = Arc::make_mut(&mut grad_rhs.data);
            let mut f_iter = grad_f1023.iter_with_index();
            while let Some((f, [c, o, k1, k2])) = f_iter.next() {
                let idx = o * rhs.strides[0]
                    + c * rhs.strides[1]
                    + k1 * rhs.strides[2]
                    + k2 * rhs.strides[3];
                buf[idx] += *f;
            }
        }

        Ok(())
    }
}
