use crate::shapes::{Dtype, Shape};
use crate::tensor::{cpu::*, Tensor};
use crate::tensor_ops::matmul::cpu_kernel::MatMulImpl;

use std::sync::Arc;

use super::{ConvTrans2DOp, ConvTrans2DKernel};

impl ConvTrans2DOp {
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
    fn convtrans2d_forward<E: Dtype>(
        &self,
        op: &ConvTrans2DOp,
        img: &[E],
        filters: &[E],
        out: &mut [E],
        buf: &mut [E],
    ) -> Result<(), CpuError>
    where
        Self: MatMulImpl<E>,
    {
        {
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
            (m, k, n),
            filters.as_ptr(),
            [k, 1],
            buf.as_ptr(),
            [n, 1],
            out.as_mut_ptr(),
            [n, 1],
        );
        Ok(())
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn convtrans2d_backward<E: Dtype>(
        &self,
        op: &ConvTrans2DOp,
        img: &[E],
        grad_img: &mut [E],
        filters_tr: &[E],
        grad_filters_tr: &mut [E],
        grad_out: &[E],
        buf: &mut [E],
    ) -> Result<(), CpuError>
    where
        Self: MatMulImpl<E>,
    {
        {
            let mut i = 0;
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
                (m, k, n),
                filters_tr.as_ptr(),
                [k, 1],
                buf.as_ptr(),
                [n, 1],
                grad_img.as_mut_ptr(),
                [n, 1],
            );
        }

        {
            // weight_g^T += img * patches^T
            // (C, O * K * K) += (C, H * W) * (H * W, O * K * K)
            let m = op.chan_in;
            let k = op.h_in * op.w_in;
            let n = op.chan_out * op.kernel * op.kernel;
            Self::matmul(
                (m, k, n),
                img.as_ptr(),
                [k, 1],
                buf.as_ptr(),
                [1, k],
                grad_filters_tr.as_mut_ptr(),
                [n, 1],
            );
        }
        Ok(())
    }
}

impl<E: Dtype> ConvTrans2DKernel<E> for Cpu
where
    Self: MatMulImpl<E>,
{
    fn forward<L: Shape, R: Shape, O: Shape>(
        &self,
        op: ConvTrans2DOp,
        lhs: &Tensor<L, E, Self>,
        rhs: &Tensor<R, E, Self>,
        out: &mut Tensor<O, E, Self>,
    ) -> Result<(), Self::Err> {
        let mut patches = self.try_alloc_zeros::<E>(op.inp_patches_shape().num_elements())?;
        let [lstride, ostride] = match L::NUM_DIMS {
            3 => [0; 2],
            4 => [lhs.strides[0], out.strides[0]],
            _ => unreachable!(),
        };
        let lhs = lhs.data.as_ref();
        let rhs = rhs.data.as_ref();
        let out = Arc::make_mut(&mut out.data);
        for i_batch in 0..op.batch {
            self.convtrans2d_forward(
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
        op: ConvTrans2DOp,
        lhs: &Tensor<L, E, Self>,
        grad_lhs: &mut Self::Vec<E>,
        rhs: &Tensor<R, E, Self>,
        grad_rhs: &mut Self::Vec<E>,
        out: &Tensor<O, E, Self>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        let f_tr_shape = op.filters_tr_shape();
        let mut patches = self.try_alloc_zeros::<E>(op.out_patches_shape().num_elements())?;
        let mut f1023 = self.try_alloc_zeros::<E>(f_tr_shape.num_elements())?;
        let mut grad_f1023 = self.try_alloc_zeros::<E>(f_tr_shape.num_elements())?;

        {
            // transpose filters in f1023
            let buf = rhs.data.as_ref();
            let mut f_idx = NdIndex::new(f_tr_shape, f_tr_shape.strides());
            while let Some((i, [c, o, k1, k2])) = f_idx.next_with_idx() {
                let idx = o * rhs.strides[0]
                    + c * rhs.strides[1]
                    + k1 * rhs.strides[2]
                    + k2 * rhs.strides[3];
                f1023[i] = buf[idx];
            }
        }

        let [lstride, ostride] = match L::NUM_DIMS {
            3 => [0; 2],
            4 => [lhs.strides[0], out.strides[0]],
            _ => unreachable!(),
        };
        let lhs = lhs.data.as_ref();

        for i_batch in 0..op.batch {
            self.convtrans2d_backward(
                &op,
                &lhs[i_batch * lstride..],
                &mut grad_lhs[i_batch * lstride..],
                &f1023,
                &mut grad_f1023,
                &grad_out[i_batch * ostride..],
                &mut patches,
            )?;
        }

        {
            // untranspose filters
            let mut f_idx = NdIndex::new(f_tr_shape, f_tr_shape.strides());
            while let Some((i, [c, o, k1, k2])) = f_idx.next_with_idx() {
                let idx = o * rhs.strides[0]
                    + c * rhs.strides[1]
                    + k1 * rhs.strides[2]
                    + k2 * rhs.strides[3];
                grad_rhs[idx] += grad_f1023[i];
            }
        }

        Ok(())
    }
}
