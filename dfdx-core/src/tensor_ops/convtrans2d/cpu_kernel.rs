use crate::prelude::Tensorlike;
use crate::shapes::{Dtype, Shape};
use crate::tensor::{cpu::*, Error, Tensor, ZerosTensor};
use crate::tensor_ops::matmul::cpu_kernel::MatMulImpl;

use std::sync::Arc;

use super::{ConvTrans2DKernel, ConvTrans2DOp};

impl ConvTrans2DOp {
    #[inline(always)]
    fn unfold_idx(&self, [k1, k2, y, x]: [usize; 4]) -> Option<[usize; 2]> {
        (y * self.stride + self.dilation * k1)
            .checked_sub(self.padding)
            .zip((x * self.stride + self.dilation * k2).checked_sub(self.padding))
            .filter(|&(oh, ow)| oh < self.h_out && ow < self.w_out)
            .map(|(oh, ow)| [oh, ow])
    }
}

impl Cpu {
    #[inline]
    fn convtrans2d_forward<E: Dtype>(
        &self,
        op: &ConvTrans2DOp,
        img: &[E],
        filters_tr: &[E],
        out: &mut [E],
        buf: &mut [E],
    ) -> Result<(), Error>
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
                                i += 1;
                                let mut y = oh + op.padding;
                                if y < op.dilation * k1 {
                                    continue;
                                }
                                y -= op.dilation * k1;
                                if y % op.stride != 0 {
                                    continue;
                                }
                                y /= op.stride;
                                if y >= op.h_in {
                                    continue;
                                }

                                let mut x = ow + op.padding;
                                if x < op.dilation * k2 {
                                    continue;
                                }
                                x -= op.dilation * k2;
                                if x % op.stride != 0 {
                                    continue;
                                }
                                x /= op.stride;
                                if x >= op.w_in {
                                    continue;
                                }

                                if y < op.h_in && x < op.w_in {
                                    buf[i - 1] = img[c * (op.w_in * op.h_in) + y * op.w_in + x];
                                }
                            }
                        }
                    }
                }
            }
        }

        // filters_tr: (G, O/G, C/G*K*K)
        // patches: (G, C/G*K*K, OH*OW)
        // output: (G, O/G, OH*OW)
        let m = op.chan_out / op.groups;
        let k = (op.chan_in / op.groups) * op.kernel * op.kernel;
        let n = op.w_out * op.h_out;
        for g in 0..op.groups {
            Self::matmul(
                (m, k, n),
                false,
                filters_tr[g * m * k..].as_ptr(),
                [k, 1],
                buf[g * k * n..].as_ptr(),
                [n, 1],
                out[g * m * n..].as_mut_ptr(),
                [n, 1],
            );
        }
        Ok(())
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn convtrans2d_backward<E: Dtype>(
        &self,
        op: &ConvTrans2DOp,
        img: &[E],
        grad_img: &mut [E],
        filters: &[E],
        grad_filters: &mut [E],
        grad_out: &[E],
        buf: &mut [E],
    ) -> Result<(), Error>
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
            // filters: (G, C/G, O/G*K*K)
            // buf: (G, O/G*K*K, H*W)
            // grad_img: (G, C/G, H * W)
            let m = op.chan_in / op.groups;
            let k = (op.chan_out / op.groups) * op.kernel * op.kernel;
            let n = op.h_in * op.w_in;
            for g in 0..op.groups {
                Self::matmul(
                    (m, k, n),
                    true,
                    filters[g * m * k..].as_ptr(),
                    [k, 1],
                    buf[g * k * n..].as_ptr(),
                    [n, 1],
                    grad_img[g * m * n..].as_mut_ptr(),
                    [n, 1],
                );
            }
        }

        {
            // img: (G, C/G, H * W)
            // buf: (G, H * W, O/G * K * K)
            // grad_filters: (G, C/G, O/G * K * K)
            let m = op.chan_in / op.groups;
            let k = op.h_in * op.w_in;
            let n = (op.chan_out / op.groups) * op.kernel * op.kernel;
            for g in 0..op.groups {
                Self::matmul(
                    (m, k, n),
                    true,
                    img[g * m * k..].as_ptr(),
                    [k, 1],
                    buf[g * k * n..].as_ptr(),
                    [1, k],
                    grad_filters[g * m * n..].as_mut_ptr(),
                    [n, 1],
                );
            }
        }
        Ok(())
    }
}

impl<E: Dtype> ConvTrans2DKernel<E> for Cpu
where
    Self: MatMulImpl<E>,
{
    fn alloc<S: Shape>(&self, s: S) -> Result<Tensor<S, E, Self>, Error> {
        self.try_zeros_like(&s)
    }

    fn forward<L: Shape, R: Shape, O: Shape>(
        &self,
        op: ConvTrans2DOp,
        lhs: &Tensor<L, E, Self>,
        rhs: &Tensor<R, E, Self>,
        out: &mut Tensor<O, E, Self>,
    ) -> Result<(), Error> {
        let patches = (op.chan_in, op.kernel, op.kernel, op.h_out, op.w_out);
        let mut patches = self.try_alloc_zeros::<E>(patches.num_elements())?;
        let f_tr_shape = [
            op.groups,
            op.chan_out / op.groups,
            op.chan_in / op.groups,
            op.kernel,
            op.kernel,
        ];
        let mut f_tr = self.try_alloc_zeros::<E>(f_tr_shape.num_elements())?;

        {
            // transpose filters in f1023
            let buf = rhs.data.as_ref();
            let mut f_idx = NdIndex::new(f_tr_shape, f_tr_shape.strides());
            while let Some((i, [g, o_over_g, c_over_g, k1, k2])) = f_idx.next_with_idx() {
                let idx = (g * (op.chan_in / op.groups) + c_over_g) * rhs.strides[0]
                    + o_over_g * rhs.strides[1]
                    + k1 * rhs.strides[2]
                    + k2 * rhs.strides[3];
                f_tr[i] = buf[idx];
            }
        }

        let [lstride, ostride] = match L::NUM_DIMS {
            3 => [0; 2],
            4 => [lhs.strides[0], out.strides[0]],
            _ => unreachable!(),
        };
        let lhs = lhs.data.as_ref();
        let out = Arc::make_mut(&mut out.data);
        for i_batch in 0..op.batch {
            self.convtrans2d_forward(
                &op,
                &lhs[i_batch * lstride..],
                &f_tr,
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
        grad_lhs: &mut Self::Vec,
        rhs: &Tensor<R, E, Self>,
        grad_rhs: &mut Self::Vec,
        out: &impl Tensorlike<O, E, Self>,
        grad_out: &Self::Vec,
    ) -> Result<(), Error> {
        let patches_shape = [op.chan_out, op.kernel, op.kernel, op.h_in, op.w_in];
        let mut patches = self.try_alloc_zeros::<E>(patches_shape.num_elements())?;

        let [lstride, ostride] = match L::NUM_DIMS {
            3 => [0; 2],
            4 => [lhs.strides[0], out.strides()[0]],
            _ => unreachable!(),
        };
        let lhs = lhs.data.as_ref();

        let rhs = rhs.data.as_ref();
        for i_batch in 0..op.batch {
            self.convtrans2d_backward(
                &op,
                &lhs[i_batch * lstride..],
                &mut grad_lhs[i_batch * lstride..],
                rhs,
                grad_rhs,
                &grad_out[i_batch * ostride..],
                &mut patches,
            )?;
        }

        Ok(())
    }
}
