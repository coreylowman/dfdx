use crate::shapes::{Dtype, Shape};
use crate::tensor::{cpu::*, *};
use crate::tensor_ops::matmul::cpu_kernel::MatMulImpl;

use super::{Conv2DKernel, Conv2DOp};

use std::sync::Arc;

impl Conv2DOp {
    #[inline(always)]
    fn unfold_idx(&self, [k1, k2, y, x]: [usize; 4]) -> Option<[usize; 2]> {
        let mut oh = y + self.padding;
        if oh < self.dilation * k1 {
            return None;
        }
        oh -= self.dilation * k1;
        if oh % self.stride != 0 {
            return None;
        }
        oh /= self.stride;
        if oh >= self.h_out {
            return None;
        }

        let mut ow = x + self.padding;
        if ow < self.dilation * k2 {
            return None;
        }
        ow -= self.dilation * k2;
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
    fn fwd<E: Dtype>(
        &self,
        op: &Conv2DOp,
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
                            let y = (oh * op.stride + op.dilation * k1).wrapping_sub(op.padding);
                            for ow in 0..op.w_out {
                                let x =
                                    (ow * op.stride + op.dilation * k2).wrapping_sub(op.padding);
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

        // filters: (G, O/G, C/G*K*K)
        // buf:     (G, C/G*K*K, OH*OW)
        // output:  (G, O/G, OH*OW)
        let m = op.chan_out / op.groups;
        let k = (op.chan_in / op.groups) * op.kernel * op.kernel;
        let n = op.w_out * op.h_out;
        for g in 0..op.groups {
            Self::matmul(
                (m, k, n),
                false,
                filters[g * m * k..].as_ptr(),
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
    fn bwd<E: Dtype>(
        &self,
        op: &Conv2DOp,
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
            // (G, C/G, H * W) += (G, C/G, O/G * K * K) * (G, O/G * K * K, H * W)
            let m = op.chan_in / op.groups;
            let k = (op.chan_out / op.groups) * op.kernel * op.kernel;
            let n = op.h_in * op.w_in;
            for g in 0..op.groups {
                Self::matmul(
                    (m, k, n),
                    true,
                    filters_tr[g * m * k..].as_ptr(),
                    [k, 1],
                    buf[g * k * n..].as_ptr(),
                    [n, 1],
                    grad_img[g * m * n..].as_mut_ptr(),
                    [n, 1],
                );
            }
        }

        {
            // weight_g^T += img * unfold(patches)^T
            // (G, C/G, O/G * K * K) += (G, C/G, H * W) * (G, H * W, O/G * K * K)
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
                    grad_filters_tr[g * m * n..].as_mut_ptr(),
                    [n, 1],
                );
            }
        }
        Ok(())
    }
}

impl<E: Dtype> Conv2DKernel<E> for Cpu
where
    Self: MatMulImpl<E>,
{
    fn alloc<S: Shape>(&self, s: S) -> Result<Tensor<S, E, Self>, Self::Err> {
        self.try_zeros_like(&s)
    }

    fn forward<L: Shape, R: Shape, O: Shape>(
        &self,
        op: Conv2DOp,
        lhs: &Tensor<L, E, Self>,
        rhs: &Tensor<R, E, Self>,
        out: &mut Tensor<O, E, Self>,
    ) -> Result<(), Self::Err> {
        let patches = (op.chan_in, op.kernel, op.kernel, op.h_out, op.w_out);
        let mut patches = self.try_alloc_zeros::<E>(patches.num_elements())?;
        let [lstride, ostride] = match L::NUM_DIMS {
            3 => [0; 2],
            4 => [lhs.strides[0], out.strides[0]],
            _ => unreachable!(),
        };
        let lhs = lhs.data.as_ref();
        let rhs = rhs.data.as_ref();
        let out = Arc::make_mut(&mut out.data);
        for i_batch in 0..op.batch {
            self.fwd(
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
        lhs: &Tensor<L, E, Self>,
        grad_lhs: &mut Self::Vec,
        rhs: &Tensor<R, E, Self>,
        grad_rhs: &mut Self::Vec,
        out: &impl Tensorlike<O, E, Self>,
        grad_out: &Self::Vec,
    ) -> Result<(), Self::Err> {
        let f_tr_shape = [
            op.groups,
            op.chan_in / op.groups,
            op.chan_out / op.groups,
            op.kernel,
            op.kernel,
        ];
        let patches_shape = [op.chan_out, op.kernel, op.kernel, op.h_in, op.w_in];
        let mut patches = self.try_alloc_zeros::<E>(patches_shape.num_elements())?;
        let mut f1023 = self.try_alloc_zeros::<E>(f_tr_shape.num_elements())?;
        let mut grad_f1023 = self.try_alloc_zeros::<E>(f_tr_shape.num_elements())?;

        {
            // transpose filters in f1023
            let buf = rhs.data.as_ref();
            let mut f_idx = NdIndex::new(f_tr_shape, f_tr_shape.strides());
            while let Some((i, [g, c_over_g, o_over_g, k1, k2])) = f_idx.next_with_idx() {
                let idx = (g * (op.chan_out / op.groups) + o_over_g) * rhs.strides[0]
                    + c_over_g * rhs.strides[1]
                    + k1 * rhs.strides[2]
                    + k2 * rhs.strides[3];
                f1023[i] = buf[idx];
            }
        }

        let [lstride, ostride] = match L::NUM_DIMS {
            3 => [0; 2],
            4 => [lhs.strides[0], out.strides()[0]],
            _ => unreachable!(),
        };
        let lhs = lhs.data.as_ref();

        for i_batch in 0..op.batch {
            self.bwd(
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
            while let Some((i, [g, c_over_g, o_over_g, k1, k2])) = f_idx.next_with_idx() {
                let idx = (g * (op.chan_out / op.groups) + o_over_g) * rhs.strides[0]
                    + c_over_g * rhs.strides[1]
                    + k1 * rhs.strides[2]
                    + k2 * rhs.strides[3];
                grad_rhs[idx] += grad_f1023[i];
            }
        }

        Ok(())
    }
}
