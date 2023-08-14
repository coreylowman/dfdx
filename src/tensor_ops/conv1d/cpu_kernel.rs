use crate::shapes::{Dtype, Shape};
use crate::tensor::{cpu::*, *};
use crate::tensor_ops::matmul::cpu_kernel::MatMulImpl;

use super::{Conv1DKernel, Conv1DOp};

use std::sync::Arc;

impl Conv1DOp {
    #[inline(always)]
    fn unfold_idx(&self, [k1, y]: [usize; 2]) -> Option<[usize; 1]> {
        let mut ol = y + self.padding;
        if ol < self.dilation * k1 {
            return None;
        }
        ol -= self.dilation * k1;
        if ol % self.stride != 0 {
            return None;
        }
        ol /= self.stride;
        if ol >= self.l_out {
            return None;
        }

        Some([ol])
    }
}

impl Cpu {
    #[inline]
    fn fwd_conv1d<E: Dtype>(
        &self,
        op: &Conv1DOp,
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
            for c in 0..(op.groups * op.chan_in) {
                for k1 in 0..op.kernel {
                    for ol in 0..op.l_out {
                        let y = (ol * op.stride + op.dilation * k1).wrapping_sub(op.padding);
                        if y < op.l_in {
                            buf[i] = img[c * op.l_in + y];
                        }
                        i += 1;
                    }
                }
            }
        }

        // (G, O / G, C * K) * (G, C * K * K, OL) = (G, O / G, OL)
        let m = op.chan_out / op.groups;
        let k = op.chan_in * op.kernel;

        let n = op.l_out;
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
    fn bwd_conv1d<E: Dtype>(
        &self,
        op: &Conv1DOp,
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
                    for y in 0..op.l_in {
                        if let Some([ol]) = op.unfold_idx([k1, y]) {
                            buf[i] = grad_out[o * op.l_out + ol];
                        }
                        i += 1;
                    }
                }
            }
        }

        {
            // img_g += filters^T * unfold(grad_out)
            // (G, C, H * W) += (G, C, O/G * K * K) * (G, O/G * K * K, L)
            let m = op.chan_in;
            let k = (op.chan_out / op.groups) * op.kernel;
            let n = op.l_in;
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
            // (G, C, O/G * K * K) += (G, C, H * W) * (G, H * W, O/G * K)
            let m = op.chan_in;
            let k = op.l_in;
            let n = (op.chan_out / op.groups) * op.kernel;
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

impl<E: Dtype> Conv1DKernel<E> for Cpu
where
    Self: MatMulImpl<E>,
{
    fn alloc<S: Shape>(&self, s: S) -> Result<Tensor<S, E, Self>, Self::Err> {
        self.try_zeros_like(&s)
    }

    fn forward<L: Shape, R: Shape, O: Shape>(
        &self,
        op: Conv1DOp,
        lhs: &Tensor<L, E, Self>,
        rhs: &Tensor<R, E, Self>,
        out: &mut Tensor<O, E, Self>,
    ) -> Result<(), Self::Err> {
        let patches = (op.groups * op.chan_in, op.kernel, op.kernel, op.l_out);
        let mut patches = self.try_alloc_zeros::<E>(patches.num_elements())?;
        let [lstride, ostride] = match L::NUM_DIMS {
            2 => [0; 2],
            3 => [lhs.strides[0], out.strides[0]],
            _ => unreachable!(),
        };
        let lhs = lhs.data.as_ref();
        let rhs = rhs.data.as_ref();
        let out = Arc::make_mut(&mut out.data);
        for i_batch in 0..op.batch {
            self.fwd_conv1d(
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
        op: Conv1DOp,
        lhs: &Tensor<L, E, Self>,
        grad_lhs: &mut Self::Vec,
        rhs: &Tensor<R, E, Self>,
        grad_rhs: &mut Self::Vec,
        out: &impl Tensorlike<O, E, Self>,
        grad_out: &Self::Vec,
    ) -> Result<(), Self::Err> {
        let f_tr_shape = [op.groups, op.chan_in, op.chan_out / op.groups, op.kernel];
        let patches_shape = [op.chan_out, op.kernel, op.kernel, op.l_in];
        let mut patches = self.try_alloc_zeros::<E>(patches_shape.num_elements())?;
        let mut f1023 = self.try_alloc_zeros::<E>(f_tr_shape.num_elements())?;
        let mut grad_f1023 = self.try_alloc_zeros::<E>(f_tr_shape.num_elements())?;

        {
            // transpose filters in f1023
            let buf = rhs.data.as_ref();
            let mut f_idx = NdIndex::new(f_tr_shape, f_tr_shape.strides());
            while let Some((i, [g, c, o, k1])) = f_idx.next_with_idx() {
                let idx = (g * (op.chan_out / op.groups) + o) * rhs.strides[0]
                    + c * rhs.strides[1]
                    + k1 * rhs.strides[2];
                f1023[i] = buf[idx];
            }
        }

        let [lstride, ostride] = match L::NUM_DIMS {
            2 => [0; 2],
            3 => [lhs.strides[0], out.strides()[0]],
            _ => unreachable!(),
        };
        let lhs = lhs.data.as_ref();

        for i_batch in 0..op.batch {
            self.bwd_conv1d(
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
            while let Some((i, [g, c, o, k1])) = f_idx.next_with_idx() {
                let idx = (g * (op.chan_out / op.groups) + o) * rhs.strides[0]
                    + c * rhs.strides[1]
                    + k1 * rhs.strides[2];
                grad_rhs[idx] += grad_f1023[i];
            }
        }

        Ok(())
    }
}
