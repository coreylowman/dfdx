use crate::shapes::{Dyn, Shape};
use crate::tensor::cpu::*;
use crate::tensor_ops::matmul::cpu_kernel::matmul;

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
    fn conv2d_forward<P: Shape<Concrete = [usize; 5]>>(
        &self,
        op: &Conv2DOp,
        img: &[f32],
        filters: &[f32],
        out: &mut [f32],
        inp_patches_buf: &mut StridedArray<P, f32>,
    ) -> Result<(), CpuError> {
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
        let m = Dyn::<'O'>(op.chan_out);
        let k = Dyn::<'A'>(op.chan_in * op.kernel * op.kernel);
        let n = Dyn::<'B'>(op.w_out * op.h_out);
        matmul(
            View::new(filters, (m, k)),
            View::new(inp_patches_buf.view().data, (k, n)),
            &mut ViewMut::new(out, (m, n)),
        );
        Ok(())
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn conv2d_backward<P: Shape<Concrete = [usize; 5]>>(
        &self,
        op: &Conv2DOp,
        img: &[f32],
        grad_img: &mut [f32],
        filters_tr: &[f32],
        grad_filters_tr: &mut [f32],
        grad_out: &[f32],
        out_patches_buf: &mut StridedArray<P, f32>,
    ) -> Result<(), CpuError> {
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
            let m = Dyn::<'I'>(op.chan_in);
            let k = Dyn::<'A'>(op.chan_out * op.kernel * op.kernel);
            let n = Dyn::<'B'>(op.w_in * op.h_in);
            matmul(
                View::new(filters_tr, (m, k)),
                View::new(out_patches_buf.view().data, (k, n)),
                &mut ViewMut::new(grad_img, (m, n)),
            );
        }

        {
            // weight_g^T += img * patches^T
            // (C, O * K * K) += (C, H * W) * (H * W, O * K * K)
            let m = Dyn::<'I'>(op.chan_in);
            let k = Dyn::<'A'>(op.h_in * op.w_in);
            let n = Dyn::<'B'>(op.chan_out * op.kernel * op.kernel);
            matmul(
                View::new(img, (m, k)),
                View::new(out_patches_buf.view().data, (n, k)).tr(),
                &mut ViewMut::new(grad_filters_tr, (m, n)),
            );
        }
        Ok(())
    }
}

impl Conv2DKernel<f32> for Cpu {
    fn forward<L: Shape, R: Shape, O: Shape>(
        &self,
        op: Conv2DOp,
        lhs: &Self::Storage<L, f32>,
        rhs: &Self::Storage<R, f32>,
        out: &mut Self::Storage<O, f32>,
    ) -> Result<(), Self::Err> {
        let in_shape = (
            Dyn::<'I'>(op.chan_in),
            Dyn::<'K'>(op.kernel),
            Dyn::<'K'>(op.kernel),
            Dyn::<'H'>(op.h_out),
            Dyn::<'W'>(op.w_out),
        );
        let mut patches: StridedArray<_, f32> = StridedArray::new(in_shape)?;
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
        lhs: &Self::Storage<L, f32>,
        grad_lhs: &mut Self::Storage<L, f32>,
        rhs: &Self::Storage<R, f32>,
        grad_rhs: &mut Self::Storage<R, f32>,
        grad_out: &Self::Storage<O, f32>,
    ) -> Result<(), Self::Err> {
        let out_shape = (
            Dyn::<'O'>(op.chan_out),
            Dyn::<'K'>(op.kernel),
            Dyn::<'K'>(op.kernel),
            Dyn::<'h'>(op.h_in),
            Dyn::<'w'>(op.w_in),
        );
        let mut patches: StridedArray<_, f32> = StridedArray::new(out_shape)?;

        let filter_shape = (
            Dyn::<'I'>(op.chan_in),
            Dyn::<'O'>(op.chan_out),
            Dyn::<'K'>(op.kernel),
            Dyn::<'K'>(op.kernel),
        );
        let mut f1023: StridedArray<_, f32> = StridedArray::new(filter_shape)?;
        let mut grad_f1023: StridedArray<_, f32> = StridedArray::new(filter_shape)?;

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
