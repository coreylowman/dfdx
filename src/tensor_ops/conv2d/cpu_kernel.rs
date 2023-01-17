use crate::shapes::Shape;
use crate::tensor::cpu::*;
use crate::tensor_ops::matmul::cpu_kernel::matmul;

use super::{Conv2DKernel, ConvParams};

use std::sync::Arc;

impl Cpu {
    #[inline]
    fn conv2d_forward<P: Shape<Concrete = [usize; 5]>>(
        &self,
        op: &ConvParams,
        img: &[f32],
        filters: &[f32],
        out: &mut [f32],
        inp_patches_buf: &mut StridedArray<P, f32>,
    ) -> Result<(), CpuError> {
        let mut patch_iter = inp_patches_buf.iter_mut_with_index();
        while let Some((p, [c, k1, k2, oh, ow])) = patch_iter.next() {
            let y = (oh * op.stride + k1).wrapping_sub(op.padding);
            let x = (ow * op.stride + k2).wrapping_sub(op.padding);
            if y < op.height_in && x < op.width_in {
                *p = img[c * (op.width_in * op.height_in) + y * op.width_in + x];
            }
        }

        // (O, C * K * K) * (C * K * K, OH * OW) = (O, OH * OW)
        let m = op.channels_out;
        let k = op.channels_in * op.kernel_size * op.kernel_size;
        let n = op.width_out * op.height_out;
        matmul(
            View::new(filters, (m, k)),
            View::new(inp_patches_buf.view().data, (k, n)),
            &mut ViewMut::new(out, (m, n)),
        );
        Ok(())
    }

    #[inline]
    fn conv2d_backward<P: Shape<Concrete = [usize; 5]>>(
        &self,
        op: &ConvParams,
        img: &[f32],
        grad_img: &mut [f32],
        filters_tr: &[f32],
        grad_filters_tr: &mut [f32],
        grad_out: &[f32],
        out_patches_buf: &mut StridedArray<P, f32>,
    ) -> Result<(), CpuError> {
        {
            let out_patches_buf = out_patches_buf.view_mut();
            for o in 0..op.channels_out {
                for oh in 0..op.height_out {
                    for ow in 0..op.width_out {
                        let g =
                            grad_out[o * (op.height_out * op.width_out) + oh * op.width_out + ow];
                        for k1 in 0..op.kernel_size {
                            for k2 in 0..op.kernel_size {
                                let y = (oh * op.stride + k1).wrapping_sub(op.padding);
                                let x = (ow * op.stride + k2).wrapping_sub(op.padding);
                                if y < op.height_in && x < op.width_in {
                                    out_patches_buf.data[o * out_patches_buf.strides[0]
                                        + k1 * out_patches_buf.strides[1]
                                        + k2 * out_patches_buf.strides[2]
                                        + y * out_patches_buf.strides[3]
                                        + x * out_patches_buf.strides[4]] = g;
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
            let m = op.channels_in;
            let k = op.channels_out * op.kernel_size * op.kernel_size;
            let n = op.height_in * op.width_in;
            matmul(
                View::new(filters_tr, (m, k)),
                View::new(out_patches_buf.view().data, (k, n)),
                &mut ViewMut::new(grad_img, (m, n)),
            );
        }

        {
            // weight_g^T += img * patches^T
            // (C, O * K * K) += (C, H * W) * (H * W, O * K * K)
            let m = op.channels_in;
            let k = op.height_in * op.width_in;
            let n = op.channels_out * op.kernel_size * op.kernel_size;
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
    fn forward<L: crate::shapes::Shape, R: crate::shapes::Shape, O: crate::shapes::Shape>(
        &self,
        op: super::ConvParams,
        lhs: &Self::Storage<L, f32>,
        rhs: &Self::Storage<R, f32>,
        out: &mut Self::Storage<O, f32>,
    ) -> Result<(), Self::Err> {
        let mut patches: StridedArray<_, f32> = StridedArray::new((
            op.channels_in,
            op.kernel_size,
            op.kernel_size,
            op.height_out,
            op.width_out,
        ))?;
        let [lstride, ostride] = if L::NUM_DIMS == 3 {
            [0; 2]
        } else {
            debug_assert_eq!(L::NUM_DIMS, 4);
            [lhs.strides[0], out.strides[0]]
        };
        let lhs = lhs.data.as_ref();
        let rhs = rhs.data.as_ref();
        let out = Arc::make_mut(&mut out.data);
        for i_batch in 0..op.batch_size {
            self.conv2d_forward(
                &op,
                &lhs[i_batch * lstride..],
                &rhs,
                &mut out[i_batch * ostride..],
                &mut patches,
            )?;
        }
        Ok(())
    }

    fn backward<L: crate::shapes::Shape, R: crate::shapes::Shape, O: crate::shapes::Shape>(
        &self,
        op: super::ConvParams,
        lhs: &Self::Storage<L, f32>,
        grad_lhs: &mut Self::Storage<L, f32>,
        rhs: &Self::Storage<R, f32>,
        grad_rhs: &mut Self::Storage<R, f32>,
        grad_out: &Self::Storage<O, f32>,
    ) -> Result<(), Self::Err> {
        let mut patches: StridedArray<_, f32> = StridedArray::new((
            op.channels_out,
            op.kernel_size,
            op.kernel_size,
            op.height_in,
            op.width_in,
        ))?;

        let mut f1023: StridedArray<_, f32> = StridedArray::new((
            op.channels_in,
            op.channels_out,
            op.kernel_size,
            op.kernel_size,
        ))?;
        let mut grad_f1023: StridedArray<_, f32> = StridedArray::new((
            op.channels_in,
            op.channels_out,
            op.kernel_size,
            op.kernel_size,
        ))?;

        {
            // transpose filters in f1023
            let rhs_buf = rhs.data.as_ref();
            let mut f_iter = f1023.iter_mut_with_index();

            while let Some((f, [c, o, k1, k2])) = f_iter.next() {
                *f = rhs_buf[o * rhs.strides[0]
                    + c * rhs.strides[1]
                    + k1 * rhs.strides[2]
                    + k2 * rhs.strides[3]];
            }
        }

        let [lstride, ostride] = if L::NUM_DIMS == 3 {
            [0; 2]
        } else {
            debug_assert_eq!(L::NUM_DIMS, 4);
            [lhs.strides[0], grad_out.strides[0]]
        };
        let lhs = lhs.data.as_ref();
        let grad_lhs = Arc::make_mut(&mut grad_lhs.data);
        let f = f1023.data.as_ref();
        let grad_f = Arc::make_mut(&mut grad_f1023.data);
        let grad_out = grad_out.data.as_ref();

        for i_batch in 0..op.batch_size {
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
            let grad_rhs_buf = Arc::make_mut(&mut grad_rhs.data);
            let mut f_iter = grad_f1023.iter_with_index();

            while let Some((f, [c, o, k1, k2])) = f_iter.next() {
                grad_rhs_buf[o * rhs.strides[0]
                    + c * rhs.strides[1]
                    + k1 * rhs.strides[2]
                    + k2 * rhs.strides[3]] = *f;
            }
        }

        Ok(())
    }
}
