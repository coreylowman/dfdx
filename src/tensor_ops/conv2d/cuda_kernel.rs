use cudarc::driver::{AsKernelParam, LaunchAsync, LaunchConfig};

use crate::tensor_ops::matmul::cuda_kernel::sgemm_batch;
use crate::{shapes::*, tensor::cuda::Cuda};

use std::sync::Arc;

const MODULE_NAME: &str = "conv2d";
const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/conv2d.ptx"));
const UNFOLD_INPUT_FN: &str = "unfold_input_into_patches";
const UNFOLD_OUTPUT_FN: &str = "unfold_output_into_patches";
const BR_TR_FILTERS_FN: &str = "transpose_and_broadcast_filters";
const COLLECT_GRADS_FN: &str = "sum_transposed_filters";
const ALL_FN_NAMES: [&str; 4] = [
    UNFOLD_INPUT_FN,
    UNFOLD_OUTPUT_FN,
    BR_TR_FILTERS_FN,
    COLLECT_GRADS_FN,
];

unsafe impl AsKernelParam for super::Conv2DOp {}

impl super::Conv2DKernel<f32> for Cuda {
    fn forward<L: Shape, R: Shape, O: Shape>(
        &self,
        op: super::Conv2DOp,
        lhs: &Self::Storage<L, f32>,
        rhs: &Self::Storage<R, f32>,
        out: &mut Self::Storage<O, f32>,
    ) -> Result<(), Self::Err> {
        assert_eq!(
            lhs.shape().strides(),
            lhs.strides,
            "Only works with contiguous image strides"
        );

        if !self.dev.has_func(MODULE_NAME, ALL_FN_NAMES[0]) {
            self.dev
                .load_ptx(PTX_SRC.into(), MODULE_NAME, &ALL_FN_NAMES)?;
        }

        let patches_numel = op.batch * op.chan_in * op.kernel * op.kernel * op.h_out * op.w_out;
        let mut patches = self.dev.alloc_zeros_async::<f32>(patches_numel)?;

        let lhs_strides = {
            if L::NUM_DIMS == 3 {
                self.dev
                    .take_async([0, lhs.strides[0], lhs.strides[1], lhs.strides[2]].into())
            } else {
                debug_assert_eq!(L::NUM_DIMS, 4);
                self.dev.take_async(lhs.strides.into())
            }
        }?;

        let unfold_fn = self.dev.get_func(MODULE_NAME, UNFOLD_INPUT_FN).unwrap();
        let cfg = LaunchConfig::for_num_elems(patches.len() as u32);
        let params = (
            op,
            lhs.data.as_ref(),
            &lhs_strides,
            &mut patches,
            patches_numel,
        );
        unsafe { unfold_fn.launch_async(cfg, params) }?;

        // (B, C * K * K, W_OUT * H_OUT)
        // (O, C * K * K)
        let m = op.chan_out;
        let k = op.chan_in * op.kernel * op.kernel;
        let n = op.h_out * op.w_out;
        unsafe {
            sgemm_batch(
                self.blas.as_ref(),
                (op.batch, m, k, n),
                rhs.data.as_ref(),
                [0, k, 1],
                &patches,
                [k * n, n, 1],
                0.0,
                Arc::make_mut(&mut out.data),
                [m * n, n, 1],
            )?;
        }

        Ok(())
    }

    fn backward<L: Shape, R: Shape, O: Shape>(
        &self,
        op: super::Conv2DOp,
        lhs: &Self::Storage<L, f32>,
        gl: &mut Self::Storage<L, f32>,
        rhs: &Self::Storage<R, f32>,
        grad_rhs: &mut Self::Storage<R, f32>,
        go: &Self::Storage<O, f32>,
    ) -> Result<(), Self::Err> {
        let patches_numel = op.batch * op.chan_out * op.kernel * op.kernel * op.h_in * op.w_in;
        let mut patches = self.dev.alloc_zeros_async::<f32>(patches_numel)?;

        {
            // unfold grad_out into patches
            let grad_out_strides = {
                if O::NUM_DIMS == 3 {
                    let strides = [0, go.strides[0], go.strides[1], go.strides[2]];
                    self.dev.take_async(strides.into())
                } else {
                    debug_assert_eq!(O::NUM_DIMS, 4);
                    self.dev.take_async(go.strides.into())
                }
            }?;

            let unfold_fn = self.dev.get_func(MODULE_NAME, UNFOLD_OUTPUT_FN).unwrap();
            let cfg = LaunchConfig::for_num_elems(patches_numel as u32);
            let params = (
                op,
                go.data.as_ref(),
                &grad_out_strides,
                &mut patches,
                patches_numel,
            );
            unsafe { unfold_fn.launch_async(cfg, params) }?;
        }

        let filters_numel = op.batch * op.chan_in * op.chan_out * op.kernel * op.kernel;
        let mut f_b1023 = self.dev.alloc_zeros_async::<f32>(filters_numel)?;
        let mut grad_f_b1023 = self.dev.alloc_zeros_async::<f32>(filters_numel)?;

        {
            // prepare filters for backward operations by
            // swapping dims 0 and 1 and adding a batch dimension
            let tr_fn = self.dev.get_func(MODULE_NAME, BR_TR_FILTERS_FN).unwrap();
            let cfg = LaunchConfig::for_num_elems(rhs.shape.num_elements() as u32);
            let params = (op, rhs.data.as_ref(), &mut f_b1023);
            unsafe { tr_fn.launch_async(cfg, params) }?;
        }

        {
            // img_g += filters * patches
            // (B, C, H * W) += (B, C, O * K * K) * (B, O * K * K, H * W)
            let m = op.chan_in;
            let k = op.chan_out * op.kernel * op.kernel;
            let n = op.h_in * op.w_in;
            unsafe {
                sgemm_batch(
                    self.blas.as_ref(),
                    (op.batch, m, k, n),
                    &f_b1023,
                    [m * k, k, 1],
                    &patches,
                    [k * n, n, 1],
                    1.0,
                    Arc::make_mut(&mut gl.data),
                    [m * n, n, 1],
                )
                .unwrap();
            }
        }

        {
            // weight_g += img * patches^T
            // (B, C, O * K * K) += (B, C, H * W) * (B, H * W, O * K * K)
            let m = op.chan_in;
            let k = op.h_in * op.w_in;
            let n = op.chan_out * op.kernel * op.kernel;
            unsafe {
                sgemm_batch(
                    self.blas.as_ref(),
                    (op.batch, m, k, n),
                    lhs.data.as_ref(),
                    [m * k, k, 1],
                    &patches,
                    [k * n, 1, k],
                    1.0,
                    &mut grad_f_b1023,
                    [m * n, n, 1],
                )
                .unwrap();
            }
        }

        {
            // sum all the gradients collected in our broadcasted grad_f
            // into grad_rhs
            let sum_fn = self.dev.get_func(MODULE_NAME, COLLECT_GRADS_FN).unwrap();
            let cfg = LaunchConfig::for_num_elems(rhs.shape.num_elements() as u32);
            let params = (op, &grad_f_b1023, Arc::make_mut(&mut grad_rhs.data));
            unsafe { sum_fn.launch_async(cfg, params) }?;
        }

        Ok(())
    }
}
