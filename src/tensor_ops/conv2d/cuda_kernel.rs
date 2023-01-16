use cudarc::driver::{AsKernelParam, LaunchAsync, LaunchConfig};

use crate::tensor_ops::matmul::cuda_kernel::sgemm;
use crate::{
    shapes::*,
    tensor::cuda::{Cuda, CudaArray},
};

use std::sync::Arc;

const MODULE_NAME: &str = "conv2d";
const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/conv2d.ptx"));
const UNFOLD_INPUT_FN: &str = "unfold_input_into_patches";
const UNFOLD_OUTPUT_FN: &str = "unfold_output_into_patches";
const TRANSPOSE_FILTERS_FN: &str = "transpose_filters";
const SUM_TRANSPOSED_FILTERS_FN: &str = "sum_transposed_filters";
const ALL_FN_NAMES: [&str; 4] = [
    UNFOLD_INPUT_FN,
    UNFOLD_OUTPUT_FN,
    TRANSPOSE_FILTERS_FN,
    SUM_TRANSPOSED_FILTERS_FN,
];

#[repr(C)]
struct ConvParams {
    channels_in: usize,
    height_in: usize,
    width_in: usize,
    stride: usize,
    padding: usize,
    kernel: usize,
    channels_out: usize,
    height_out: usize,
    width_out: usize,
}

unsafe impl AsKernelParam for ConvParams {}

impl<const K: usize, const S: usize, const P: usize, const C: usize, const O: usize>
    super::Conv2DKernel<f32, C, O, K, S, P> for Cuda
{
    fn forward<const H: usize, const W: usize>(
        &self,
        lhs: &Self::Storage<Rank3<C, H, W>, f32>,
        rhs: &Self::Storage<Rank4<O, C, K, K>, f32>,
    ) -> Result<
        Self::Storage<Rank3<O, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }>, f32>,
        Self::Err,
    > {
        if !self.dev.has_func(MODULE_NAME, ALL_FN_NAMES[0]) {
            self.dev
                .load_ptx(PTX_SRC.into(), MODULE_NAME, &ALL_FN_NAMES)?;
        }

        let height_out = (H + 2 * P - K) / S + 1;
        let width_out = (W + 2 * P - K) / S + 1;
        let patches_numel = C * K * K * height_out * width_out;
        let mut patches = self.dev.alloc_zeros_async::<f32>(patches_numel)?;

        let lhs_strides = self.dev.take_async(lhs.strides.into())?;

        let unfold_fn = self.dev.get_func(MODULE_NAME, UNFOLD_INPUT_FN).unwrap();
        let cfg = LaunchConfig::for_num_elems(patches.len() as u32);
        let params = (
            ConvParams {
                channels_in: C,
                height_in: H,
                width_in: W,
                stride: S,
                padding: P,
                kernel: K,
                channels_out: O,
                height_out,
                width_out,
            },
            lhs.data.as_ref(),
            &lhs_strides,
            &mut patches,
            patches_numel,
        );
        unsafe { unfold_fn.launch_async(cfg, params) }?;

        let shape = (Const, Const, Const);
        let strides = shape.strides();
        let mut storage = self.dev.alloc_zeros_async::<f32>(shape.num_elements())?;

        let m = O;
        let k = C * K * K;
        let n = width_out * height_out;
        unsafe {
            sgemm(
                self.blas.as_ref(),
                (m, k, n),
                rhs.data.as_ref(),
                [k, 1],
                &patches,
                [n, 1],
                0.0,
                &mut storage,
                [n, 1],
            )?;
        }

        Ok(CudaArray {
            data: Arc::new(storage),
            shape,
            strides,
        })
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
        let height_out = (H + 2 * P - K) / S + 1;
        let width_out = (W + 2 * P - K) / S + 1;
        let patches_numel = O * K * K * H * W;
        let mut patches = self.dev.alloc_zeros_async::<f32>(patches_numel)?;
        let grad_out_strides = self.dev.take_async(grad_out.strides.into())?;

        {
            let unfold_fn = self.dev.get_func(MODULE_NAME, UNFOLD_OUTPUT_FN).unwrap();
            let cfg = LaunchConfig::for_num_elems(patches.len() as u32);
            let params = (
                ConvParams {
                    channels_in: C,
                    height_in: H,
                    width_in: W,
                    stride: S,
                    padding: P,
                    kernel: K,
                    channels_out: O,
                    height_out,
                    width_out,
                },
                grad_out.data.as_ref(),
                &grad_out_strides,
                &mut patches,
                patches_numel,
            );
            unsafe { unfold_fn.launch_async(cfg, params) }?;
        }

        {
            todo!("call transpose_filters");
        }

        {
            // img_g += filters^T * unfold(grad_out)
            todo!("call sgemm");
        }

        {
            // weight_g^T += img * patches^T
            todo!("allocate zeros for grad_rhs and call sgemm");
        }

        {
            todo!("call sum_transposed_filters to add transposed filters to grad_rhs")
        }
        Ok(())
    }
}

impl<const K: usize, const S: usize, const P: usize, const C: usize, const O: usize>
    super::Conv2DBatchedKernel<f32, C, O, K, S, P> for Cuda
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
        todo!()
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
        todo!()
    }
}
