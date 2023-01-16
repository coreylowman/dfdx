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
const ALL_FN_NAMES: [&str; 1] = [UNFOLD_INPUT_FN];

#[repr(C)]
struct ConvParams {
    stride: usize,
    padding: usize,
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

        let out_height = (H + 2 * P - K) / S + 1;
        let out_width = (W + 2 * P - K) / S + 1;

        let patches_shape = (C, K, K, out_height, out_width);
        let patches_strides = patches_shape.strides();
        let patches_numel = patches_shape.num_elements();

        let mut patches = self.dev.alloc_zeros_async::<f32>(patches_numel)?;
        let unfold_fn = self.dev.get_func(MODULE_NAME, UNFOLD_INPUT_FN).unwrap();

        let lhs_dims = self.dev.take_async(lhs.shape.concrete().into())?;
        let lhs_strides = self.dev.take_async(lhs.strides.into())?;

        let patches_dims = self.dev.take_async(patches_shape.concrete().into())?;
        let patches_strides = self.dev.take_async(patches_strides.into())?;

        let cfg = LaunchConfig::for_num_elems(patches.len() as u32);
        let params = (
            ConvParams {
                stride: S,
                padding: P,
            },
            lhs.data.as_ref(),
            &lhs_dims,
            &lhs_strides,
            &mut patches,
            patches_numel,
            &patches_dims,
            &patches_strides,
        );
        unsafe { unfold_fn.launch_async(cfg, params) }?;

        let shape = (Const, Const, Const);
        let strides = shape.strides();
        let mut storage = self.dev.alloc_zeros_async::<f32>(shape.num_elements())?;

        let m = O;
        let k = C * K * K;
        let n = out_width * out_height;
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
