#![allow(clippy::needless_range_loop)]

use crate::{
    shapes::{RemoveDimTo, ReplaceDimTo, Shape},
    tensor::cuda::{Cuda, CudaArray},
};
use cudarc::device::{CudaSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

const GATHER_PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/gather.ptx"));
const GATHER_MODULE_NAME: &str = "gather";
const GATHER_FWD_FN_NAME: &str = "gather_forward";
const GATHER_BWD_FN_NAME: &str = "gather_backward";
const GATHER_ALL_FN_NAMES: [&str; 2] = [GATHER_FWD_FN_NAME, GATHER_BWD_FN_NAME];

impl super::ReplaceDimKernel<f32> for Cuda {
    fn forward<Src: Shape, Dst: Shape, Idx: Shape>(
        &self,
        inp: &Self::Storage<Src, f32>,
        idx: &Self::Storage<Idx, usize>,
    ) -> Result<Self::Storage<Dst, f32>, Self::Err>
    where
        Src: ReplaceDimTo<Dst, Idx>,
    {
        if !self.dev.has_func(GATHER_MODULE_NAME, GATHER_FWD_FN_NAME) {
            self.dev.load_ptx(
                GATHER_PTX_SRC.into(),
                GATHER_MODULE_NAME,
                &GATHER_ALL_FN_NAMES,
            )?;
        }

        let dst = inp.shape.replace(idx.shape);
        let numel = dst.num_elements();
        let mut storage = self.dev.alloc_zeros_async::<f32>(numel)?;

        let inp_dims: CudaSlice<usize> = self.dev.take_async(inp.shape.concrete().into())?;
        let idx_dims: CudaSlice<usize> = self.dev.take_async(idx.shape.concrete().into())?;
        let inp_strides: CudaSlice<usize> = self.dev.take_async(inp.strides.into())?;
        let idx_strides: CudaSlice<usize> = self.dev.take_async(idx.strides.into())?;

        let fwd_fn = self
            .dev
            .get_func(GATHER_MODULE_NAME, GATHER_FWD_FN_NAME)
            .unwrap();
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let params = (
            numel,             // const size_t numel,
            inp.data.as_ref(), // const float *inp,
            Src::NUM_DIMS,     // const size_t inp_num_dims,
            &inp_dims,         // const size_t *inp_dims,
            &inp_strides,      // const size_t *inp_strides,
            idx.data.as_ref(), // const float *idx,
            Idx::NUM_DIMS,     // const size_t idx_num_dims,
            &idx_dims,         // const size_t *idx_dims,
            &idx_strides,      // const size_t *idx_strides,
            &mut storage,      // float *out,
            Dst::NUM_DIMS,     // const size_t out_num_dims,
        );
        unsafe { fwd_fn.launch_async(cfg, params) }?;

        Ok(CudaArray {
            data: Arc::new(storage),
            shape: dst,
            strides: dst.strides(),
        })
    }

    fn backward<Src: Shape, Dst: Shape, Idx: Shape>(
        &self,
        grad_inp: &mut Self::Storage<Src, f32>,
        idx: &Self::Storage<Idx, usize>,
        grad_out: &Self::Storage<Dst, f32>,
    ) -> Result<(), Self::Err>
    where
        Src: ReplaceDimTo<Dst, Idx>,
    {
        let bwd_fn = self
            .dev
            .get_func(GATHER_MODULE_NAME, GATHER_BWD_FN_NAME)
            .unwrap();
        let numel = grad_out.data.len();

        let inp_dims: CudaSlice<usize> = self.dev.take_async(grad_inp.shape.concrete().into())?;
        let idx_dims: CudaSlice<usize> = self.dev.take_async(idx.shape.concrete().into())?;
        let inp_strides: CudaSlice<usize> = self.dev.take_async(grad_inp.strides.into())?;
        let idx_strides: CudaSlice<usize> = self.dev.take_async(idx.strides.into())?;

        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let params = (
            numel,                             // const size_t numel,
            Arc::make_mut(&mut grad_inp.data), // float *grad_inp,
            Src::NUM_DIMS,                     // const size_t inp_num_dims,
            &inp_dims,                         // const size_t *inp_dims,
            &inp_strides,                      // const size_t *inp_strides,
            idx.data.as_ref(),                 // const float *idx,
            Idx::NUM_DIMS,                     // const size_t idx_num_dims,
            &idx_dims,                         // const size_t *idx_dims,
            &idx_strides,                      // const size_t *idx_strides,
            grad_out.data.as_ref(),            // const float *grad_out,
            Dst::NUM_DIMS,                     // const size_t out_num_dims,
        );
        unsafe { bwd_fn.launch_async(cfg, params) }?;
        Ok(())
    }
}

const SELECT_PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/select.ptx"));
const SELECT_MODULE_NAME: &str = "select";
const SELECT_FWD_FN_NAME: &str = "select_forward";
const SELECT_BWD_FN_NAME: &str = "select_backward";
const SELECT_ALL_FN_NAMES: [&str; 2] = [SELECT_FWD_FN_NAME, SELECT_BWD_FN_NAME];

impl super::RemoveDimKernel<f32> for Cuda {
    fn forward<Src: Shape, Dst: Shape, Idx: Shape>(
        &self,
        inp: &Self::Storage<Src, f32>,
        idx: &Self::Storage<Idx, usize>,
    ) -> Result<Self::Storage<Dst, f32>, Self::Err>
    where
        Src: RemoveDimTo<Dst, Idx>,
    {
        if !self.dev.has_func(SELECT_MODULE_NAME, SELECT_FWD_FN_NAME) {
            self.dev.load_ptx(
                SELECT_PTX_SRC.into(),
                SELECT_MODULE_NAME,
                &SELECT_ALL_FN_NAMES,
            )?;
        }

        let dst = inp.shape.remove(idx.shape);
        let numel = dst.num_elements();
        let mut storage = self.dev.alloc_zeros_async::<f32>(numel)?;

        let inp_dims: CudaSlice<usize> = self.dev.take_async(inp.shape.concrete().into())?;
        let idx_dims: CudaSlice<usize> = self.dev.take_async(idx.shape.concrete().into())?;
        let dst_dims: CudaSlice<usize> = self.dev.take_async(dst.concrete().into())?;
        let inp_strides: CudaSlice<usize> = self.dev.take_async(inp.strides.into())?;
        let idx_strides: CudaSlice<usize> = self.dev.take_async(idx.strides.into())?;
        let dst_strides: CudaSlice<usize> = self.dev.take_async(dst.strides().into())?;

        let fwd_fn = self
            .dev
            .get_func(SELECT_MODULE_NAME, SELECT_FWD_FN_NAME)
            .unwrap();
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let params = (
            numel,             // const size_t numel,
            inp.data.as_ref(), // const float *inp,
            Src::NUM_DIMS,     // const size_t inp_num_dims,
            &inp_dims,         // const size_t *inp_dims,
            &inp_strides,      // const size_t *inp_strides,
            idx.data.as_ref(), // const float *idx,
            Idx::NUM_DIMS,     // const size_t idx_num_dims,
            &idx_dims,         // const size_t *idx_dims,
            &idx_strides,      // const size_t *idx_strides,
            &mut storage,      // float *out,
            &dst_dims,         // const size_t *out_dims,
            &dst_strides,      // const size_t *out_strides
        );
        unsafe { fwd_fn.launch_async(cfg, params) }?;

        Ok(CudaArray {
            data: Arc::new(storage),
            shape: dst,
            strides: dst.strides(),
        })
    }

    fn backward<Src: Shape, Dst: Shape, Idx: Shape>(
        &self,
        grad_inp: &mut Self::Storage<Src, f32>,
        idx: &Self::Storage<Idx, usize>,
        grad_out: &Self::Storage<Dst, f32>,
    ) -> Result<(), Self::Err>
    where
        Src: RemoveDimTo<Dst, Idx>,
    {
        let bwd_fn = self
            .dev
            .get_func(SELECT_MODULE_NAME, SELECT_BWD_FN_NAME)
            .unwrap();
        let numel = grad_out.data.len();

        let inp_dims: CudaSlice<usize> = self.dev.take_async(grad_inp.shape.concrete().into())?;
        let idx_dims: CudaSlice<usize> = self.dev.take_async(idx.shape.concrete().into())?;
        let out_dims: CudaSlice<usize> = self.dev.take_async(grad_out.shape.concrete().into())?;
        let inp_strides: CudaSlice<usize> = self.dev.take_async(grad_inp.strides.into())?;
        let idx_strides: CudaSlice<usize> = self.dev.take_async(idx.strides.into())?;
        let out_strides: CudaSlice<usize> = self.dev.take_async(grad_out.strides.into())?;

        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let params = (
            numel,                             // const size_t numel,
            Arc::make_mut(&mut grad_inp.data), // float *grad_inp,
            Src::NUM_DIMS,                     // const size_t inp_num_dims,
            &inp_dims,                         // const size_t *inp_dims,
            &inp_strides,                      // const size_t *inp_strides,
            idx.data.as_ref(),                 // const float *idx,
            Idx::NUM_DIMS,                     // const size_t idx_num_dims,
            &idx_dims,                         // const size_t *idx_dims,
            &idx_strides,                      // const size_t *idx_strides,
            grad_out.data.as_ref(),            // const float *grad_out,
            &out_dims,                         // const size_t *out_dims,
            &out_strides,                      // const size_t *out_strides
        );
        unsafe { bwd_fn.launch_async(cfg, params) }?;
        Ok(())
    }
}
