use super::ReplaceWhereKernel;
use crate::{
    shapes::Shape,
    tensor::cuda::{Cuda, CudaArray},
};
use cudarc::driver::{CudaSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/replace_where.ptx"));
const MODULE_NAME: &str = "replace_where";
const FWD_FN_NAME: &str = "replace_where_forward";
const BWD_FN_NAME: &str = "replace_where_backward";
const ALL_FN_NAMES: [&str; 2] = [FWD_FN_NAME, BWD_FN_NAME];

impl ReplaceWhereKernel<f32> for Cuda {
    fn forward<S: Shape>(
        &self,
        lhs: &Self::Storage<S, f32>,
        cond: &Self::Storage<S, bool>,
        rhs: &Self::Storage<S, f32>,
    ) -> Result<Self::Storage<S, f32>, Self::Err> {
        if !self.dev.has_func(MODULE_NAME, FWD_FN_NAME) {
            self.dev
                .load_ptx(PTX_SRC.into(), MODULE_NAME, &ALL_FN_NAMES)?;
        }

        let shape = lhs.shape;
        let strides = lhs.shape.strides();
        let numel = shape.num_elements();

        let mut storage = self.dev.alloc_zeros_async::<f32>(numel)?;

        let dims: CudaSlice<usize> = self.dev.take_async(shape.concrete().into())?;
        let cond_strides: CudaSlice<usize> = self.dev.take_async(cond.strides.into())?;
        let lhs_strides: CudaSlice<usize> = self.dev.take_async(lhs.strides.into())?;
        let rhs_strides: CudaSlice<usize> = self.dev.take_async(rhs.strides.into())?;

        let fwd_fn = self.dev.get_func(MODULE_NAME, FWD_FN_NAME).unwrap();
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let params = (
            numel,              // const size_t numel,
            S::NUM_DIMS,        // const size_t num_dims,
            &dims,              // const size_t *dims,
            cond.data.as_ref(), // const float *cond,
            &cond_strides,      // const size_t *cond_strides,
            lhs.data.as_ref(),  // const float *lhs,
            &lhs_strides,       // const size_t *lhs_strides,
            rhs.data.as_ref(),  // const float *rhs,
            &rhs_strides,       // const size_t *rhs_strides,
            &mut storage,       // float *out,
        );
        unsafe { fwd_fn.launch_async(cfg, params) }?;
        Ok(CudaArray {
            data: Arc::new(storage),
            shape,
            strides,
        })
    }

    fn backward<S: Shape>(
        &self,
        grad_lhs: &mut Self::Storage<S, f32>,
        cond: &Self::Storage<S, bool>,
        grad_rhs: &mut Self::Storage<S, f32>,
        grad_out: &Self::Storage<S, f32>,
    ) -> Result<(), Self::Err> {
        let bwd_fn = self.dev.get_func(MODULE_NAME, BWD_FN_NAME).unwrap();
        let numel = cond.shape.num_elements();

        let dims: CudaSlice<usize> = self.dev.take_async(cond.shape.concrete().into())?;
        let lhs_strides: CudaSlice<usize> = self.dev.take_async(grad_lhs.strides.into())?;
        let cond_strides: CudaSlice<usize> = self.dev.take_async(cond.strides.into())?;
        let rhs_strides: CudaSlice<usize> = self.dev.take_async(grad_rhs.strides.into())?;

        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let params = (
            numel,                             // const size_t numel,
            S::NUM_DIMS,                       // const size_t num_dims,
            &dims,                             // const size_t *dims,
            cond.data.as_ref(),                // bool *cond,
            &cond_strides,                     // const size_t *cond_strides,
            Arc::make_mut(&mut grad_lhs.data), // float *grad_lhs,
            &lhs_strides,                      // const size_t *lhs_strides,
            Arc::make_mut(&mut grad_rhs.data), // float *grad_rhs,
            &rhs_strides,                      // const size_t *rhs_strides,
            grad_out.data.as_ref(),            // const float *grad_out,
        );
        unsafe { bwd_fn.launch_async(cfg, params) }?;
        Ok(())
    }
}
