use super::BooleanKernel;
use crate::prelude::{cuda::CudaArray, *};
use cudarc::driver::*;

use std::sync::Arc;

const MODULE_NAME: &str = "boolean";
const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/boolean.ptx"));
const ALL_FN_NAMES: [&str; 4] = ["boolean_not", "boolean_and", "boolean_or", "boolean_xor"];

impl Cuda {
    fn call_binary<S: Shape>(
        &self,
        fn_name: &str,
        lhs: &CudaArray<S, bool>,
        rhs: &CudaArray<S, bool>,
    ) -> Result<CudaArray<S, bool>, <Self as HasErr>::Err> {
        if !self.dev.has_func(MODULE_NAME, fn_name) {
            self.dev
                .load_ptx(PTX_SRC.into(), MODULE_NAME, &ALL_FN_NAMES)?;
        }

        let shape = lhs.shape;
        let strides = lhs.shape.strides();
        let numel = shape.num_elements();

        let mut storage = unsafe { self.dev.alloc_async(numel) }?;

        let dims: CudaSlice<usize> = self.dev.take_async(shape.concrete().into())?;
        let lhs_strides: CudaSlice<usize> = self.dev.take_async(lhs.strides.into())?;
        let rhs_strides: CudaSlice<usize> = self.dev.take_async(rhs.strides.into())?;

        let fwd_fn = self.dev.get_func(MODULE_NAME, fn_name).unwrap();
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let params = (
            numel,             // const size_t numel,
            S::NUM_DIMS,       // const size_t num_dims,
            &dims,             // const size_t *dims,
            lhs.data.as_ref(), // const bool *lhs,
            &lhs_strides,      // const size_t *lhs_strides,
            rhs.data.as_ref(), // const bool *rhs,
            &rhs_strides,      // const size_t *rhs_strides,
            &mut storage,      // bool *out,
        );
        unsafe { fwd_fn.launch_async(cfg, params) }?;
        Ok(CudaArray {
            data: Arc::new(storage),
            shape,
            strides,
        })
    }
}

impl BooleanKernel for Cuda {
    fn not<S: Shape>(
        &self,
        inp: &Self::Storage<S, bool>,
    ) -> Result<Self::Storage<S, bool>, Self::Err> {
        if !self.dev.has_func(MODULE_NAME, "boolean_not") {
            self.dev
                .load_ptx(PTX_SRC.into(), MODULE_NAME, &ALL_FN_NAMES)?;
        }

        let numel = inp.data.len();
        let mut storage = unsafe { self.dev.alloc_async(numel) }?;

        let fwd_fn = self.dev.get_func(MODULE_NAME, "boolean_not").unwrap();
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let params = (
            numel,             // const size_t numel,
            inp.data.as_ref(), // const bool *inp,
            &mut storage,      // bool *out
        );
        unsafe { fwd_fn.launch_async(cfg, params) }?;

        Ok(CudaArray {
            data: Arc::new(storage),
            shape: inp.shape,
            strides: inp.strides,
        })
    }

    fn and<S: Shape>(
        &self,
        lhs: &Self::Storage<S, bool>,
        rhs: &Self::Storage<S, bool>,
    ) -> Result<Self::Storage<S, bool>, Self::Err> {
        self.call_binary("boolean_and", lhs, rhs)
    }

    fn or<S: Shape>(
        &self,
        lhs: &Self::Storage<S, bool>,
        rhs: &Self::Storage<S, bool>,
    ) -> Result<Self::Storage<S, bool>, Self::Err> {
        self.call_binary("boolean_or", lhs, rhs)
    }

    fn xor<S: Shape>(
        &self,
        lhs: &Self::Storage<S, bool>,
        rhs: &Self::Storage<S, bool>,
    ) -> Result<Self::Storage<S, bool>, Self::Err> {
        self.call_binary("boolean_xor", lhs, rhs)
    }
}
