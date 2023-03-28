use super::BooleanKernel;
use crate::{
    shapes::Shape,
    tensor::{launch_cfg, Cuda, CudaError, Tensor},
};
use cudarc::driver::*;

const MODULE_NAME: &str = "boolean";
const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/boolean.ptx"));
const ALL_FN_NAMES: [&str; 4] = ["boolean_not", "boolean_and", "boolean_or", "boolean_xor"];

impl Cuda {
    fn call_binary<S: Shape>(
        &self,
        fn_name: &str,
        lhs: &Tensor<S, bool, Self>,
        rhs: &Tensor<S, bool, Self>,
    ) -> Result<Tensor<S, bool, Self>, CudaError> {
        if !self.dev.has_func(MODULE_NAME, fn_name) {
            self.dev
                .load_ptx(PTX_SRC.into(), MODULE_NAME, &ALL_FN_NAMES)?;
        }

        let shape = lhs.shape;
        let strides = lhs.shape.strides();
        let numel = shape.num_elements();

        let mut storage = unsafe { self.dev.alloc(numel) }?;

        let dims: CudaSlice<usize> = self.dev.htod_copy(shape.concrete().into())?;
        let lhs_strides: CudaSlice<usize> = self.dev.htod_copy(lhs.strides.into())?;
        let rhs_strides: CudaSlice<usize> = self.dev.htod_copy(rhs.strides.into())?;

        let fwd_fn = self.dev.get_func(MODULE_NAME, fn_name).unwrap();
        let cfg = launch_cfg(numel as u32);
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
        unsafe { fwd_fn.launch(cfg, params) }?;
        Ok(self.build_tensor(shape, strides, storage))
    }
}

impl BooleanKernel for Cuda {
    fn not<S: Shape>(
        &self,
        inp: &Tensor<S, bool, Self>,
    ) -> Result<Tensor<S, bool, Self>, Self::Err> {
        if !self.dev.has_func(MODULE_NAME, "boolean_not") {
            self.dev
                .load_ptx(PTX_SRC.into(), MODULE_NAME, &ALL_FN_NAMES)?;
        }

        let numel = inp.data.len();
        let mut storage = unsafe { self.dev.alloc(numel) }?;

        let fwd_fn = self.dev.get_func(MODULE_NAME, "boolean_not").unwrap();
        let cfg = launch_cfg(numel as u32);
        let params = (
            numel,             // const size_t numel,
            inp.data.as_ref(), // const bool *inp,
            &mut storage,      // bool *out
        );
        unsafe { fwd_fn.launch(cfg, params) }?;

        Ok(self.build_tensor(inp.shape, inp.strides, storage))
    }

    fn and<S: Shape>(
        &self,
        lhs: &Tensor<S, bool, Self>,
        rhs: &Tensor<S, bool, Self>,
    ) -> Result<Tensor<S, bool, Self>, Self::Err> {
        self.call_binary("boolean_and", lhs, rhs)
    }

    fn or<S: Shape>(
        &self,
        lhs: &Tensor<S, bool, Self>,
        rhs: &Tensor<S, bool, Self>,
    ) -> Result<Tensor<S, bool, Self>, Self::Err> {
        self.call_binary("boolean_or", lhs, rhs)
    }

    fn xor<S: Shape>(
        &self,
        lhs: &Tensor<S, bool, Self>,
        rhs: &Tensor<S, bool, Self>,
    ) -> Result<Tensor<S, bool, Self>, Self::Err> {
        self.call_binary("boolean_xor", lhs, rhs)
    }
}
