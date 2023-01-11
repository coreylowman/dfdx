use crate::{
    shapes::{Dtype, HasSameNumelAs, Shape},
    tensor::cuda::{Cuda, CudaArray},
    tensor_ops::ops::{BinaryKernel, UnaryKernel},
};
use cudarc::device::{AsKernelParam, CudaSlice, LaunchAsync, LaunchConfig, ValidAsZeroBits};
use std::sync::Arc;

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/reshape.ptx"));
const MODULE_NAME: &str = "reshape";
const FWD_FN_NAME: &str = "reshape_forward";
const BWD_FN_NAME: &str = "reshape_backward";
const ALL_FN_NAMES: [&str; 2] = [FWD_FN_NAME, BWD_FN_NAME];

impl<E: Dtype + ValidAsZeroBits> super::ReshapeKernel<E> for Cuda {
    fn forward<Src: Shape, Dst: Shape>(
        &self,
        dst: Dst,
        inp: &Self::Storage<Src, E>,
    ) -> Result<Self::Storage<Dst, E>, Self::Err>
    where
        Src: HasSameNumelAs<Dst>,
    {
        if !self.dev.has_func(MODULE_NAME, FWD_FN_NAME) {
            self.dev
                .load_ptx(PTX_SRC.into(), MODULE_NAME, &ALL_FN_NAMES)?;
        }

        let numel = inp.data.len();
        let mut storage = self.dev.alloc_zeros_async::<E>(numel)?;

        let fwd_fn = self.dev.get_func(MODULE_NAME, FWD_FN_NAME).unwrap();
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let params = (
            numel,             // const size_t numel,
            inp.data.as_ref(), // const float *inp,
            &mut storage,      // float *out
        );
        unsafe { fwd_fn.launch_async(cfg, params) }?;

        Ok(CudaArray {
            data: Arc::new(storage),
            shape: dst,
            strides: dst.strides(),
        })
    }

    fn backward<Src: Shape, Dst: Shape>(
        &self,
        grad_inp: &mut Self::Storage<Src, E>,
        grad_out: &Self::Storage<Dst, E>,
    ) -> Result<(), Self::Err>
    where
        Src: HasSameNumelAs<Dst>,
    {
        let bwd_fn = self.dev.get_func(MODULE_NAME, BWD_FN_NAME).unwrap();
        let numel = grad_inp.data.len();
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let params = (
            numel,                             // const size_t numel,
            Arc::make_mut(&mut grad_inp.data), // float *grad_inp,
            grad_out.data.as_ref(),            // const float *grad_out
        );
        unsafe { bwd_fn.launch_async(cfg, params) }?;
        Ok(())
    }
}
