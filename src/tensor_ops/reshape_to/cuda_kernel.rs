use crate::{
    shapes::{HasSameNumelAs, Shape},
    tensor::cuda::{Cuda, CudaArray},
};
use cudarc::device::{CudaSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/reshape.ptx"));
const MODULE_NAME: &str = "reshape";
const FWD_FN_NAME: &str = "reshape_forward";
const BWD_FN_NAME: &str = "reshape_backward";
const ALL_FN_NAMES: [&str; 2] = [FWD_FN_NAME, BWD_FN_NAME];

impl super::ReshapeKernel<f32> for Cuda {
    fn forward<Src: Shape, Dst: Shape>(
        &self,
        dst: Dst,
        inp: &Self::Storage<Src, f32>,
    ) -> Result<Self::Storage<Dst, f32>, Self::Err>
    where
        Src: HasSameNumelAs<Dst>,
    {
        if !self.dev.has_func(MODULE_NAME, FWD_FN_NAME) {
            self.dev
                .load_ptx(PTX_SRC.into(), MODULE_NAME, &ALL_FN_NAMES)?;
        }

        let numel = inp.data.len();
        let mut storage = self.dev.alloc_zeros_async::<f32>(numel)?;

        let inp_dims: CudaSlice<usize> = self.dev.take_async(inp.shape.concrete().into())?;
        let dst_dims: CudaSlice<usize> = self.dev.take_async(dst.concrete().into())?;
        let inp_strides: CudaSlice<usize> = self.dev.take_async(inp.strides.into())?;
        let dst_strides: CudaSlice<usize> = self.dev.take_async(dst.strides().into())?;

        let fwd_fn = self.dev.get_func(MODULE_NAME, FWD_FN_NAME).unwrap();
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let params = (
            numel,             // const size_t numel,
            inp.data.as_ref(), // const float *inp,
            Src::NUM_DIMS,     // const size_t inp_num_dims,
            &inp_dims,         // const size_t *inp_dims,
            &inp_strides,      // const size_t *inp_strides,
            &mut storage,      // float *out
            Dst::NUM_DIMS,     // const size_t out_num_dims,
            &dst_dims,         // const size_t *out_dims,
            &dst_strides,      // const size_t *out_strides,
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
        grad_inp: &mut Self::Storage<Src, f32>,
        grad_out: &Self::Storage<Dst, f32>,
    ) -> Result<(), Self::Err>
    where
        Src: HasSameNumelAs<Dst>,
    {
        let bwd_fn = self.dev.get_func(MODULE_NAME, BWD_FN_NAME).unwrap();
        let numel = grad_inp.data.len();

        let inp_dims: CudaSlice<usize> = self.dev.take_async(grad_inp.shape.concrete().into())?;
        let out_dims: CudaSlice<usize> = self.dev.take_async(grad_out.shape.concrete().into())?;
        let inp_strides: CudaSlice<usize> = self.dev.take_async(grad_inp.strides.into())?;
        let out_strides: CudaSlice<usize> = self.dev.take_async(grad_out.strides.into())?;

        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let params = (
            numel,                             // const size_t numel,
            Arc::make_mut(&mut grad_inp.data), // float *grad_inp,
            Src::NUM_DIMS,                     // const size_t inp_num_dims,
            &inp_dims,                         // const size_t *inp_dims,
            &inp_strides,                      // const size_t *inp_strides,
            grad_out.data.as_ref(),            // const float *grad_out,
            Dst::NUM_DIMS,                     // const size_t out_num_dims,
            &out_dims,                         // const size_t *out_dims,
            &out_strides,                      // const size_t *out_strides
        );
        unsafe { bwd_fn.launch_async(cfg, params) }?;
        Ok(())
    }
}
