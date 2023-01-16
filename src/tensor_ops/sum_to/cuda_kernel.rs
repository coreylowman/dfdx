use crate::{
    shapes::{Axes, BroadcastStridesTo, ReduceShapeTo, Shape},
    tensor::cuda::{Cuda, CudaArray},
};

use cudarc::driver::{CudaSlice, LaunchAsync, LaunchConfig};

use std::sync::Arc;

const MODULE_NAME: &str = "sum_to";
const FWD_FN_NAME: &str = "sum_to_forward";
const BWD_FN_NAME: &str = "sum_to_backward";
const ALL_FN_NAMES: [&str; 2] = [FWD_FN_NAME, BWD_FN_NAME];
const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/sum_to.ptx"));

impl super::SumKernel<f32> for Cuda {
    fn forward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        dst: Dst,
        inp: &Self::Storage<Src, f32>,
    ) -> Result<Self::Storage<Dst, f32>, Self::Err>
    where
        Src: ReduceShapeTo<Dst, Ax>,
    {
        if !self.dev.has_func(MODULE_NAME, FWD_FN_NAME) {
            self.dev
                .load_ptx(PTX_SRC.into(), MODULE_NAME, &ALL_FN_NAMES)?;
        }

        let fwd_fn = self.dev.get_func(MODULE_NAME, FWD_FN_NAME).unwrap();

        let dims: CudaSlice<usize> = self.dev.take_async(inp.shape.concrete().into())?;
        let inp_strides: CudaSlice<usize> = self.dev.take_async(inp.strides.into())?;
        let out_strides: Src::Concrete =
            BroadcastStridesTo::<Src, Ax>::broadcast_strides(&dst, dst.strides());
        let out_strides: CudaSlice<usize> = self.dev.take_async(out_strides.into())?;

        let mut storage = self.dev.alloc_zeros_async::<f32>(dst.num_elements())?;

        let inp_numel = inp.data.len();
        let numel = inp.shape.num_elements();
        let mul = (numel / inp_numel) as f32;

        let cfg = LaunchConfig::for_num_elems(inp_numel as u32);
        let params = (
            inp_numel,         // const size_t numel,
            Src::NUM_DIMS,     // const size_t num_dims,
            mul,               // const float mul,
            &dims,             // const size_t *dims,
            inp.data.as_ref(), // const float *inp,
            &inp_strides,      // const size_t *inp_strides,
            &mut storage,      // float *out,
            &out_strides,      // const size_t *out_strides
        );
        unsafe { fwd_fn.launch_async(cfg, params) }?;
        Ok(CudaArray {
            data: Arc::new(storage),
            shape: dst,
            strides: dst.strides(),
        })
    }

    fn backward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        grad_inp: &mut Self::Storage<Src, f32>,
        grad_out: &Self::Storage<Dst, f32>,
    ) -> Result<(), Self::Err>
    where
        Src: ReduceShapeTo<Dst, Ax>,
    {
        let bwd_fn = self.dev.get_func(MODULE_NAME, BWD_FN_NAME).unwrap();

        let dims: CudaSlice<usize> = self.dev.take_async(grad_inp.shape.concrete().into())?;
        let inp_strides: CudaSlice<usize> = self.dev.take_async(grad_inp.strides.into())?;
        let out_strides: Src::Concrete =
            BroadcastStridesTo::<Src, Ax>::broadcast_strides(&grad_out.shape, grad_out.strides);
        let out_strides: CudaSlice<usize> = self.dev.take_async(out_strides.into())?;

        let inp_numel = grad_inp.data.len();
        let numel = grad_inp.shape.num_elements();
        let mul = (numel / inp_numel) as f32;

        let cfg = LaunchConfig::for_num_elems(inp_numel as u32);
        let params = (
            inp_numel,                         // const size_t numel,
            Src::NUM_DIMS,                     // const size_t num_dims,
            mul,                               // const float mul,
            &dims,                             // const size_t *dims,
            Arc::make_mut(&mut grad_inp.data), // float *grad_inp,
            &inp_strides,                      // const size_t *inp_strides,
            grad_out.data.as_ref(),            // const float *grad_out,
            &out_strides,                      // const size_t *out_strides
        );
        unsafe { bwd_fn.launch_async(cfg, params) }?;
        Ok(())
    }
}
