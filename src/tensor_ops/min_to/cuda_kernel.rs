use crate::tensor_ops::internal_reshapes::permute_for_reductions;
use crate::{
    shapes::{Axes, BroadcastStridesTo, ReduceShapeTo, Shape},
    tensor::cuda::{Cuda, CudaArray},
};

use cudarc::driver::{CudaSlice, LaunchAsync, LaunchConfig};

use std::sync::Arc;

const MODULE_NAME: &str = "min_to";
const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/min_to.ptx"));


macro_rules! impl_min_reduce {
    ($TypeName:ty, $Fwd:tt, $Bwd:tt, $Fill:tt) => {
        impl super::MinReduceKernel<$TypeName> for Cuda {
            fn forward<Src: Shape, Dst: Shape, Ax: Axes>(
                &self,
                dst: Dst,
                inp: &Self::Storage<Src, $TypeName>,
            ) -> Result<Self::Storage<Dst, $TypeName>, Self::Err>
            where
                Src: ReduceShapeTo<Dst, Ax>,
            {
                if !self.dev.has_func(MODULE_NAME, $Fwd) {
                    self.dev
                        .load_ptx(PTX_SRC.into(), MODULE_NAME, &[$Fwd, $Bwd, $Fill])?;
                }

                let fill_fn = self.dev.get_func(MODULE_NAME, "fill_with").unwrap();
                let mut storage = unsafe {
                    let mut storage = self.dev.alloc_async::<$TypeName>(dst.num_elements())?;
                    fill_fn.launch_async(
                        LaunchConfig::for_num_elems(dst.num_elements() as u32),
                        (&mut storage, $TypeName::INFINITY, dst.num_elements()),
                    )?;
                    storage
                };
        
                let fwd_fn = self.dev.get_func(MODULE_NAME, FWD_FN_NAME).unwrap();
        
                let (dims, strides) = permute_for_reductions::<_, Ax>(inp.shape.concrete(), inp.strides);
                let dims: CudaSlice<usize> = self.dev.take_async(dims)?;
                let strides: CudaSlice<usize> = self.dev.take_async(strides)?;
        
                let physical_numel = inp.data.len();
                let chunk_len = physical_numel / dst.num_elements();
        
                let cfg = LaunchConfig::for_num_elems(physical_numel as u32);
                let params = (
                    physical_numel,    // const size_t numel,
                    dims.len(),        // const size_t num_dims,
                    chunk_len,         // const size_t chunk_len,
                    inp.data.as_ref(), // const float *inp,
                    &dims,             // const size_t *dims,
                    &strides,          // const size_t *strides,
                    &mut storage,      // float *out
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
                inp: &Self::Storage<Src, $TypeName>,
                grad_inp: &mut Self::Storage<Src, $TypeName>,
                out: &Self::Storage<Dst, $TypeName>,
                grad_out: &Self::Storage<Dst, $TypeName>,
            ) -> Result<(), Self::Err>
            where
                Src: ReduceShapeTo<Dst, Ax>,
            {
                let bwd_fn = self.dev.get_func(MODULE_NAME, $Bwd).unwrap();
        
                let dims: CudaSlice<usize> = self.dev.take_async(grad_inp.shape.concrete().into())?;
                let inp_strides: CudaSlice<usize> = self.dev.take_async(grad_inp.strides.into())?;
                let out_strides: Src::Concrete =
                    BroadcastStridesTo::<Src, Ax>::broadcast_strides(&grad_out.shape, grad_out.strides);
                let out_strides: CudaSlice<usize> = self.dev.take_async(out_strides.into())?;
        
                let physical_numel = grad_inp.data.len();
                let virtual_numel = grad_inp.shape.num_elements();
                let elems_per_thread = (virtual_numel / physical_numel) as $TypeName;
        
                let cfg = LaunchConfig::for_num_elems(physical_numel as u32);
                let params = (
                    physical_numel,                    // const size_t numel,
                    Src::NUM_DIMS,                     // const size_t num_dims,
                    elems_per_thread,                  // const float elems_per_thread,
                    &dims,                             // const size_t *dims,
                    inp.data.as_ref(),                 // const float *inp,
                    Arc::make_mut(&mut grad_inp.data), // float *grad_inp,
                    &inp_strides,                      // const size_t *inp_strides,
                    out.data.as_ref(),                 // const float *out,
                    grad_out.data.as_ref(),            // const float *grad_out,
                    &out_strides,                      // const size_t *out_strides
                );
                unsafe { bwd_fn.launch_async(cfg, params) }?;
                Ok(())
            }
        }
                
    };
}
