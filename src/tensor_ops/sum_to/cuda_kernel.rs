use crate::{
    shapes::{Axes, BroadcastStridesTo, ReduceShapeTo, Shape},
    tensor::cuda::{Cuda, CudaArray},
};

use cudarc::driver::{CudaSlice, LaunchAsync, LaunchConfig};

use std::sync::Arc;
use std::vec::Vec;

const MODULE_NAME: &str = "sum_to";
const FWD_FN_NAME: &str = "sum_to_forward";
const BWD_FN_NAME: &str = "sum_to_backward";
const ALL_FN_NAMES: [&str; 2] = [FWD_FN_NAME, BWD_FN_NAME];
const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/sum_to.ptx"));

/// Moves all axes in Ax to the end of dims and strides and removes broadcasted dimensions
/// so that a cuda kernel called for each physical element of the input tensor will place elements
/// to be reduced with each other next to each other in memory.
fn permute_for_reductions<I, Ax: Axes>(dims: I, strides: I) -> (Vec<usize>, Vec<usize>)
where
    I: IntoIterator<Item = usize>,
{
    let mut tmp = dims
        .into_iter()
        .zip(strides.into_iter())
        .map(|x| (false, x))
        .collect::<Vec<_>>();

    for i in Ax::as_array().into_iter() {
        tmp[i as usize].0 = true;
    }

    // requires stable sorting to keep non-summed axes in the correct order
    tmp.sort_by_key(|x| x.0);

    tmp.into_iter()
        .map(|(_, x)| x)
        .filter(|(_, stride)| *stride != 0)
        .unzip()
}

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

        let (dims, strides) = permute_for_reductions::<_, Ax>(inp.shape.concrete(), inp.strides);
        let num_dims = dims.len();
        let dims: CudaSlice<usize> = self.dev.take_async(dims)?;
        let inp_strides: CudaSlice<usize> = self.dev.take_async(strides)?;

        let mut storage = self.dev.alloc_zeros_async::<f32>(dst.num_elements())?;

        let physical_numel = inp.data.len();
        let virtual_numel = inp.shape.num_elements();
        let elems_per_thread = (virtual_numel / physical_numel) as f32;

        let chunk_len = physical_numel / dst.num_elements();

        let cfg = LaunchConfig::for_num_elems(physical_numel as u32);
        let params = (
            physical_numel,    // const size_t numel,
            num_dims,          // const size_t num_dims,
            elems_per_thread,  // const float elems_per_thread,
            chunk_len,         // const size_t chunk_len,
            &dims,             // const size_t *dims,
            inp.data.as_ref(), // const float *inp,
            &inp_strides,      // const size_t *inp_strides,
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

        let physical_numel = grad_inp.data.len();
        let virtual_numel = grad_inp.shape.num_elements();
        let elems_per_thread = (virtual_numel / physical_numel) as f32;

        let cfg = LaunchConfig::for_num_elems(physical_numel as u32);
        let params = (
            physical_numel,                    // const size_t numel,
            Src::NUM_DIMS,                     // const size_t num_dims,
            elems_per_thread,                  // const float elems_per_thread,
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
