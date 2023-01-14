use crate::shapes::*;
use crate::tensor::cuda::{Cuda, CudaArray};

use cudarc::driver::{LaunchAsync, LaunchConfig};
use std::sync::Arc;

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/broadcast_to.ptx"));

impl<E: Dtype> super::BroadcastKernel<E> for Cuda {
    fn forward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        dst: Dst,
        inp: &Self::Storage<Src, E>,
    ) -> Result<Self::Storage<Dst, E>, Self::Err>
    where
        Src: BroadcastShapeTo<Dst, Ax>,
    {
        Ok(CudaArray {
            data: inp.data.clone(),
            shape: dst,
            strides: inp.shape.broadcast_strides(inp.strides),
        })
    }
    fn backward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        grad_inp: &mut Self::Storage<Src, E>,
        grad_out: &Self::Storage<Dst, E>,
    ) -> Result<(), Self::Err>
    where
        Src: BroadcastShapeTo<Dst, Ax>,
    {
        if !self.dev.has_func("broadcast_to", "sum") {
            self.dev
                .load_ptx(PTX_SRC.into(), "broadcast_to", &["sum"])?;
        }

        let f = self.dev.get_func("broadcast_to", "sum").unwrap();

        let numel = grad_inp.data.len();
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let params = (
            numel,                             // const size_t numel,
            grad_out.data.as_ref(),            // const float *inp,
            Arc::make_mut(&mut grad_inp.data), // float *out
        );
        unsafe { f.launch_async(cfg, params) }?;
        Ok(())
    }
}
