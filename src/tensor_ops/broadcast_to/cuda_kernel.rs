use crate::shapes::*;
use crate::tensor::cuda::{Cuda, CudaArray};

use cudarc::driver::{LaunchAsync, LaunchConfig};
use std::sync::Arc;

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/broadcast_to.ptx"));

macro_rules! impl_broadcast {
    ($TypeName:tt, $SumFn:tt) => {
        impl super::BroadcastKernel<$TypeName> for Cuda {
            fn forward<Src: Shape, Dst: Shape, Ax: Axes>(
                &self,
                dst: Dst,
                inp: &Self::Storage<Src, $TypeName>,
            ) -> Result<Self::Storage<Dst, $TypeName>, Self::Err>
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
                grad_inp: &mut Self::Storage<Src, $TypeName>,
                grad_out: &Self::Storage<Dst, $TypeName>,
            ) -> Result<(), Self::Err>
            where
                Src: BroadcastShapeTo<Dst, Ax>,
            {
                if !self.dev.has_func("broadcast_to", $SumFn) {
                    self.dev
                        .load_ptx(PTX_SRC.into(), "broadcast_to", &[$SumFn])?;
                }
                let f = self.dev.get_func("broadcast_to", $SumFn).unwrap();
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
    };
}

impl_broadcast!(f32, "sum_f32");
impl_broadcast!(f64, "sum_f64");
