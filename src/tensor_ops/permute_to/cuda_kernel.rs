use crate::shapes::*;
use crate::tensor::cuda::{Cuda, CudaArray};

use cudarc::driver::{LaunchAsync, LaunchConfig};
use std::sync::Arc;

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/permute_to.ptx"));

macro_rules! impl_permute {
    ($TypeName:ty, $SumFn:tt) => {
        impl super::PermuteKernel<$TypeName> for Cuda {
            fn forward<Src: Shape, Dst: Shape, Ax: Axes>(
                &self,
                inp: &Self::Storage<Src, $TypeName>,
            ) -> Result<Self::Storage<Dst, $TypeName>, Self::Err>
            where
                Src: PermuteShapeTo<Dst, Ax>,
            {
                Ok(CudaArray {
                    data: inp.data.clone(),
                    shape: inp.shape.permuted(),
                    strides: inp.shape.permute_strides(inp.strides),
                })
            }
            fn backward<Src: Shape, Dst: Shape, Ax: Axes>(
                &self,
                grad_inp: &mut Self::Storage<Src, $TypeName>,
                grad_out: &Self::Storage<Dst, $TypeName>,
            ) -> Result<(), Self::Err>
            where
                Src: PermuteShapeTo<Dst, Ax>,
            {
                if !self.dev.has_func("permute_to", $SumFn) {
                    self.dev.load_ptx(PTX_SRC.into(), "permute_to", &[$SumFn])?;
                }

                let f = self.dev.get_func("permute_to", $SumFn).unwrap();

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

impl_permute!(f32, "sum_f32");
impl_permute!(f64, "sum_f64");
