use crate::shapes::*;
use crate::tensor::cuda::{Cuda, CudaArray};

use cudarc::driver::{LaunchAsync, LaunchConfig};
use std::sync::Arc;

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/permute_to.ptx"));

trait HasCudaKernel<E> {
    const MOD: &'static str;
    const FNS: &'static [&'static str];
}

impl HasCudaKernel<f32> for Cuda {
    const MOD: &'static str = "permute_f32";
    const FNS: &'static [&'static str] = &["sum_f32"];
}

impl HasCudaKernel<f64> for Cuda {
    const MOD: &'static str = "permute_f64";
    const FNS: &'static [&'static str] = &["sum_f64"];
}

impl<E: Dtype> super::PermuteKernel<E> for Cuda
where
    Self: HasCudaKernel<E>,
{
    fn forward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        inp: &Self::Storage<Src, E>,
    ) -> Result<Self::Storage<Dst, E>, Self::Err>
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
        grad_inp: &mut Self::Storage<Src, E>,
        grad_out: &Self::Storage<Dst, E>,
    ) -> Result<(), Self::Err>
    where
        Src: PermuteShapeTo<Dst, Ax>,
    {
        if !self.dev.has_func(Self::MOD, Self::FNS[0]) {
            self.dev.load_ptx(PTX_SRC.into(), Self::MOD, Self::FNS)?;
        }

        let f = self.dev.get_func(Self::MOD, Self::FNS[0]).unwrap();

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
