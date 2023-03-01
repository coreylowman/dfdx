use crate::shapes::*;
use crate::tensor::{cuda::Cuda, Tensor};
use crate::unique_id::unique_id;

use cudarc::driver::{LaunchAsync, LaunchConfig};

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/broadcast_to.ptx"));

trait HasCudaKernel<E> {
    const MOD: &'static str;
    const FNS: &'static [&'static str];
}

impl HasCudaKernel<f32> for Cuda {
    const MOD: &'static str = "broadcast_f32";
    const FNS: &'static [&'static str] = &["sum_f32"];
}

impl HasCudaKernel<f64> for Cuda {
    const MOD: &'static str = "broadcast_f64";
    const FNS: &'static [&'static str] = &["sum_f64"];
}

impl<E: Dtype> super::BroadcastKernel<E> for Cuda
where
    Self: HasCudaKernel<E>,
{
    fn forward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        dst: Dst,
        inp: &Tensor<Src, E, Self>,
    ) -> Result<Tensor<Dst, E, Self>, Self::Err>
    where
        Src: BroadcastShapeTo<Dst, Ax>,
    {
        Ok(Tensor {
            id: unique_id(),
            data: inp.data.clone(),
            shape: dst,
            strides: inp.shape.broadcast_strides(inp.strides),
            device: self.clone(),
            tape: Default::default(),
        })
    }
    fn backward(
        &self,
        grad_inp: &mut Self::Vec<E>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        if !self.dev.has_func(Self::MOD, Self::FNS[0]) {
            self.dev.load_ptx(PTX_SRC.into(), Self::MOD, Self::FNS)?;
        }
        let f = self.dev.get_func(Self::MOD, Self::FNS[0]).unwrap();
        let numel = grad_inp.len();
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        unsafe { f.launch_async(cfg, (numel, grad_out, grad_inp)) }?;
        Ok(())
    }
}
