use crate::{
    shapes::*,
    tensor::{cuda::Cuda, Tensor},
};

use cudarc::driver::{DeviceSlice, LaunchAsync, LaunchConfig};

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/ema.ptx"));

trait HasCudaKernel<E> {
    const FN: &'static str;
}
impl HasCudaKernel<f32> for Cuda {
    const FN: &'static str = "ema_f32";
}
impl HasCudaKernel<f64> for Cuda {
    const FN: &'static str = "ema_f64";
}

impl<E: Dtype> super::EmaKernel<E> for Cuda
where
    Self: HasCudaKernel<E>,
{
    fn forward<S: Shape>(
        &self,
        dst: &mut Tensor<S, E, Self>,
        src: &Tensor<S, E, Self>,
        decay: E,
    ) -> Result<(), Self::Err> {
        if !self.dev.has_func(Self::FN, Self::FN) {
            self.dev.load_ptx(PTX_SRC.into(), Self::FN, &[Self::FN])?;
        }
        let numel = src.data.len();
        let fwd_fn = self.dev.get_func(Self::FN, Self::FN).unwrap();
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let params = (
            numel,
            src.data.as_ref(),
            std::sync::Arc::make_mut(&mut dst.data),
            decay,
        );
        unsafe { fwd_fn.launch(cfg, params) }?;
        Ok(())
    }
}
