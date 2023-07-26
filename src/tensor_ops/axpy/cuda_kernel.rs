use crate::{
    dtypes::*,
    tensor::{launch_cfg, Cuda},
};

use cudarc::driver::{DeviceSlice, LaunchAsync};

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/axpy.ptx"));

trait HasCudaKernel<E> {
    const FN: &'static str;
}
#[cfg(feature = "f16")]
impl HasCudaKernel<AMP<f16>> for Cuda {
    const FN: &'static str = "axpy_f16";
}
#[cfg(feature = "f16")]
impl HasCudaKernel<f16> for Cuda {
    const FN: &'static str = "axpy_f16";
}
impl HasCudaKernel<f32> for Cuda {
    const FN: &'static str = "axpy_f32";
}
impl HasCudaKernel<f64> for Cuda {
    const FN: &'static str = "axpy_f64";
}

impl<E: Dtype> super::AxpyKernel<E> for Cuda
where
    Self: HasCudaKernel<E>,
{
    fn forward(
        &self,
        a: &mut Self::Vec,
        alpha: E,
        b: &Self::Vec,
        beta: E,
    ) -> Result<(), Self::Err> {
        if !self.dev.has_func(Self::FN, Self::FN) {
            self.dev.load_ptx(PTX_SRC.into(), Self::FN, &[Self::FN])?;
        }
        let numel = a.len();
        let fwd_fn = self.dev.get_func(Self::FN, Self::FN).unwrap();
        let cfg = launch_cfg::<128>(numel as u32);
        unsafe { fwd_fn.launch(cfg, (numel, a, alpha, b, beta)) }?;
        Ok(())
    }
}
