use super::RMSpropConfig;
use crate::{
    dtypes::*,
    tensor::{launch_cfg, Cuda},
    tensor_ops::optim::*,
};

use cudarc::driver::{DeviceRepr, DeviceSlice, LaunchAsync};

#[repr(C)]
struct CudaRMSpropConfig {
    lr: f64,
    alpha: f64,
    eps: f64,
    centered: bool,
    has_momentum: bool,
    momentum: f64,
    weight_decay_type: WeightDecayType,
    weight_decay: f64,
}

unsafe impl DeviceRepr for CudaRMSpropConfig {}

fn rmsprop_config_to_cuda(config: &RMSpropConfig) -> CudaRMSpropConfig {
    let (weight_decay_type, weight_decay) = weight_decay_to_cuda(config.weight_decay);
    let (has_momentum, momentum) = if let Some(m) = config.momentum {
        (true, m)
    } else {
        (false, Default::default())
    };

    CudaRMSpropConfig {
        lr: config.lr,
        alpha: config.alpha,
        eps: config.eps,
        centered: config.centered,
        has_momentum,
        momentum,
        weight_decay_type,
        weight_decay,
    }
}

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/rmsprop.ptx"));

trait HasCudaKernel<E> {
    const MOD: &'static str;
    const FWD: &'static str;
}

#[cfg(feature = "f16")]
impl HasCudaKernel<f16> for Cuda {
    const MOD: &'static str = "rmsprop_f16";
    const FWD: &'static str = "rmsprop_update_f16";
}

#[cfg(feature = "f16")]
impl HasCudaKernel<AMP<f16>> for Cuda {
    const MOD: &'static str = "rmsprop_amp_f16";
    const FWD: &'static str = "rmsprop_update_amp_f16";
}

impl HasCudaKernel<f32> for Cuda {
    const MOD: &'static str = "rmsprop_f32";
    const FWD: &'static str = "rmsprop_update_f32";
}

impl HasCudaKernel<f64> for Cuda {
    const MOD: &'static str = "rmsprop_f64";
    const FWD: &'static str = "rmsprop_update_f64";
}

impl<E: Dtype> super::RMSpropKernel<E> for Cuda
where
    Self: HasCudaKernel<E>,
{
    fn rmsprop_kernel(
        &self,
        cfg: &RMSpropConfig,
        param: &mut Self::Vec,
        momentum: &mut Self::Vec,
        square_avg: &mut Self::Vec,
        grad_avg: &mut Self::Vec,
        grad: &Self::Vec,
    ) -> Result<(), Self::Err> {
        if !self.dev.has_func(Self::MOD, Self::FWD) {
            self.dev.load_ptx(PTX_SRC.into(), Self::MOD, &[Self::FWD])?;
        }

        let opt_cfg = rmsprop_config_to_cuda(cfg);
        let numel = param.len();
        let func = self.dev.get_func(Self::MOD, Self::FWD).unwrap();
        let cfg = launch_cfg::<128>(numel as u32);
        let params = (opt_cfg, numel, param, momentum, square_avg, grad_avg, grad);
        unsafe { func.launch(cfg, params) }?;
        Ok(())
    }
}
