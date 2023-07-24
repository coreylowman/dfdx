use super::SgdConfig;

use crate::{
    dtypes::*,
    tensor::{launch_cfg, Cuda},
    tensor_ops::optim::*,
};

use cudarc::driver::{DeviceRepr, DeviceSlice, LaunchAsync};

#[repr(C)]
struct CudaSgdConfig {
    lr: f64,
    momentum_type: MomentumType,
    momentum: f64,
    weight_decay_type: WeightDecayType,
    weight_decay: f64,
}

unsafe impl DeviceRepr for CudaSgdConfig {}

fn sgd_config_to_cuda(config: &SgdConfig) -> CudaSgdConfig {
    let (momentum_type, momentum) = momentum_to_cuda(config.momentum);
    let (weight_decay_type, weight_decay) = weight_decay_to_cuda(config.weight_decay);

    CudaSgdConfig {
        lr: config.lr,
        momentum_type,
        momentum,
        weight_decay_type,
        weight_decay,
    }
}

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/sgd.ptx"));

trait HasCudaKernel<E> {
    const MOD: &'static str;
    const FWD: &'static str;
}

#[cfg(feature = "f16")]
impl HasCudaKernel<f16> for Cuda {
    const MOD: &'static str = "sgd_f16";
    const FWD: &'static str = "sgd_update_f16";
}

#[cfg(feature = "f16")]
impl HasCudaKernel<AMP<f16>> for Cuda {
    const MOD: &'static str = "sgd_amp_f16";
    const FWD: &'static str = "sgd_update_amp_f16";
}

impl HasCudaKernel<f32> for Cuda {
    const MOD: &'static str = "sgd_f32";
    const FWD: &'static str = "sgd_update_f32";
}

impl HasCudaKernel<f64> for Cuda {
    const MOD: &'static str = "sgd_f64";
    const FWD: &'static str = "sgd_update_f64";
}

impl<E: Dtype> super::SgdKernel<E> for Cuda
where
    Self: HasCudaKernel<E>,
{
    fn sgd_kernel(
        &self,
        cfg: &SgdConfig,
        param: &mut Self::Vec,
        velocity: &mut Self::Vec,
        grad: &Self::Vec,
    ) -> Result<(), Self::Err> {
        if !self.dev.has_func(Self::MOD, Self::FWD) {
            self.dev.load_ptx(PTX_SRC.into(), Self::MOD, &[Self::FWD])?;
        }

        let opt_cfg = sgd_config_to_cuda(cfg);
        let numel = param.len();
        let func = self.dev.get_func(Self::MOD, Self::FWD).unwrap();
        let cfg = launch_cfg::<128>(numel as u32);
        unsafe { func.launch(cfg, (opt_cfg, numel, param, velocity, grad)) }?;
        Ok(())
    }
}
