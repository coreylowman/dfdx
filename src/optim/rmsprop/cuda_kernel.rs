use super::RMSpropConfig;
use crate::{
    optim::optimizer::*,
    shapes::*,
    tensor::{launch_cfg, Cuda},
};

use cudarc::driver::{DeviceRepr, DeviceSlice, LaunchAsync};

#[repr(C)]
struct CudaRMSpropConfig<E> {
    lr: E,
    alpha: E,
    eps: E,
    centered: bool,
    has_momentum: bool,
    momentum: E,
    weight_decay_type: WeightDecayType,
    weight_decay: E,
}

unsafe impl<E: DeviceRepr> DeviceRepr for CudaRMSpropConfig<E> {}

fn rmsprop_config_to_cuda<E: Default + Copy>(config: &RMSpropConfig<E>) -> CudaRMSpropConfig<E> {
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
    fn update(
        &self,
        cfg: &RMSpropConfig<E>,
        param: &mut Self::Vec<E>,
        momentum: &mut Self::Vec<E>,
        square_avg: &mut Self::Vec<E>,
        grad_avg: &mut Self::Vec<E>,
        grad: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        if !self.dev.has_func(Self::MOD, Self::FWD) {
            self.dev.load_ptx(PTX_SRC.into(), Self::MOD, &[Self::FWD])?;
        }

        let opt_cfg = rmsprop_config_to_cuda(cfg);
        let numel = param.len();
        let func = self.dev.get_func(Self::MOD, Self::FWD).unwrap();
        let cfg = launch_cfg(numel as u32);
        let params = (opt_cfg, numel, param, momentum, square_avg, grad_avg, grad);
        unsafe { func.launch(cfg, params) }?;
        Ok(())
    }
}
