use crate::{
    optim::optimizer::*,
    shapes::*,
    tensor::{launch_cfg, Cuda},
};

use cudarc::driver::{DeviceRepr, DeviceSlice, LaunchAsync};

#[repr(C)]
struct CudaAdamConfig<E> {
    lr: E,
    beta1: E,
    beta2: E,
    eps: E,
    weight_decay_type: WeightDecayType,
    weight_decay: E,
}

unsafe impl<E: DeviceRepr> DeviceRepr for CudaAdamConfig<E> {}

fn adam_config_to_cuda<E: Default + Copy>(config: &super::AdamConfig<E>) -> CudaAdamConfig<E> {
    let (weight_decay_type, weight_decay) = weight_decay_to_cuda(config.weight_decay);

    CudaAdamConfig {
        lr: config.lr,
        beta1: config.betas[0],
        beta2: config.betas[1],
        eps: config.eps,
        weight_decay_type,
        weight_decay,
    }
}

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/adam.ptx"));

trait HasCudaKernel<E> {
    const MOD: &'static str;
    const FWD: &'static str;
}

impl HasCudaKernel<f32> for Cuda {
    const MOD: &'static str = "adam_f32";
    const FWD: &'static str = "adam_update_f32";
}

impl HasCudaKernel<f64> for Cuda {
    const MOD: &'static str = "adam_f64";
    const FWD: &'static str = "adam_update_f64";
}

impl<E: Dtype> super::AdamKernel<E> for Cuda
where
    Self: HasCudaKernel<E>,
{
    fn update(
        &self,
        t: i32,
        cfg: &super::AdamConfig<E>,
        param: &mut Self::Vec<E>,
        moment1: &mut Self::Vec<E>,
        moment2: &mut Self::Vec<E>,
        grad: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        if !self.dev.has_func(Self::MOD, Self::FWD) {
            self.dev.load_ptx(PTX_SRC.into(), Self::MOD, &[Self::FWD])?;
        }

        let opt_cfg = adam_config_to_cuda(cfg);
        let numel = param.len();
        let func = self.dev.get_func(Self::MOD, Self::FWD).unwrap();
        let cfg = launch_cfg(numel as u32);
        let t = <E>::from_i32(t).unwrap();
        let params = (opt_cfg, numel, t, param, moment1, moment2, grad);
        unsafe { func.launch(cfg, params) }?;
        Ok(())
    }
}
