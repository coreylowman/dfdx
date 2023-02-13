use crate::{optim::optimizer::*, shapes::*, tensor::Cuda};
use cudarc::driver::{AsKernelParam, LaunchAsync, LaunchConfig};
use std::sync::Arc;

#[repr(C)]
struct CudaAdamConfig<E> {
    lr: E,
    beta1: E,
    beta2: E,
    eps: E,
    weight_decay_type: WeightDecayType,
    weight_decay: E,
}

unsafe impl<E> AsKernelParam for CudaAdamConfig<E> {}

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

impl<E: Dtype + AsKernelParam> super::AdamKernel<E> for Cuda
where
    Self: HasCudaKernel<E>,
{
    fn update<S: Shape>(
        &self,
        t: i32,
        cfg: &super::AdamConfig<E>,
        param: &mut Self::Storage<S, E>,
        moment1: &mut Self::Storage<S, E>,
        moment2: &mut Self::Storage<S, E>,
        grad: Self::Storage<S, E>,
    ) -> Result<(), Self::Err> {
        if !self.dev.has_func(Self::MOD, Self::FWD) {
            self.dev.load_ptx(PTX_SRC.into(), Self::MOD, &[Self::FWD])?;
        }

        let adam_cfg = adam_config_to_cuda(cfg);
        let numel = param.shape.num_elements();

        let func = self.dev.get_func(Self::MOD, Self::FWD).unwrap();
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let params = (
            adam_cfg,                         // const AdamConfig cfg,
            numel,                            // const size_t numel,
            <E>::from_i32(t).unwrap(),        // const float t,
            Arc::make_mut(&mut param.data),   // float* param,
            Arc::make_mut(&mut moment1.data), // float* moment1,
            Arc::make_mut(&mut moment2.data), // float* moment2,
            grad.data.as_ref(),               // const float* grad
        );
        unsafe { func.launch_async(cfg, params) }?;
        Ok(())
    }
}
