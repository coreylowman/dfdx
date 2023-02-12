use super::RMSpropConfig;
use crate::{optim::optimizer::*, shapes::*, tensor::Cuda};
use cudarc::driver::{AsKernelParam, LaunchAsync, LaunchConfig};
use std::sync::Arc;

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

unsafe impl<E> AsKernelParam for CudaRMSpropConfig<E> {}

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

impl<E: Dtype + AsKernelParam> super::RMSpropKernel<E> for Cuda
where
    Self: HasCudaKernel<E>,
{
    fn update<S: Shape>(
        &self,
        cfg: &RMSpropConfig<E>,
        param: &mut Self::Storage<S, E>,
        momentum: &mut Self::Storage<S, E>,
        square_avg: &mut Self::Storage<S, E>,
        grad_avg: &mut Self::Storage<S, E>,
        grad: Self::Storage<S, E>,
    ) -> Result<(), Self::Err> {
        if !self.dev.has_func(Self::MOD, Self::FWD) {
            self.dev.load_ptx(PTX_SRC.into(), Self::MOD, &[Self::FWD])?;
        }

        let rmsprop_cfg = rmsprop_config_to_cuda(cfg);
        let numel = param.shape.num_elements();

        let func = self.dev.get_func(Self::MOD, Self::FWD).unwrap();
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let params = (
            rmsprop_cfg,                         // const RMSpropConfig cfg,
            numel,                               // const size_t numel,
            Arc::make_mut(&mut param.data),      // float* param,
            Arc::make_mut(&mut momentum.data),   // float* momentum,
            Arc::make_mut(&mut square_avg.data), // float* square_avg,
            Arc::make_mut(&mut grad_avg.data),   // float* grad_avg,
            grad.data.as_ref(),                  // const float* grad
        );
        unsafe { func.launch_async(cfg, params) }?;
        Ok(())
    }
}
