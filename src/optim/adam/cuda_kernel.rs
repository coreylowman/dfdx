use super::AdamConfig;
use crate::optim::optimizer::*;
use crate::{shapes::Shape, tensor::Cuda};
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

fn adam_config_to_cuda<E: Default + Copy>(config: &AdamConfig<E>) -> CudaAdamConfig<E> {
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

const MODULE_NAME: &str = "adam";
const FN_NAME: &str = "adam_update";
const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/adam.ptx"));

impl super::AdamKernel<f32> for Cuda {
    fn update<S: Shape>(
        &self,
        t: i32,
        cfg: &AdamConfig<f32>,
        param: &mut Self::Storage<S, f32>,
        moment1: &mut Self::Storage<S, f32>,
        moment2: &mut Self::Storage<S, f32>,
        grad: Self::Storage<S, f32>,
    ) -> Result<(), Self::Err> {
        debug_assert_eq!(param.data.len(), grad.data.len());
        debug_assert_eq!(param.shape, grad.shape);
        debug_assert_eq!(param.strides, grad.strides);

        if !self.dev.has_func(MODULE_NAME, FN_NAME) {
            self.dev.load_ptx(PTX_SRC.into(), MODULE_NAME, &[FN_NAME])?;
        }

        let adam_cfg = adam_config_to_cuda(cfg);
        let numel = param.shape.num_elements();

        let func = self.dev.get_func(MODULE_NAME, FN_NAME).unwrap();
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let params = (
            adam_cfg,                         // const AdamConfig cfg,
            numel,                            // const size_t numel,
            t as f32,                         // const float t,
            Arc::make_mut(&mut param.data),   // float* param,
            Arc::make_mut(&mut moment1.data), // float* moment1,
            Arc::make_mut(&mut moment2.data), // float* moment2,
            grad.data.as_ref(),               // const float* grad
        );
        unsafe { func.launch_async(cfg, params) }?;
        Ok(())
    }
}
