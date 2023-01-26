use super::SgdConfig;
use crate::optim::optimizer::*;
use crate::{shapes::Shape, tensor::Cuda};
use cudarc::driver::{AsKernelParam, LaunchAsync, LaunchConfig};
use std::sync::Arc;

#[repr(C)]
struct CudaSgdConfig<E> {
    lr: E,
    momentum_type: MomentumType,
    momentum: E,
    weight_decay_type: WeightDecayType,
    weight_decay: E,
}

unsafe impl<E> AsKernelParam for CudaSgdConfig<E> {}

fn sgd_config_to_cuda<E: Default + Copy>(config: &SgdConfig<E>) -> CudaSgdConfig<E> {
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

const MODULE_NAME: &str = "sgd";
const FN_NAME: &str = "sgd_update";
const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/sgd.ptx"));

impl super::SgdKernel<f32> for Cuda {
    fn update<S: Shape>(
        &self,
        cfg: &SgdConfig<f32>,
        param: &mut Self::Storage<S, f32>,
        velocity: &mut Self::Storage<S, f32>,
        grad: Self::Storage<S, f32>,
    ) -> Result<(), Self::Err> {
        debug_assert_eq!(param.data.len(), grad.data.len());
        debug_assert_eq!(param.shape, grad.shape);
        debug_assert_eq!(param.strides, grad.strides);

        if !self.dev.has_func(MODULE_NAME, FN_NAME) {
            self.dev.load_ptx(PTX_SRC.into(), MODULE_NAME, &[FN_NAME])?;
        }

        let sgd_cfg = sgd_config_to_cuda(cfg);
        let numel = param.shape.num_elements();

        let func = self.dev.get_func(MODULE_NAME, FN_NAME).unwrap();
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let params = (
            sgd_cfg,                           // const SgdConfig cfg,
            numel,                             // const size_t numel,
            Arc::make_mut(&mut param.data),    // float* param,
            Arc::make_mut(&mut velocity.data), // float* velocity,
            grad.data.as_ref(),                // const float* grad
        );
        unsafe { func.launch_async(cfg, params) }?;
        Ok(())
    }
}
