use super::RMSpropConfig;
use crate::optim::optimizer::*;
use crate::{shapes::Shape, tensor::Cuda};
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

const MODULE_NAME: &str = "rmsprop";
const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/rmsprop.ptx"));

macro_rules! impl_rmsprop {
    ($TypeName:ty, $Fwd:tt) => {
        impl super::RMSpropKernel<$TypeName> for Cuda {
            fn update<S: Shape>(
                &self,
                cfg: &RMSpropConfig<$TypeName>,
                param: &mut Self::Storage<S, $TypeName>,
                momentum: &mut Self::Storage<S, $TypeName>,
                square_avg: &mut Self::Storage<S, $TypeName>,
                grad_avg: &mut Self::Storage<S, $TypeName>,
                grad: Self::Storage<S, $TypeName>,
            ) -> Result<(), Self::Err> {
                if !self.dev.has_func(MODULE_NAME, $Fwd) {
                    self.dev.load_ptx(PTX_SRC.into(), MODULE_NAME, &[$Fwd])?;
                }

                let rmsprop_cfg = rmsprop_config_to_cuda(cfg);
                let numel = param.shape.num_elements();

                let func = self.dev.get_func(MODULE_NAME, $Fwd).unwrap();
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
    };
}

impl_rmsprop!(f32, "rmsprop_update_f32");
impl_rmsprop!(f64, "rmsprop_update_f64");
