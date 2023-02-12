use crate::{optim::optimizer::*, shapes::Shape, tensor::Cuda};
use cudarc::driver::{AsKernelParam, LaunchAsync, LaunchConfig};
use num_traits::FromPrimitive;
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

const MODULE_NAME: &str = "adam";
const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/adam.ptx"));

macro_rules! impl_adam {
    ($TypeName:ty, $Fwd:tt) => {
        impl super::AdamKernel<$TypeName> for Cuda {
            fn update<S: Shape>(
                &self,
                t: i32,
                cfg: &super::AdamConfig<$TypeName>,
                param: &mut Self::Storage<S, $TypeName>,
                moment1: &mut Self::Storage<S, $TypeName>,
                moment2: &mut Self::Storage<S, $TypeName>,
                grad: Self::Storage<S, $TypeName>,
            ) -> Result<(), Self::Err> {
                if !self.dev.has_func(MODULE_NAME, $Fwd) {
                    self.dev.load_ptx(PTX_SRC.into(), MODULE_NAME, &[$Fwd])?;
                }

                let adam_cfg = adam_config_to_cuda(cfg);
                let numel = param.shape.num_elements();

                let func = self.dev.get_func(MODULE_NAME, $Fwd).unwrap();
                let cfg = LaunchConfig::for_num_elems(numel as u32);
                let params = (
                    adam_cfg,                          // const AdamConfig cfg,
                    numel,                             // const size_t numel,
                    <$TypeName>::from_i32(t).unwrap(), // const float t,
                    Arc::make_mut(&mut param.data),    // float* param,
                    Arc::make_mut(&mut moment1.data),  // float* moment1,
                    Arc::make_mut(&mut moment2.data),  // float* moment2,
                    grad.data.as_ref(),                // const float* grad
                );
                unsafe { func.launch_async(cfg, params) }?;
                Ok(())
            }
        }
    };
}

impl_adam!(f32, "adam_update_f32");
impl_adam!(f64, "adam_update_f64");
