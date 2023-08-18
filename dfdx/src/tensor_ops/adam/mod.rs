mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use crate::{
    shapes::{Dtype, Shape},
    tensor::{Storage, Tensor},
};

use super::WeightDecay;

/// Configuration of hyperparameters for [crate::optim::Adam].
///
/// Changing all default parameters:
/// ```rust
/// # use dfdx::{prelude::*, optim::*};
/// AdamConfig {
///     lr: 1e-2,
///     betas: [0.1, 0.2],
///     eps: 1e-6,
///     weight_decay: Some(WeightDecay::L2(1e-1)),
/// };
/// ```
#[derive(Debug, Clone, Copy)]
pub struct AdamConfig {
    /// Learning rate. Defaults to `1e-3`.
    pub lr: f64,

    /// Betas from Adam paper. Defaults to `[0.9, 0.999]`.
    pub betas: [f64; 2],

    /// Epsilon for numerical stability. Defaults to `1e-8`.
    pub eps: f64,

    /// Optional weight decay. Defaults to `None`.
    pub weight_decay: Option<WeightDecay>,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            betas: [0.9, 0.999],
            eps: 1e-8,
            weight_decay: None,
        }
    }
}

pub trait AdamKernel<E: Dtype>: Storage<E> {
    fn adam_kernel(
        &self,
        t: i32,
        cfg: &AdamConfig,
        param: &mut Self::Vec,
        moment1: &mut Self::Vec,
        moment2: &mut Self::Vec,
        grad: &Self::Vec,
    ) -> Result<(), Self::Err>;
}

impl AdamConfig {
    pub fn try_update<S: Shape, E: Dtype, D: AdamKernel<E>>(
        &self,
        t: i32,
        param: &mut Tensor<S, E, D>,
        moment1: &mut D::Vec,
        moment2: &mut D::Vec,
        grad: &D::Vec,
    ) -> Result<(), D::Err> {
        param.device.adam_kernel(
            t,
            self,
            std::sync::Arc::make_mut(&mut param.data),
            moment1,
            moment2,
            grad,
        )
    }
}
