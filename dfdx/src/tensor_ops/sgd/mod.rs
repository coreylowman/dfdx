mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use crate::{
    shapes::{Dtype, Shape},
    tensor::{Storage, Tensor},
};

use super::optim::{Momentum, WeightDecay};

/// Configuration of hyperparameters for [crate::optim::Sgd].
///
/// Using different learning rate:
/// ```rust
/// # use dfdx::{prelude::*, optim::*};
/// SgdConfig {
///     lr: 1e-1,
///     momentum: None,
///     weight_decay: None,
/// };
/// ```
///
/// Using classic momentum:
/// ```rust
/// # use dfdx::{prelude::*, optim::*};
/// SgdConfig {
///     lr: 1e-2,
///     momentum: Some(Momentum::Classic(0.5)),
///     weight_decay: None,
/// };
/// ```
///
/// Using nesterov momentum:
/// ```rust
/// # use dfdx::{prelude::*, optim::*};
/// SgdConfig {
///     lr: 1e-3,
///     momentum: Some(Momentum::Nesterov(0.25)),
///     weight_decay: None,
/// };
/// ```
///
/// Using L2 weight decay:
/// ```rust
/// # use dfdx::{prelude::*, optim::*};
/// SgdConfig {
///     lr: 1e-3,
///     momentum: None,
///     weight_decay: Some(WeightDecay::L2(1e-2)),
/// };
/// ```
///
/// Using decoupled weight decay:
/// ```rust
/// # use dfdx::{prelude::*, optim::*};
/// SgdConfig {
///     lr: 1e-3,
///     momentum: None,
///     weight_decay: Some(WeightDecay::Decoupled(1e-2)),
/// };
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SgdConfig {
    /// Learning rate. Defaults to `1e-2`
    pub lr: f64,

    /// Optional momentum. Defaults to `None`.
    pub momentum: Option<Momentum>,

    /// Optional weight decay. Defaults to `None`.
    pub weight_decay: Option<WeightDecay>,
}

impl Default for SgdConfig {
    fn default() -> Self {
        Self {
            lr: 1e-2,
            momentum: None,
            weight_decay: None,
        }
    }
}

pub trait SgdKernel<E: Dtype>: Storage<E> {
    fn sgd_kernel(
        &self,
        cfg: &SgdConfig,
        param: &mut Self::Vec,
        velocity: &mut Self::Vec,
        grad: &Self::Vec,
    ) -> Result<(), Self::Err>;
}

impl SgdConfig {
    /// Updates a single tensor using SGD.
    pub fn try_update<S: Shape, E: Dtype, D: SgdKernel<E>>(
        &self,
        param: &mut Tensor<S, E, D>,
        velocity: &mut D::Vec,
        grad: &D::Vec,
    ) -> Result<(), D::Err> {
        param.device.sgd_kernel(
            self,
            std::sync::Arc::make_mut(&mut param.data),
            velocity,
            grad,
        )
    }
}
