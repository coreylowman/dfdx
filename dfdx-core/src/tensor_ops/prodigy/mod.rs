mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use crate::{
    shapes::{Dtype, Shape},
    tensor::{Error, Storage, Tensor},
};

/// Configuration of hyperparameters for Prodigy.
///
/// Changing some default parameters:
/// ```rust
/// # use dfdx_core::prelude::*;
/// ProdigyConfig {
///     lr: 1e-4,
///     weight_decay: Some(WeightDecay::L2(1e-1)),
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ProdigyConfig {
    /// Learning rate adjustment parameter.
    /// Increases or decreases the Prodigy learning rate.
    ///
    /// Defaults to `1.0`.
    pub lr: f64,

    /// Betas coefficients used for computing running averages of gradient and its square.
    ///
    /// Defaults to `[0.9, 0.999]`.
    pub betas: [f64; 2],

    /// Coefficients for computing the Prodidy stepsize using running averages.
    /// If set to `None`, uses the value of square root of beta2 (ie. betas[1]).
    ///
    /// Defaults to `None`.
    pub beta3: Option<f64>,

    /// Term added to the denominator outside of the root operation to improve numerical stability.
    ///
    /// Defaults to `1e-8`.
    pub eps: f64,

    /// Optional weight decay.
    ///
    /// Defaults to `None`.
    pub weight_decay: Option<super::WeightDecay>,

    /// Turn on Adam's bias correction.
    ///
    /// Defaults to `false`.
    pub use_bias_correction: bool,

    /// Remove lr from the denominator of D estimate to avoid issues during warm-up stage.
    ///
    /// Defaults to `false`.
    pub safeguard_warmup: bool,

    /// Initial D estimate for D-adaptation. Rarely needs changing.
    ///
    /// Defaults to `1e-6`.
    pub d0: f64,

    /// Coefficient in the expression for the estimate of d.
    /// Values such as `0.5` and `2.0` typically work as well for a smaller or higher learning rate, respectively.
    /// Changing this parameter is the preferred way to tune the optimizer.
    ///
    /// Defaults to `1.0`.
    pub d_coef: f64,

    /// Prevent the D estimate from growing faster than this multiplicative rate.
    /// Use infinite for unrestricted. Values like 1.02 give a kind of learning
    /// rate warmup effect.
    ///
    /// Defaults to `f64::INFINITY`.
    pub growth_rate: f64,
}

impl Default for ProdigyConfig {
    fn default() -> Self {
        Self {
            lr: 1.0,
            betas: [0.9, 0.999],
            beta3: None,
            eps: 1e-8,
            weight_decay: None,
            use_bias_correction: false,
            safeguard_warmup: false,
            d0: 1e-6,
            d_coef: 1.0,
            growth_rate: f64::INFINITY,
        }
    }
}

pub trait ProdigyKernel<E: Dtype>: Storage<E> {
    #[allow(clippy::too_many_arguments)]
    fn prodigy_kernel(
        &self,
        k: i32,
        d: &mut f64,
        d_max: &mut f64,
        d_numerator: &mut f64,
        cfg: &ProdigyConfig,
        param: &mut Self::Vec,
        s: &mut Self::Vec,
        p0: &mut Self::Vec,
        p0b: &mut Self::Vec,
        moment1: &mut Self::Vec,
        moment2: &mut Self::Vec,
        grad: &Self::Vec,
    ) -> Result<(), Error>;
}

impl ProdigyConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn try_update<S: Shape, E: Dtype, D: ProdigyKernel<E>>(
        &self,
        k: i32,
        d: &mut f64,
        d_max: &mut f64,
        d_numerator: &mut f64,
        param: &mut Tensor<S, E, D>,
        s: &mut D::Vec,
        p0: &mut D::Vec,
        p0b: &mut D::Vec,
        moment1: &mut D::Vec,
        moment2: &mut D::Vec,
        grad: &D::Vec,
    ) -> Result<(), crate::tensor::Error> {
        param.device.prodigy_kernel(
            k,
            d,
            d_max,
            d_numerator,
            self,
            std::sync::Arc::make_mut(&mut param.data),
            s,
            p0,
            p0b,
            moment1,
            moment2,
            grad,
        )
    }
}
