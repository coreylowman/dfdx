use std::marker::PhantomData;

use dfdx::{
    shapes::{Dtype, Shape},
    tensor::{Gradients, Storage, Tensor, Tensorlike, UniqueId},
    tensor_ops::{AdamConfig, Device},
};

/// An implementation of the Adam optimizer from
/// [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
///
/// # Example Usage
/// ```rust
/// # use dfdx::prelude::*;
/// # use dfdx_nn::{*, optim::*};
/// # type Model = Tensor<Rank0, f32, Cpu>;
/// # let dev: Cpu = Default::default();
/// # let model: Model = dev.zeros();
/// let mut opt: Adam<Model, f32, Cpu> = Adam::new(&model, AdamConfig {
///     lr: 1e-2,
///     betas: [0.5, 0.25],
///     eps: 1e-6,
///     weight_decay: Some(WeightDecay::Decoupled(1e-2)),
/// });
/// ```
///
/// See module level documentation at [crate::optim] for examples of how to actually use an optimizer.
#[derive(Debug, Clone)]
pub struct Adam<M, E: Dtype, D: Storage<E>> {
    /// Hyperparameter configuration
    pub cfg: AdamConfig,

    t: i32,
    moment1: Gradients<E, D>,
    moment2: Gradients<E, D>,

    marker: PhantomData<*const M>,
}

impl<M, E: Dtype, D: Storage<E>> Adam<M, E, D> {
    /// Constructs using hyperparameters from `cfg`.
    pub fn new(_model: &M, cfg: AdamConfig) -> Self {
        Self {
            cfg,
            t: 0,
            moment1: Gradients::leaky(),
            moment2: Gradients::leaky(),
            marker: PhantomData,
        }
    }
}

impl<M, E: Dtype, D: Device<E>> crate::Optimizer<M, E, D> for Adam<M, E, D> {
    fn update_tensor<S: Shape>(
        &mut self,
        t: &mut Tensor<S, E, D>,
        gradients: &Gradients<E, D>,
        missing_params: &mut Vec<UniqueId>,
    ) -> Result<(), D::Err> {
        let g = gradients.get_ref_checked(t);
        match g {
            None => missing_params.push(t.id()),
            Some(g) => {
                let m_t = self.moment1.get_or_alloc_mut(t)?;
                let v_t = self.moment2.get_or_alloc_mut(t)?;
                self.cfg.try_update(self.t, t, m_t, v_t, g)?;
            }
        }
        Ok(())
    }

    fn update(
        &mut self,
        module: &mut M,
        gradients: &Gradients<E, D>,
    ) -> Result<(), dfdx_nn_core::OptimizerUpdateError<<D>::Err>>
    where
        M: dfdx_nn_core::UpdateParams<E, D>,
    {
        self.t = self.t.checked_add(1).unwrap();

        // NOTE: the rest of this is identical to default implementation of update.
        let mut missing_tensors = Vec::new();
        module
            .try_update_params(self, gradients, &mut missing_tensors)
            .map_err(crate::OptimizerUpdateError::DeviceError)?;
        if missing_tensors.is_empty() {
            Ok(())
        } else {
            Err(crate::OptimizerUpdateError::UnusedTensors(missing_tensors))
        }
    }
}
