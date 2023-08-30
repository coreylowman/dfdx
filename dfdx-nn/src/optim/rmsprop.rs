use std::marker::PhantomData;

use dfdx::{
    shapes::{Dtype, Shape},
    tensor::{Gradients, Storage, Tensor, Tensorlike, UniqueId},
    tensor_ops::{Device, RMSpropConfig},
};

/// RMSprop As described in [Hinton, 2012](http://www.cs.toronto.edu/%7Etijmen/csc321/slides/lecture_slides_lec6.pdf).
///
/// This implementation is based off of RMSprop from
/// [pytorch-image-models](https://github.com/rwightman/pytorch-image-models/blob/master/timm/optim/rmsprop_tf.py)
/// because the pytorch implementation has [some issues](https://github.com/pytorch/pytorch/issues/23796).
///
/// The main difference between the pytorch implementation is that [RMSpropConfig::eps] is added inside of the sqrt()
/// operation.
///
/// The `lr_in_momentum` option is not provided because it didn't seem to make a difference in testing.
///
/// # Example Usage
///
/// Constructing using new:
/// ```rust
/// # use dfdx::prelude::*;
/// # use dfdx_nn::{*, optim::*};
/// # type Model = Tensor<Rank0, f32, Cpu>;
/// # let dev: Cpu = Default::default();
/// # let model: Model = dev.zeros();
/// let rmsprop: RMSprop<Model, f32, Cpu> = RMSprop::new(&model, RMSpropConfig {
///     lr: 1e-3,
///     alpha: 0.5,
///     eps: 1e-8,
///     momentum: Some(0.5),
///     centered: false,
///     weight_decay: Some(WeightDecay::Decoupled(1e-1)),
/// });
#[derive(Debug, Clone)]
pub struct RMSprop<M, E: Dtype, D: Storage<E>> {
    /// Hyperparameter configuration
    pub cfg: RMSpropConfig,

    step: usize,
    momentums: Gradients<E, D>,
    square_avg: Gradients<E, D>,
    grad_avg: Gradients<E, D>,

    marker: PhantomData<*const M>,
}

impl<M, E: Dtype, D: Storage<E>> RMSprop<M, E, D> {
    /// Constructs using hyperparameters from `cfg`.
    pub fn new(_model: &M, cfg: RMSpropConfig) -> Self {
        Self {
            cfg,
            step: 0,
            momentums: Gradients::leaky(),
            square_avg: Gradients::leaky(),
            grad_avg: Gradients::leaky(),
            marker: PhantomData,
        }
    }
}

impl<M, E: Dtype, D: Device<E>> crate::Optimizer<M, E, D> for RMSprop<M, E, D> {
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
                let m = self.momentums.get_or_alloc_mut(t)?;
                let sa = self.square_avg.get_or_alloc_mut(t)?;
                let ga = self.grad_avg.get_or_alloc_mut(t)?;

                if self.step == 0 {
                    t.device().try_fill_with_ones(sa)?;
                }

                self.cfg.try_update(t, m, sa, ga, g)?;
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
        // NOTE: the rest of this is identical to default implementation of update.
        let mut missing_tensors = Vec::new();
        module
            .try_update_params(self, gradients, &mut missing_tensors)
            .map_err(crate::OptimizerUpdateError::DeviceError)?;
        let r = if missing_tensors.is_empty() {
            Ok(())
        } else {
            Err(crate::OptimizerUpdateError::UnusedTensors(missing_tensors))
        };
        self.step += 1;
        r
    }
}
