use dfdx::{
    shapes::{Dtype, Shape},
    tensor::{Gradients, Storage, Tensor, Tensorlike, UniqueId},
    tensor_ops::{Device, SgdConfig},
};

/// Implementation of Stochastic Gradient Descent. Based on [pytorch's implementation](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)
///
/// Nesterov Momentum is implemented as described in
/// [On the importance of initialization and momentum in deep learning](https://proceedings.mlr.press/v28/sutskever13.html).
///
/// Weight decay is implemented as described in
/// [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
/// Both L2 weight_decay and decoupled weight_decay are available.
///
/// # Example Usage
///
/// ```rust
/// # use dfdx::{prelude::*, optim::*};
/// # let dev: Cpu = Default::default();
/// # type Model = Tensor<Rank0, f32, Cpu>;
/// # let mut model: Model = dev.zeros();
/// let mut opt: Sgd<Model, f32, Cpu> = Sgd::new(&model, SgdConfig {
///     lr: 1e-3,
///     momentum: Some(Momentum::Classic(0.5)),
///     weight_decay: Some(WeightDecay::L2(0.01)),
/// });
/// ```
///
/// See module level documentation at [crate::optim] for examples of how to actually use an optimizer.
#[derive(Debug, Clone)]
pub struct Sgd<M, E: Dtype, D: Storage<E>> {
    pub cfg: SgdConfig,
    velocity: Gradients<E, D>,
    module: std::marker::PhantomData<*const M>,
}

impl<M, E: Dtype, D: Storage<E>> Sgd<M, E, D> {
    pub fn new(_model: &M, cfg: SgdConfig) -> Self {
        Self {
            cfg,
            velocity: Gradients::leaky(),
            module: std::marker::PhantomData,
        }
    }
}

impl<M, E: Dtype, D: Device<E>> crate::Optimizer<M, E, D> for Sgd<M, E, D> {
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
                let v = self.velocity.get_or_alloc_mut(t)?;
                self.cfg.try_update(t, v, g)?;
            }
        }
        Ok(())
    }
}
