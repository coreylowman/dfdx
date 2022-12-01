use crate::{
    arrays::{Dtype, Shape},
    gradients::Gradients,
    tensor::{DeviceStorage, Tensor},
    unique_id::{HasUniqueId, UniqueId},
};

/// L2 and decoupled regularization methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightDecay<E> {
    /// Weight decay applied to the gradients before any momentum updates. Equivalent to L2 regularization.
    L2(E),

    /// Weight decay applied after any momentum updates, without modifying the gradients.
    /// See [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
    Decoupled(E),
}

/// Momentum used for [Sgd]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Momentum<E> {
    /// Momentum that is applied to the velocity of a parameter directly.
    Classic(E),

    /// Momentum that is applied to both velocity and gradients. See [Sgd] nesterov paper for more.
    Nesterov(E),
}

/// All optimizers must implement the update function, which takes an object
/// that implements [CanUpdateWithGradients], and calls [CanUpdateWithGradients::update].
///
/// # Notes
///
/// 1. [CanUpdateWithGradients] requires an object that implements [crate::gradients::GradientProvider].
/// A common implementation involves implementing both [Optimizer] and [crate::gradients::GradientProvider]
/// on one struct, and passing self to [CanUpdateWithGradients::update]. See [super::Sgd] for an example
/// of implementing this trait.
///
/// 2. Update takes ownership of [Gradients], so update cannot be called
/// with the same gradients object.
///
/// 3. Optimizer itself is generic over M, not the update method. This means a single optimizer object
/// can only work on objects of type `M`. This also requires you to specify the model up front for the optimizer.
pub trait Optimizer<M, D: DeviceStorage> {
    /// Updates all of `module`'s parameters using `gradients`.
    ///
    /// Requires a `&mut self` because the optimizer may change some internally
    /// tracked values.
    fn update(
        &mut self,
        module: &mut M,
        gradients: Gradients<D>,
    ) -> Result<(), OptimizerUpdateError<D>>;
}

/// Represents something that can return a gradient for a given key.
///
/// This is very similar to what [Gradients] does, however the intention
/// is that any this object be passed to [CanUpdateWithGradients].
///
/// [Gradients] does **not** implement this, so you *have* to go through
/// an optimizer to update a [CanUpdateWithGradients]. Although it very easily
/// could.
///
/// See [crate::optim::Sgd] and [crate::optim::Adam] for examples on implementing this.
pub trait UpdateParams<D: DeviceStorage, E: Dtype> {
    /// Retrieves the data associated with `p` if there is any.
    /// This can modify `self`, for instance if velocities are calculated
    /// based on the associated data!
    fn update_param<S: Shape>(
        &mut self,
        p: &mut Tensor<S, E, D>,
        unused: &mut UnusedTensors,
    ) -> Result<(), D::Err>;
}

/// Holds [UniqueId] of tensors that were missing gradients during
/// [CanUpdateWithGradients::update()], and therefore are unused
#[derive(Debug, Default)]
pub struct UnusedTensors {
    pub ids: std::vec::Vec<UniqueId>,
}

impl UnusedTensors {
    /// Adds a single unnammed parameter
    pub fn add<T: HasUniqueId>(&mut self, t: &T) {
        self.ids.push(*t.id());
    }

    /// Returns `true` if there are no missing gradients present
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }
}

/// Represents something that can be updated with [GradientProvider].
///
/// Most implementations of this trait will have sub structs that also
/// implement [CanUpdateWithGradients].
pub trait CanUpdateWithGradients<D: DeviceStorage, E: Dtype>: Sized {
    /// Updates self given the [GradientProvider]. When any parameters that
    /// are NOT present in `G`, then this function should
    /// add the tensor's [UniqueId] to [UnusedTensors].
    fn update<U>(&mut self, updater: &mut U, unused: &mut UnusedTensors) -> Result<(), D::Err>
    where
        U: UpdateParams<D, E>;
}

impl<S: Shape, E: Dtype, D: DeviceStorage> CanUpdateWithGradients<D, E> for Tensor<S, E, D> {
    /// Subtracts the gradient for the tensor from [HasArrayData::mut_data].
    fn update<O: UpdateParams<D, E>>(
        &mut self,
        opt: &mut O,
        unused: &mut UnusedTensors,
    ) -> Result<(), D::Err> {
        opt.update_param(self, unused)
    }
}

/// An error indicating that a parameter was not used in gradient
/// computation, and was therefore not present in [Gradients]
/// while a [CanUpdateWithGradients] was trying to update it.
#[derive(Debug)]
pub enum OptimizerUpdateError<D: DeviceStorage> {
    UnusedParams(UnusedTensors),
    DeviceError(D::Err),
}

impl<D: DeviceStorage> std::fmt::Display for OptimizerUpdateError<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnusedParams(unused) => write!(f, "Unused tensors: {:?}", unused),
            Self::DeviceError(err) => write!(f, "{err}"),
        }
    }
}

#[cfg(feature = "std")]
impl<D: DeviceStorage + std::fmt::Debug> std::error::Error for OptimizerUpdateError<D> {}

#[allow(clippy::from_over_into)]
impl<D: DeviceStorage> Into<Result<(), OptimizerUpdateError<D>>> for UnusedTensors {
    fn into(self) -> Result<(), OptimizerUpdateError<D>> {
        if self.is_empty() {
            Ok(())
        } else {
            Err(OptimizerUpdateError::UnusedParams(self))
        }
    }
}
