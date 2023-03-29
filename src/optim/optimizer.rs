use crate::{
    shapes::{Dtype, Shape, Unit},
    tensor::{DeviceStorage, Gradients, HasErr, Tensor, UniqueId},
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

/// Used to communicate the "WeightDecay" enum to cuda kernels
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub(super) enum WeightDecayType {
    None,
    L2,
    Decoupled,
}

#[cfg(feature = "cuda")]
pub(super) fn weight_decay_to_cuda<E: Default>(wd: Option<WeightDecay<E>>) -> (WeightDecayType, E) {
    match wd {
        None => (WeightDecayType::None, Default::default()),
        Some(WeightDecay::L2(x)) => (WeightDecayType::L2, x),
        Some(WeightDecay::Decoupled(x)) => (WeightDecayType::Decoupled, x),
    }
}

/// Momentum used for [super::Sgd] and others
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Momentum<E> {
    /// Momentum that is applied to the velocity of a parameter directly.
    Classic(E),

    /// Momentum that is applied to both velocity and gradients. See [super::Sgd] nesterov paper for more.
    Nesterov(E),
}

/// Used to communicate the "Momentum" enum to cuda kernels
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub(super) enum MomentumType {
    None,
    Classic,
    Nesterov,
}

#[cfg(feature = "cuda")]
pub(super) fn momentum_to_cuda<E: Default>(wd: Option<Momentum<E>>) -> (MomentumType, E) {
    match wd {
        None => (MomentumType::None, Default::default()),
        Some(Momentum::Classic(x)) => (MomentumType::Classic, x),
        Some(Momentum::Nesterov(x)) => (MomentumType::Nesterov, x),
    }
}

/// All optimizers must implement the update function, which takes a `M`
/// and updates all of its parameters.
///
/// # Notes
///
/// 2. Update takes ownership of [Gradients], so update cannot be called
/// with the same gradients object.
///
/// 3. Optimizer itself is generic over M, not the update method. This means a single optimizer object
/// can only work on objects of type `M`. This also requires you to specify the model up front for the optimizer.
pub trait Optimizer<M, D: DeviceStorage<E>, E: Dtype> {
    /// Updates all of `module`'s parameters using `gradients`.
    ///
    /// Requires a `&mut self` because the optimizer may change some internally
    /// tracked values.
    fn update(
        &mut self,
        module: &mut M,
        gradients: &Gradients<E, D>,
    ) -> Result<(), OptimizerUpdateError<D>>;
}

/// Holds [UniqueId] of tensors that were missing gradients during
/// update, and therefore are unused
#[derive(Debug, Default)]
pub struct UnusedTensors {
    pub ids: std::vec::Vec<UniqueId>,
}

impl UnusedTensors {
    /// Adds a single unnammed parameter
    pub fn add<S: Shape, E: Unit, D: DeviceStorage<E>, T>(&mut self, t: &Tensor<S, E, D, T>) {
        self.ids.push(t.id);
    }

    /// Returns `true` if there are no missing gradients present
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn clear(&mut self) {
        self.ids.clear();
    }
}

/// An error indicating that a parameter was not used in gradient
/// computation, and was therefore not present in [Gradients]
/// during an update.
#[derive(Debug)]
pub enum OptimizerUpdateError<D: HasErr> {
    UnusedParams(UnusedTensors),
    DeviceError(D::Err),
}

impl<D: HasErr> std::fmt::Display for OptimizerUpdateError<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnusedParams(unused) => write!(f, "Unused tensors: {unused:?}"),
            Self::DeviceError(err) => write!(f, "{err}"),
        }
    }
}

#[cfg(feature = "std")]
impl<D: HasErr + std::fmt::Debug> std::error::Error for OptimizerUpdateError<D> {}

#[allow(clippy::from_over_into)]
impl<D: HasErr> Into<Result<(), OptimizerUpdateError<D>>> for UnusedTensors {
    fn into(self) -> Result<(), OptimizerUpdateError<D>> {
        if self.is_empty() {
            Ok(())
        } else {
            Err(OptimizerUpdateError::UnusedParams(self))
        }
    }
}
