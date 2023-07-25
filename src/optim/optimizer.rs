use crate::{
    shapes::{Dtype, Shape, Unit},
    tensor::{Gradients, Storage, Tensor, UniqueId},
};

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
pub trait Optimizer<M, D: Storage<E>, E: Dtype> {
    /// Updates all of `module`'s parameters using `gradients`.
    ///
    /// Requires a `&mut self` because the optimizer may change some internally
    /// tracked values.
    fn update(
        &mut self,
        module: &mut M,
        gradients: &Gradients<E, D>,
    ) -> Result<(), OptimizerUpdateError<D::Err>>;
}

/// Holds [UniqueId] of tensors that were missing gradients during
/// update, and therefore are unused
#[derive(Debug, Default)]
pub struct UnusedTensors {
    pub ids: std::vec::Vec<UniqueId>,
}

impl UnusedTensors {
    /// Adds a single unnammed parameter
    pub fn add<S: Shape, E: Unit, D: Storage<E>, T>(&mut self, t: &Tensor<S, E, D, T>) {
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
pub enum OptimizerUpdateError<Err> {
    UnusedParams(UnusedTensors),
    DeviceError(Err),
}

impl<Err: std::fmt::Display> std::fmt::Display for OptimizerUpdateError<Err> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnusedParams(unused) => write!(f, "Unused tensors: {unused:?}"),
            Self::DeviceError(err) => write!(f, "{err}"),
        }
    }
}

#[cfg(feature = "std")]
impl<Err: std::fmt::Debug + std::fmt::Display> std::error::Error for OptimizerUpdateError<Err> {}

#[allow(clippy::from_over_into)]
impl<Err> Into<Result<(), OptimizerUpdateError<Err>>> for UnusedTensors {
    fn into(self) -> Result<(), OptimizerUpdateError<Err>> {
        if self.is_empty() {
            Ok(())
        } else {
            Err(OptimizerUpdateError::UnusedParams(self))
        }
    }
}
