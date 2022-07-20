use crate::prelude::{CanUpdateWithGradients, Gradients, UnusedParamsError};

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
pub trait Optimizer<M: CanUpdateWithGradients> {
    /// Updates all of `module`'s parameters using `gradients`.
    ///
    /// Requires a `&mut self` because the optimizer may change some internally
    /// tracked values.
    fn update(&mut self, module: &mut M, gradients: Gradients) -> Result<(), UnusedParamsError>;
}
