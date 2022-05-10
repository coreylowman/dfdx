use crate::prelude::{CanUpdateWithGradients, Gradients};

/// All optimizers must implement the update function, which takes an object
/// that implements [CanUpdateWithGradients], and calls [CanUpdateWithGradients::update].
///
/// Note that [CanUpdateWithGradients] requires an object that implements [GradientProvider].
/// A common implementation involves implementing both [Optimizer] and [GradientProvider]
/// on one struct, and passing self to [CanUpdateWithGradients::update]. See [Sgd] for an example
/// of implementing this trait.
///
/// Note that update takes ownership of [gradients]: Gradients, so update cannot be called
/// with the same gradients object.
pub trait Optimizer {
    fn update<M: CanUpdateWithGradients>(&mut self, module: &mut M, gradients: Gradients);
}
