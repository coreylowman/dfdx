use crate::prelude::{CanUpdateWithGradients, Gradients};

pub trait Optimizer {
    fn update<M: CanUpdateWithGradients>(&mut self, module: &mut M, gradients: Gradients);
}
