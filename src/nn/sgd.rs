use super::traits::Optimizer;
use crate::gradients::GradientTape;
use crate::prelude::{backward, IsShapedArray, Tensor0D, WithTape};

#[derive(Debug)]
pub struct Sgd {
    pub lr: f32,
}

impl Default for Sgd {
    fn default() -> Self {
        Self { lr: 1e-2 }
    }
}

impl Optimizer for Sgd {
    fn compute_gradients(&mut self, loss: Tensor0D<WithTape>) -> (f32, Box<GradientTape>) {
        let loss_value = loss.data()[()];
        let mut gradients = backward(loss);
        gradients.scale(self.lr);
        (loss_value, gradients)
    }
}
