use crate::prelude::*;

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
        let loss_value = *loss.data();
        let mut gradients = backward(loss);
        // gradients.scale(self.lr);
        // TODO
        (loss_value, gradients)
    }
}
