use crate::prelude::*;

/// ReLU but maintains a small gradient if the input values are negative.
#[derive(Debug, Clone, Copy, CustomModule)]
pub struct LeakyReLU(pub f64);

impl Default for LeakyReLU {
    fn default() -> Self {
        Self(0.05)
    }
}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<S, E, D, T>> for LeakyReLU {
    type Output = Tensor<S, E, D, T>;
    fn try_forward(&self, x: Tensor<S, E, D, T>) -> Result<Self::Output, Error> {
        x.try_prelu(E::from_f64(self.0).unwrap())
    }
}
