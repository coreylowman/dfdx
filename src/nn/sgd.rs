use super::traits::Optimizer;
use crate::gradients::OnGradientTape;
use crate::tensor::Tensor;
use std::ops::{Deref, DerefMut};

#[derive(Debug)]
pub struct SgdConfig {
    pub lr: f32,
}

impl Default for SgdConfig {
    fn default() -> Self {
        Self { lr: 1e-2 }
    }
}

#[derive(Default, Debug)]
pub struct Sgd<M> {
    pub cfg: SgdConfig,
    pub module: M,
}

impl<M> Sgd<M>
where
    M: Default,
{
    pub fn with_config(cfg: SgdConfig) -> Self {
        Self {
            cfg,
            module: Default::default(),
        }
    }
}

impl<M> Deref for Sgd<M> {
    type Target = M;
    fn deref(&self) -> &Self::Target {
        &self.module
    }
}

impl<M> DerefMut for Sgd<M> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.module
    }
}

impl<M> Optimizer<M> for Sgd<M>
where
    M: OnGradientTape,
{
    fn step<T: Tensor>(&mut self, loss: &mut T) {
        let mut tape = loss.backward().unwrap();
        tape.scale(self.cfg.lr);
        self.module.update(&tape);
    }
}
