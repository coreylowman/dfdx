use crate::traits::{Module, Optimizer, Tensor};
use std::ops::{Deref, DerefMut};

#[derive(Debug)]
pub struct SgdConfig {
    lr: f32,
}

impl Default for SgdConfig {
    fn default() -> Self {
        Self { lr: 1e-2 }
    }
}

#[derive(Default, Debug)]
pub struct Sgd<M: Module> {
    cfg: SgdConfig,
    module: M,
}

impl<M: Module> Sgd<M> {
    pub fn new(cfg: SgdConfig, module: M) -> Self {
        Self { cfg, module }
    }
}

impl<M: Module> Deref for Sgd<M> {
    type Target = M;

    fn deref(&self) -> &Self::Target {
        &self.module
    }
}

impl<M: Module> DerefMut for Sgd<M> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.module
    }
}

impl<M: Module> Optimizer<M> for Sgd<M> {
    fn step<T: Tensor>(&mut self, loss: &mut T) {
        let mut tape = loss.backward().unwrap();
        tape.scale(self.cfg.lr);
        self.update(&tape);
    }
}
