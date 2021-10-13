use super::optimizer::Optimizer;
use crate::nn::module::Module;
use crate::tensor::Tensor;
use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

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
pub struct Sgd<M, I, O> {
    pub cfg: SgdConfig,
    pub module: M,
    marker: PhantomData<(I, O)>,
}

impl<M, I, O> Sgd<M, I, O> {
    pub fn new(cfg: SgdConfig, module: M) -> Self {
        Self {
            cfg,
            module,
            marker: PhantomData,
        }
    }
}

impl<M, I, O> Deref for Sgd<M, I, O> {
    type Target = M;

    fn deref(&self) -> &Self::Target {
        &self.module
    }
}

impl<M, I, O> DerefMut for Sgd<M, I, O> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.module
    }
}

impl<M, I, O> Optimizer<M, I, O> for Sgd<M, I, O>
where
    M: Module<I, O>,
    I: Tensor,
    O: Tensor,
{
    fn step<T: Tensor>(&mut self, loss: &mut T) {
        let mut tape = loss.backward().unwrap();
        tape.scale(self.cfg.lr);
        self.update(&tape);
    }
}
