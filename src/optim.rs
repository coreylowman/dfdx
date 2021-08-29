use crate::traits::{Module, Optimizer, Tensor};
use std::ops::{Deref, DerefMut};

#[derive(Default, Debug)]
pub struct Sgd<M: Module> {
    lr: f32,
    module: M,
}

impl<M: Module> Sgd<M> {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            module: Default::default(),
        }
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
        let mut tape = *loss.backward();
        tape.scale(self.lr);
        self.update(&tape);
    }
}
