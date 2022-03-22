pub use super::*;
use crate::gradients::GradientTape;

#[derive(Default, Debug)]
pub struct WithTape(pub(crate) Box<GradientTape>);

#[derive(Default, Debug)]
pub struct NoTape;

pub trait TapeHolder {
    fn update_with<F: FnMut(&mut Box<GradientTape>)>(&mut self, update_fn: F);
}

impl TapeHolder for WithTape {
    fn update_with<F: FnMut(&mut Box<GradientTape>)>(&mut self, mut update_fn: F) {
        update_fn(&mut self.0)
    }
}

impl TapeHolder for NoTape {
    fn update_with<F: FnMut(&mut Box<GradientTape>)>(&mut self, _update_fn: F) {}
}
