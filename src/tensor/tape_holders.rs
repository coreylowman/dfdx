pub use super::*;
use crate::gradients::GradientTape;

#[derive(Default, Debug)]
pub struct WithTape(pub(crate) Box<GradientTape>);

#[derive(Default, Debug)]
pub struct NoTape;

pub trait TapeHolder {
    fn add_operation<F: 'static + FnOnce(&mut GradientTape)>(&mut self, operation: F);
}

impl TapeHolder for WithTape {
    fn add_operation<F: 'static + FnOnce(&mut GradientTape)>(&mut self, operation: F) {
        self.0.add_operation(operation)
    }
}

impl TapeHolder for NoTape {
    fn add_operation<F: 'static + FnOnce(&mut GradientTape)>(&mut self, _operation: F) {}
}
