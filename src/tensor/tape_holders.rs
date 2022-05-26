pub use super::*;
use crate::{gradients::GradientTape, prelude::Gradients};

/// Contains a boxed [GradientTape]. When [TapeHolder::add_backward_op] is called,
/// this function passes the operation directly to [GradientTape].
#[derive(Default, Debug)]
pub struct WithTape(pub(crate) Box<GradientTape>);

/// Contains nothing. When [TapeHolder::add_backward_op] is called, this function does nothing.
#[derive(Default, Debug, Clone, Copy)]
pub struct NoTape;

/// Something that can add a gradient operation to [GradientTape].
pub trait TapeHolder {
    fn add_backward_op<F: 'static + FnOnce(&mut Gradients)>(&mut self, operation: F);
}

impl TapeHolder for WithTape {
    fn add_backward_op<F: 'static + FnOnce(&mut Gradients)>(&mut self, operation: F) {
        self.0.add_backward_op(operation)
    }
}

impl TapeHolder for NoTape {
    fn add_backward_op<F: 'static + FnOnce(&mut Gradients)>(&mut self, _operation: F) {}
}
