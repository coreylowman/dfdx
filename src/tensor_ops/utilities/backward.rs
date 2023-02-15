use crate::gradients::{Gradients, OwnedTape, Tape};
use crate::shapes::{Dtype, Rank0};
use crate::tensor::{HasErr, OneFillStorage, SplitTape, Tensor};

/// Runs backprop algorithm with all operations contained in the tape that `t` has.
///
/// This function takes ownership of `self` and returns [Gradients].
pub trait Backward: HasErr {
    /// Runs backprop
    fn backward(self) -> Gradients {
        self.try_backward().unwrap()
    }
    /// Fallible version of [Backward::backward]
    fn try_backward(self) -> Result<Gradients, Self::Err>;
}

impl<E: Dtype, D: OneFillStorage<E>> Backward for Tensor<Rank0, E, D, OwnedTape<D>> {
    fn try_backward(self) -> Result<Gradients, Self::Err> {
        let (t, mut tape) = self.split_tape();
        tape.add_backward_op(move |grads| grads.get_mut(&t).try_fill_with_ones());
        tape.0.execute()
    }
}
