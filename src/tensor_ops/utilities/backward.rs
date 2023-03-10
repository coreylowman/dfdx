use crate::shapes::{Dtype, Rank0};
use crate::tensor::*;

/// Runs backprop algorithm with all operations contained in the tape that `t` has.
///
/// This function takes ownership of `self` and returns [Gradients].
pub trait Backward<E: Dtype, D: DeviceStorage>: HasErr {
    /// Runs backprop
    fn backward(self) -> Gradients<E, D> {
        self.try_backward().unwrap()
    }
    /// Fallible version of [Backward::backward]
    fn try_backward(self) -> Result<Gradients<E, D>, Self::Err>;
}

impl<E: Dtype, D: OneFillStorage<E>> Backward<E, D> for Tensor<Rank0, E, D, OwnedTape<E, D>> {
    fn try_backward(self) -> Result<Gradients<E, D>, Self::Err> {
        let (t, mut tape) = self.split_tape();
        tape.add_backward_op(move |grads| t.device.try_fill_with_ones(grads.get_mut(&t)));
        let mut grads = tape.execute()?;
        grads.drop_non_leafs();
        Ok(grads)
    }
}
