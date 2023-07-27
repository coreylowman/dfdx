use crate::shapes::Rank0;
use crate::tensor::*;
use std::sync::{Arc, Mutex};

/// Runs backprop algorithm with all operations contained in the tape that `t` has.
///
/// This function takes ownership of `self` and returns [Gradients].
pub trait Backward<E, D: Storage<E>>: HasErr {
    /// Runs backprop
    fn backward(self) -> Gradients<E, D> {
        self.try_backward().unwrap()
    }
    /// Fallible version of [Backward::backward]
    fn try_backward(self) -> Result<Gradients<E, D>, Self::Err>;
}

impl<E: 'static + Clone, D: OneFillStorage<E>> Backward<E, D>
    for Tensor<Rank0, E, D, OwnedTape<E, D>>
{
    fn try_backward(self) -> Result<Gradients<E, D>, Self::Err> {
        let (t, mut tape) = self.split_tape();
        let t_ghost = t.ghost();
        tape.add_backward_op(move |grads| {
            grads.try_alloc_for(&t_ghost)?;
            t.device.try_fill_with_ones(grads.get_mut(&t_ghost))
        });
        let mut grads = tape.execute()?;
        grads.drop_non_leafs();
        Ok(grads)
    }
}

impl<E: 'static + Clone, D: OneFillStorage<E>> Backward<E, D>
    for Tensor<Rank0, E, D, Arc<Mutex<OwnedTape<E, D>>>>
{
    fn try_backward(self) -> Result<Gradients<E, D>, Self::Err> {
        let (t, tape) = self.split_tape();
        let t_ghost = t.ghost();
        let mut tape = tape.lock().unwrap();
        tape.add_backward_op(move |grads| {
            grads.try_alloc_for(&t_ghost)?;
            t.device.try_fill_with_ones(grads.get_mut(&t_ghost))
        });
        let mut grads = tape.execute()?;
        grads.drop_non_leafs();
        Ok(grads)
    }
}
