use crate::arrays::{Dtype, Rank0};
use crate::gradients::{Gradients, OwnedTape, Tape};
use crate::tensor::{DeviceStorage, OneFillStorage, Tensor};

/// Runs backprop algorithm with all operations contained in the tape that `t` has.
///
/// This function takes ownership of `t` and returns [Gradients].
pub trait TryBackward<D: DeviceStorage>: Sized {
    fn backward(self) -> Gradients<D> {
        self.try_backward().unwrap()
    }
    fn try_backward(self) -> Result<Gradients<D>, D::Err>;
}

impl<E: Dtype, D: DeviceStorage + OneFillStorage<E>> TryBackward<D>
    for Tensor<Rank0, E, D, OwnedTape<D>>
{
    fn try_backward(self) -> Result<Gradients<D>, D::Err> {
        let (t, mut tape) = self.split_tape();
        tape.add_backward_op(move |grads| {
            let g = grads.get_mut(&t)?;
            t.device.try_fill_with_ones(g)?;
            Ok(())
        });
        tape.0.execute()
    }
}
