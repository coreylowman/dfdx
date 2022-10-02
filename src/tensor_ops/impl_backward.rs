use crate::prelude::*;

/// Runs backprop algorithm with all operations contained in the tape that `t` has.
///
/// This function takes ownership of `t` and returns [Gradients].
///
/// Note that `t` is required to have [OwnedTape], which means it currently owns the [crate::gradients::GradientTape].
pub fn backward(t: Tensor0D<OwnedTape>) -> Gradients {
    let (t, mut tape) = t.split_tape();
    tape.add_backward_op(move |grads| {
        Cpu::fill(grads.mut_gradient(&t), &mut |v| *v = 1.0);
    });
    tape.0.execute()
}

impl Tensor0D<OwnedTape> {
    pub fn backward(self) -> Gradients {
        backward(self)
    }
}
