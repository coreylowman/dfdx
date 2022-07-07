//! Some utilities for moving tape between tensors & adding backward op during that movement.
//! These are provided because:
//!
//! 1. .split_tape() & put_tape() are very repetitive
//! 2. Creating .phantom() for un-owned data is repetitive and error prone
//! 3. Forces a more standard way in all the operations of doing the above.
//! 4. You can't really separate these operations since they are very inter-dependent. So it makes
//!    sense to have a single unit for doing it.

use crate::prelude::*;

/// Moves tape from `inp` to `out`, and does `tape.add_backward_op()` with `f`
pub(super) fn move_tape_and_add_backward_op<Inp, Out, F>(
    inp: Inp,
    out: Out::NoTape,
    mut f: F,
) -> Out
where
    Inp: Tensor,
    Out: Tensor<Tape = Inp::Tape>,
    F: 'static + FnMut(Inp::NoTape, PhantomTensor<Out::NoTape>, &mut Gradients),
{
    let phantom_out = out.phantom();
    let (t, mut tape) = inp.split_tape();
    tape.add_backward_op(move |grads| f(t, phantom_out, grads));
    out.put_tape(tape)
}

/// Moves tape from `lhs` to `out`, and does `tape.add_backward_op()` with `f`
pub(super) fn move_tape_and_add_backward_binop<Lhs, Rhs, Out, F>(
    lhs: Lhs,
    rhs: &Rhs,
    out: Out::NoTape,
    mut f: F,
) -> Out
where
    Lhs: Tensor,
    Rhs: 'static + Tensor,
    Out: Tensor<Tape = Lhs::Tape>,
    F: 'static + FnMut(Lhs::NoTape, PhantomTensor<Rhs>, PhantomTensor<Out::NoTape>, &mut Gradients),
{
    let phantom_rhs = rhs.phantom();
    let phantom_out = out.phantom();
    let (lhs, mut tape) = lhs.split_tape();
    tape.add_backward_op(move |grads| f(lhs, phantom_rhs, phantom_out, grads));
    out.put_tape(tape)
}
