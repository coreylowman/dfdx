//! Some utilities for moving tape between tensors & adding backward op during that movement.
//! These are provided because:
//!
//! 1. .split_tape() & put_tape() are very repetitive
//! 2. Creating .phantom() for un-owned data is repetitive and error prone
//! 3. Forces a more standard way in all the operations of doing the above.
//! 4. You can't really separate these operations since they are very inter-dependent. So it makes
//!    sense to have a single unit for doing it.

use crate::devices::{AllocateZeros, Device, ForEachElement};
use crate::gradients::{Gradients, Tape};
use crate::prelude::*;

/// `f(t)`. Applies a function `f` to every element of the [Tensor]. The derivative
/// `df` must also be provided.
///
/// This is primarily used to implement standard functions such as [relu()], [exp()], etc.
/// But users can also implement their own activations with this.
pub(crate) fn map<T: Tensor<Dtype = f32>, F, Df>(t: T, f: F, mut df: Df) -> T
where
    F: 'static + FnMut(&f32) -> f32,
    Df: 'static + FnMut(&f32) -> f32,
{
    let result = T::NoTape::new_boxed(T::Device::map(t.data(), f));
    move_tape_and_add_backward_op(t, result, move |t, result, grads| {
        let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);
        T::Device::foreach_mrr(t_grad, t.data(), result_grad, &mut |g, t, r| {
            *g += df(t) * r;
        });
    })
}

/// Same as [map()], but calls `df` with the result of `f(x)`. This can potentially remove an allocation.
pub(crate) fn map_df_uses_fx<T: Tensor<Dtype = f32>, F, Df>(mut t: T, mut f: F, mut df: Df) -> T
where
    F: FnMut(&f32) -> f32,
    Df: 'static + FnMut(&f32) -> f32,
{
    T::Device::foreach_m(t.mut_data(), &mut |x| *x = f(x)); // clones if there is more than 1 reference to t
    let (t, mut tape) = t.split_tape();
    let result = t.clone(); // will always a new reference to t, not start a new one
    let phantom_result = result.phantom();
    tape.add_backward_op(move |grads| {
        let (t_grad, result_grad) = grads.mut_and_ref(&t, &phantom_result);
        T::Device::foreach_mrr(t_grad, t.data(), result_grad, &mut |g, fx, r| {
            *g += df(fx) * r;
        });
    });
    result.put_tape(tape)
}

/// Applies a binary function `f`, it's partial wrt. x `dfdx`, and its partial wrt. y `dfdy`
/// to a pair of [Tensor]s `lhs` and `rhs.
///
/// This is primarily used to implement [add()], [sub()], [mul()], and [div()].
pub(crate) fn binary_map<
    T: Tensor<Dtype = f32>,
    F: FnMut(&f32, &f32) -> f32,
    Dfdx: FnMut(&f32, &f32) -> f32,
    Dfdy: FnMut(&f32, &f32) -> f32,
>(
    mut lhs: T,
    rhs: &T::NoTape,
    mut f: F,
    mut dfdx: Dfdx,
    mut dfdy: Dfdy,
) -> T {
    let mut result = T::NoTape::zeros();
    let mut rhs_deriv: Box<T::Array> = T::Device::zeros();

    // Clone rhs.data() into rhs_deriv
    rhs_deriv.as_mut().clone_from(rhs.data());

    // compute result & derivatives
    T::Device::foreach_mmm(
        result.mut_data(),
        lhs.mut_data(),
        rhs_deriv.as_mut(),
        &mut |o, l, r| {
            *o = f(l, r);
            let dx = dfdx(l, r);
            *r = dfdy(l, r);
            *l = dx;
        },
    );

    move_tape_and_add_backward_binop(lhs, rhs, result, move |lhs, rhs, result, grads| {
        let (lhs_grad, result_grad) = grads.mut_and_ref(&lhs, &result);
        T::Device::addmul(lhs_grad, lhs.data(), result_grad);

        let (rhs_grad, result_grad) = grads.mut_and_ref(&rhs, &result);
        T::Device::addmul(rhs_grad, rhs_deriv.as_ref(), result_grad);
    })
}

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
