use crate::prelude::*;

pub(super) mod add {
    pub fn f(x: &f32, y: &f32) -> f32 {
        x + y
    }
    pub fn dfdx(_x: &f32, _y: &f32) -> f32 {
        1.0
    }
    pub fn dfdy(_x: &f32, _y: &f32) -> f32 {
        1.0
    }
}

pub(super) mod sub {
    pub fn f(x: &f32, y: &f32) -> f32 {
        x - y
    }
    pub fn dfdx(_x: &f32, _y: &f32) -> f32 {
        1.0
    }
    pub fn dfdy(_x: &f32, _y: &f32) -> f32 {
        -1.0
    }
}

pub(super) mod mul {
    pub fn f(x: &f32, y: &f32) -> f32 {
        x * y
    }
    pub fn dfdx(_x: &f32, y: &f32) -> f32 {
        *y
    }
    pub fn dfdy(x: &f32, _y: &f32) -> f32 {
        *x
    }
}

pub(super) mod div {
    pub fn f(x: &f32, y: &f32) -> f32 {
        x * y.recip()
    }
    pub fn dfdx(_x: &f32, y: &f32) -> f32 {
        y.recip()
    }
    pub fn dfdy(x: &f32, y: &f32) -> f32 {
        (-x) * y.powi(2).recip()
    }
}

pub(super) mod minimum {
    pub fn f(x: &f32, y: &f32) -> f32 {
        x.min(*y)
    }
    pub fn dfdx(x: &f32, y: &f32) -> f32 {
        if x < y {
            1.0
        } else if x > y {
            0.0
        } else {
            0.5
        }
    }

    pub fn dfdy(x: &f32, y: &f32) -> f32 {
        if y < x {
            1.0
        } else if y > x {
            0.0
        } else {
            0.5
        }
    }
}

/// Applies a binary function `f`, it's partial wrt. x `dfdx`, and its partial wrt. y `dfdy`
/// to a pair of [Tensor]s `lhs` and `rhs.
///
/// This is primarily used to implement [add()], [sub()], [mul()], and [div()].
pub(super) fn binary_map<T: Tensor<Dtype = f32>>(
    lhs: T,
    rhs: &T::NoTape,
    f: fn(&f32, &f32) -> f32,
    dfdx: fn(&f32, &f32) -> f32,
    dfdy: fn(&f32, &f32) -> f32,
) -> T {
    let (mut lhs, mut tape) = lhs.split_tape();
    let mut result = T::NoTape::zeros();
    let mut rhs_deriv: Box<T::Array> = T::Device::zeros();

    // Clone rhs.data() into rhs_deriv
    rhs_deriv.as_mut().clone_from(rhs.data());

    // compute result & derivatives
    let (o, l, r) = (result.mut_data(), lhs.mut_data(), rhs_deriv.as_mut());
    f_and_dfs::<T::Array, T::Device>(o, l, r, f, dfdx, dfdy);

    let _rhs = rhs.phantom();
    let _result = result.phantom();
    tape.add_backward_op(move |grads| {
        let (lhs_grad, result_grad) = grads.mut_and_ref(&lhs, &_result);
        T::Device::addmul(lhs_grad, lhs.data(), result_grad);

        let (rhs_grad, result_grad) = grads.mut_and_ref(&_rhs, &_result);
        T::Device::addmul(rhs_grad, rhs_deriv.as_ref(), result_grad);
    });
    result.put_tape(tape)
}

/// Apply binary function `f` to `lhs` and `rhs`, where `rhs` is broadcasted `M` times to be the same shape as `lhs`.
/// `dfdx` and `dfdy` are the partial derivatives of f wrt. x and y respectively.
///
/// `f`, `dfdx`, and `dfdy` are all the same type.
///
/// Generics:
/// - `M`: The first dimension of `lhs`.
pub(super) fn binary_map_broadcast_rhs_first<const M: usize, Lhs, Rhs>(
    lhs: Lhs,
    rhs: &Rhs,
    f: fn(&f32, &f32) -> f32,
    dfdx: fn(&f32, &f32) -> f32,
    dfdy: fn(&f32, &f32) -> f32,
) -> Lhs
where
    Rhs: 'static + Tensor<Dtype = f32, Tape = NoTape>,
    Lhs: Tensor<Dtype = f32, Array = [Rhs::Array; M]>,
{
    let (mut lhs, mut tape) = lhs.split_tape();
    let mut result = Lhs::NoTape::zeros();
    let mut rhs_deriv: Box<Lhs::Array> = Lhs::Device::zeros();

    // clone rhs.data() into rhs_deriv
    for i in 0..M {
        rhs_deriv[i].clone_from(rhs.data());
    }

    // compute result & derivatives
    let (o, l, r) = (result.mut_data(), lhs.mut_data(), rhs_deriv.as_mut());
    f_and_dfs::<Lhs::Array, Lhs::Device>(o, l, r, f, dfdx, dfdy);

    let _rhs = rhs.phantom();
    let _result = result.phantom();
    tape.add_backward_op(move |grads| {
        let (lhs_grad, result_grad) = grads.mut_and_ref(&lhs, &_result);
        Lhs::Device::addmul(lhs_grad, lhs.data(), result_grad);

        let (rhs_grad, result_grad) = grads.mut_and_ref(&_rhs, &_result);
        for i in 0..M {
            Rhs::Device::addmul(rhs_grad, &rhs_deriv[i], &result_grad[i]);
        }
    });
    result.put_tape(tape)
}

/// Applies a binary function `f`, it's partial wrt. x `dfdx`, and its partial wrt. y `dfdy`
/// to a pair of [Tensor]s `lhs` and `rhs. Note that `rhs` has it's last dimension reduced,
/// so therefore it's last dimension is broadcasted to `lhs`'s last dimension.
///
/// This is primarily used to implement [add_broadcast_rhs_last()],
/// [sub_broadcast_rhs_last()], [mul_broadcast_rhs_last()], and [div_broadcast_rhs_last()].
pub(super) fn binary_map_broadcast_rhs_last<T: Tensor<Dtype = f32>>(
    lhs: T,
    rhs: <T::LastDimReduced as Tensor>::NoTape,
    f: fn(&f32, &f32) -> f32,
    dfdx: fn(&f32, &f32) -> f32,
    dfdy: fn(&f32, &f32) -> f32,
) -> T {
    let mut result = T::NoTape::zeros();
    let (mut lhs, mut tape) = lhs.split_tape();
    let mut rhs_deriv: Box<T::Array> = T::Device::zeros();

    // clone rhs.data() into rhs_deriv.
    T::Device::foreach_mb(rhs_deriv.as_mut(), Broadcast(rhs.data()), &mut |o, r| {
        *o = *r;
    });

    // compute result & derivatives at the same time
    let (o, l, r) = (result.mut_data(), lhs.mut_data(), rhs_deriv.as_mut());
    f_and_dfs::<T::Array, T::Device>(o, l, r, f, dfdx, dfdy);

    let _result = result.phantom();
    tape.add_backward_op(move |grads| {
        let (lhs_grad, result_grad) = grads.mut_and_ref(&lhs, &_result);
        T::Device::addmul(lhs_grad, lhs.data(), result_grad);

        let (rhs_grad, result_grad) = grads.mut_and_ref(&rhs, &_result);
        let rhs_grad = BroadcastMut(rhs_grad);
        T::Device::foreach_brr(rhs_grad, rhs_deriv.as_ref(), result_grad, &mut |g, d, r| {
            *g += d * r;
        });
    });
    result.put_tape(tape)
}

fn f_and_dfs<T: CountElements, Device: ForEachElement<T>>(
    out: &mut T,
    lhs: &mut T,
    rhs: &mut T,
    f: fn(&T::Dtype, &T::Dtype) -> T::Dtype,
    dfdx: fn(&T::Dtype, &T::Dtype) -> T::Dtype,
    dfdy: fn(&T::Dtype, &T::Dtype) -> T::Dtype,
) {
    Device::foreach_mmm(out, lhs, rhs, &mut |o, l, r| {
        *o = f(l, r);
        let dx = dfdx(l, r);
        *r = dfdy(l, r);
        *l = dx;
    });
}
