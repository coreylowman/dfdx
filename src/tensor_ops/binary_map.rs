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
    use std::ops::Neg;

    pub fn f(x: &f32, y: &f32) -> f32 {
        x * y.recip()
    }
    pub fn dfdx(_x: &f32, y: &f32) -> f32 {
        y.recip()
    }
    pub fn dfdy(x: &f32, y: &f32) -> f32 {
        x.neg() * y.powi(2).recip()
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
pub(super) fn binary_map<T: Tensor<Dtype = f32>, F, Dfdx, Dfdy>(
    lhs: T,
    rhs: &T::NoTape,
    f: F,
    mut dfdx: Dfdx,
    dfdy: Dfdy,
) -> T
where
    F: FnMut(&f32, &f32) -> f32,
    Dfdx: FnMut(&f32, &f32) -> f32,
    Dfdy: FnMut(&f32, &f32) -> f32,
{
    let result = T::NoTape::new_boxed(T::Device::zip_map(lhs.data(), rhs.data(), f));
    let (mut lhs, mut tape) = lhs.split_tape();
    let mut rhs_deriv: Box<T::Array> = T::Device::zip_map(lhs.data(), rhs.data(), dfdy);
    T::Device::zip_map_assign(lhs.mut_data(), rhs.data(), &mut |l, r| *l = dfdx(l, r));
    let _rhs = rhs.phantom();
    let _result = result.phantom();
    tape.add_backward_op(move |grads| {
        let result_grad: &T::Array = grads.ref_gradient(&_result);
        T::Device::mul_assign(lhs.mut_data(), result_grad);
        T::Device::mul_assign(rhs_deriv.as_mut(), result_grad);
        T::Device::add_assign(grads.mut_gradient(&lhs), lhs.data());
        T::Device::add_assign(grads.mut_gradient(&_rhs), rhs_deriv.as_ref());
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
pub(super) fn binary_map_broadcast_rhs_first<const M: usize, Lhs, Rhs, F, Dfdx, Dfdy>(
    lhs: Lhs,
    rhs: &Rhs,
    mut f: F,
    mut dfdx: Dfdx,
    mut dfdy: Dfdy,
) -> Lhs
where
    Rhs: 'static + Tensor<Dtype = f32, Tape = NoTape>,
    Lhs: Tensor<Dtype = f32, Array = [Rhs::Array; M]>,
    F: FnMut(&f32, &f32) -> f32,
    Dfdx: FnMut(&f32, &f32) -> f32,
    Dfdy: FnMut(&f32, &f32) -> f32,
    Lhs::Device: Device<Lhs::Array> + Device<Rhs::Array>,
{
    let result = Lhs::NoTape::new_boxed(Lhs::Device::broadcast_rhs_first(
        lhs.data(),
        rhs.data(),
        &mut f,
    ));

    let (mut lhs, mut tape) = lhs.split_tape();
    let _rhs = rhs.phantom();
    let _result = result.phantom();

    // calculate derivatives
    let mut rhs_deriv: Box<Lhs::Array> =
        Lhs::Device::broadcast_rhs_first(lhs.data(), rhs.data(), &mut dfdy);
    Lhs::Device::broadcast_rhs_first_assign(lhs.mut_data(), rhs.data(), &mut |l, r| {
        *l = dfdx(l, r)
    });

    tape.add_backward_op(move |grads| {
        let result_grad: &Lhs::Array = grads.ref_gradient(&_result);
        // chain rule
        Lhs::Device::mul_assign(lhs.mut_data(), result_grad);
        Lhs::Device::mul_assign(rhs_deriv.as_mut(), result_grad);

        // sum first dimension
        let mut d_grad_rhs: Box<Rhs::Array> = Lhs::Device::zeros();
        for i in 0..M {
            Rhs::Device::add_assign(d_grad_rhs.as_mut(), &rhs_deriv[i]);
        }

        // gather gradients
        Lhs::Device::add_assign(grads.mut_gradient(&lhs), lhs.data());
        Rhs::Device::add_assign(grads.mut_gradient(&_rhs), d_grad_rhs.as_ref());
    });
    result.put_tape(tape)
}

/// Applies a binary function `f`, it's partial wrt. x `dfdx`, and its partial wrt. y `dfdy`
/// to a pair of [Tensor]s `lhs` and `rhs. Note that `rhs` has it's last dimension reduced,
/// so therefore it's last dimension is broadcasted to `lhs`'s last dimension.
///
/// This is primarily used to implement [add_broadcast_rhs_last()],
/// [sub_broadcast_rhs_last()], [mul_broadcast_rhs_last()], and [div_broadcast_rhs_last()].
pub(super) fn binary_map_broadcast_rhs_last<T: Tensor<Dtype = f32>, F, Dfdx, Dfdy>(
    lhs: T,
    mut rhs: <T::LastDimReduced as Tensor>::NoTape,
    f: F,
    mut dfdx: Dfdx,
    dfdy: Dfdy,
) -> T
where
    F: FnMut(&f32, &f32) -> f32,
    Dfdx: FnMut(&f32, &f32) -> f32,
    Dfdy: FnMut(&f32, &f32) -> f32,
{
    let result = T::NoTape::new_boxed(T::Device::zip_map(lhs.data(), rhs.data(), f));
    let (mut lhs, mut tape) = lhs.split_tape();
    let mut rhs_deriv: Box<T::Array> = T::Device::zip_map(lhs.data(), rhs.data(), dfdy);
    T::Device::zip_map_assign(lhs.mut_data(), rhs.data(), &mut |l, r| *l = dfdx(l, r));
    let _result = result.phantom();
    tape.add_backward_op(move |grads| {
        let result_grad: &T::Array = grads.ref_gradient(&_result);
        T::Device::mul_assign(lhs.mut_data(), result_grad);
        T::Device::mul_assign(rhs_deriv.as_mut(), result_grad);
        T::Device::add_assign(grads.mut_gradient(&lhs), lhs.data());
        T::Device::reduce_last_dim_into(rhs_deriv.as_ref(), rhs.mut_data(), &mut |x, y| x + y);
        <T::LastDimReduced as HasDevice>::Device::add_assign(grads.mut_gradient(&rhs), rhs.data());
    });
    result.put_tape(tape)
}
