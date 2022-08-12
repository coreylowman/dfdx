use super::utils::move_tape_and_add_backward_binop;
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
pub(crate) fn binary_map<
    T: Tensor<Dtype = f32>,
    F: FnMut(&f32, &f32) -> f32,
    Dfdx: FnMut(&f32, &f32) -> f32,
    Dfdy: FnMut(&f32, &f32) -> f32,
>(
    mut lhs: T,
    rhs: &T::NoTape,
    f: F,
    dfdx: Dfdx,
    dfdy: Dfdy,
) -> T {
    let mut result = T::NoTape::zeros();
    let mut rhs_deriv: Box<T::Array> = T::Device::zeros();

    // Clone rhs.data() into rhs_deriv
    rhs_deriv.as_mut().clone_from(rhs.data());

    // compute result & derivatives
    let (o, l, r) = (result.mut_data(), lhs.mut_data(), rhs_deriv.as_mut());
    f_and_dfs::<T::Array, T::Device, F, Dfdx, Dfdy>(o, l, r, f, dfdx, dfdy);

    move_tape_and_add_backward_binop(lhs, rhs, result, move |lhs, rhs, result, grads| {
        let (lhs_grad, result_grad) = grads.mut_and_ref(&lhs, &result);
        T::Device::addmul(lhs_grad, lhs.data(), result_grad);

        let (rhs_grad, result_grad) = grads.mut_and_ref(&rhs, &result);
        T::Device::addmul(rhs_grad, rhs_deriv.as_ref(), result_grad);
    })
}

fn f_and_dfs<
    T: CountElements,
    Device: ForEachElement<T>,
    F: FnMut(&T::Dtype, &T::Dtype) -> T::Dtype,
    Dfdx: FnMut(&T::Dtype, &T::Dtype) -> T::Dtype,
    Dfdy: FnMut(&T::Dtype, &T::Dtype) -> T::Dtype,
>(
    out: &mut T,
    lhs: &mut T,
    rhs: &mut T,
    mut f: F,
    mut dfdx: Dfdx,
    mut dfdy: Dfdy,
) {
    Device::foreach_mmm(out, lhs, rhs, &mut |o, l, r| {
        *o = f(l, r);
        let dx = dfdx(l, r);
        *r = dfdy(l, r);
        *l = dx;
    });
}
