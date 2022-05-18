use crate::prelude::*;

// TODO abstract these all together somehow

pub trait Array: std::ops::IndexMut<usize, Output = Self::Element> {
    const SIZE: usize;
    type Element;
}

impl<const M: usize> Array for [f32; M] {
    const SIZE: usize = M;
    type Element = f32;
}

impl<T: Array, const M: usize> Array for [T; M] {
    const SIZE: usize = M;
    type Element = T;
}

pub fn broadcast_outer_add<Lhs, Rhs>(lhs: Lhs, rhs: &Rhs) -> Lhs
where
    Lhs: Tensor,
    Rhs: 'static + Tensor<TapeHolder = NoTape>,
    Rhs::Array: Array,
    Lhs::Array: Array<Element = Rhs::Array>,
    Lhs::Device: Device<Lhs::Array> + Device<Rhs::Array>,
{
    let mut result: Box<Lhs::Array> = Lhs::Device::zeros();
    for i in 0..Lhs::Array::SIZE {
        Lhs::Device::zip_map_into(&lhs.data()[i], rhs.data(), &mut result[i], |x, y| x + y);
    }

    let mut lhs_deriv = Lhs::Device::map(lhs.data(), |_| 1.0);
    let rhs_deriv = Lhs::Device::map(rhs.data(), |_| 1.0);

    let result = Lhs::NoTape::new_boxed(result);
    let (lhs, mut tape_holder) = lhs.split_tape_holder();
    let _rhs = rhs.phantom();
    let _result = result.phantom();
    tape_holder.add_operation(move |tape| {
        Lhs::Device::mul_assign(lhs_deriv.as_mut(), tape.ref_gradient(&_result));
        Lhs::Device::add_assign(tape.mut_gradient(&lhs), lhs_deriv.as_ref());

        for i in 0..Lhs::Array::SIZE {
            let d_grad_rhs = Lhs::Device::mul(rhs_deriv.as_ref(), &tape.ref_gradient(&_result)[i]);
            Lhs::Device::add_assign(tape.mut_gradient(&_rhs), d_grad_rhs.as_ref());
        }
    });
    result.with_tape_holder(tape_holder)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_outer_add() {
        let a: Tensor2D<3, 5> = Tensor2D::ones();
        let b: Tensor1D<5> = Tensor1D::ones();
        let r = broadcast_outer_add(a.trace(), &b);
        assert_eq!(r.data(), &[[2.0; 5]; 3]);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&a), &[[1.0 / 15.0; 5]; 3]);
        assert_eq!(gradients.ref_gradient(&b), &[0.20000002; 5]);
    }
}
