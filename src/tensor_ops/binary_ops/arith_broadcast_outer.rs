use crate::prelude::*;

// TODO abstract these all together somehow

pub fn broadcast_outer_add<Lhs, Rhs>(lhs: Lhs, rhs: &Rhs) -> Lhs
where
    Lhs: Tensor,
    Rhs: 'static + Tensor<TapeHolder = NoTape>,
    Rhs::Array: Array,
    Lhs::Array: Array<Element = Rhs::Array>,
    Cpu: Device<Lhs::Array> + Device<Rhs::Array>,
{
    let mut result: Box<Lhs::Array> = Cpu::zeros();
    for i in 0..Lhs::Array::SIZE {
        Cpu::zip_map_into(&lhs.data()[i], rhs.data(), &mut result[i], |x, y| x + y);
    }

    let lhs_deriv = Cpu::map(lhs.data(), |_| 1.0);
    let rhs_deriv = Cpu::map(rhs.data(), |_| 1.0);

    let result = Lhs::NoTape::new_boxed(result);
    let (lhs, mut tape_holder) = lhs.split_tape_holder();
    let _rhs = rhs.phantom();
    let _result = result.phantom();
    tape_holder.add_operation(move |tape| {
        let d_grad_lhs = Cpu::mul(lhs_deriv.as_ref(), tape.ref_gradient(&_result));
        Cpu::add_assign(tape.mut_gradient(&lhs), &d_grad_lhs);

        for i in 0..Lhs::Array::SIZE {
            let d_grad_rhs = Cpu::mul(rhs_deriv.as_ref(), &tape.ref_gradient(&_result)[i]);
            Cpu::add_assign(tape.mut_gradient(&_rhs), d_grad_rhs.as_ref());
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
