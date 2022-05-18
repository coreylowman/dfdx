use crate::prelude::*;

pub fn broadcast_inner_sub<Lhs, Rhs>(lhs: Lhs, rhs: &Rhs) -> Lhs
where
    Lhs: Tensor,
    Rhs: 'static + Tensor<TapeHolder = NoTape>,
    Lhs::Device: Device<Lhs::Array>
        + Device<Rhs::Array>
        + ReduceInnerElements<Lhs::Array, Output = Rhs::Array>
        + ZipMapElements<Lhs::Array, Rhs::Array>,
{
    let result = Lhs::NoTape::new_boxed(Lhs::Device::sub(lhs.data(), rhs.data()));
    let mut lhs_deriv = Lhs::Device::map(lhs.data(), |_| 1.0);
    let rhs_deriv = Lhs::Device::map(rhs.data(), |_| -1.0);
    let _rhs = rhs.phantom();
    let _result = result.phantom();
    let (lhs, mut tape_holder) = lhs.split_tape_holder();
    tape_holder.add_operation(move |tape| {
        Lhs::Device::mul_assign(lhs_deriv.as_mut(), tape.ref_gradient(&_result));
        Lhs::Device::add_assign(tape.mut_gradient(&lhs), lhs_deriv.as_ref());

        let d_grad_rhs = Lhs::Device::reduce_inner(
            &Lhs::Device::mul(tape.ref_gradient(&_result), rhs_deriv.as_ref()),
            |x, y| x + y,
        );
        Lhs::Device::add_assign(tape.mut_gradient(&_rhs), d_grad_rhs.as_ref());
    });
    result.with_tape_holder(tape_holder)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_sub_1d() {
        let a: Tensor1D<3> = Tensor1D::new([1.0, 2.0, 3.0]);
        let b = Tensor0D::new(1.0);
        let r = broadcast_inner_sub(a.trace(), &b);
        assert_eq!(r.data(), &[0.0, 1.0, 2.0]);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&a), &[1.0 / 3.0; 3]);
        assert_eq!(gradients.ref_gradient(&b), &-1.0);
    }

    #[test]
    fn test_broadcast_sub_2d() {
        let a: Tensor2D<2, 3> = Tensor2D::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let b: Tensor1D<2> = Tensor1D::new([1.0, 2.0]);
        let r = broadcast_inner_sub(a.trace(), &b);
        assert_eq!(r.data(), &[[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]]);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&a), &[[1.0 / 6.0; 3]; 2]);
        assert_eq!(gradients.ref_gradient(&b), &[-0.5; 2]);
    }
}
