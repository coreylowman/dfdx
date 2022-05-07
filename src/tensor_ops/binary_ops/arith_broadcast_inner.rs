use crate::prelude::*;

pub fn broadcast_inner_sub<Lhs, Rhs>(lhs: Lhs, rhs: &Rhs) -> Lhs
where
    Lhs: Tensor,
    Rhs: 'static + Tensor<TapeHolder = NoTape>,
    Lhs::ArrayType: ReduceInnerElements<Output = Rhs::ArrayType>
        + ZipMapElements<Rhs::ArrayType>
        + ZipMapElements<Lhs::ArrayType>,
{
    let result = Lhs::NoTape::new(lhs.data().sub(rhs.data()));
    let (lhs, mut tape_holder) = lhs.split_tape_holder();
    let lhs_deriv = lhs.data().map_elems(|_| 1.0);
    let rhs_deriv = rhs.data().map_elems(|_| -1.0);
    let _rhs = rhs.phantom();
    let _result = result.phantom();
    tape_holder.add_operation(move |tape| {
        let d_grad_lhs = lhs_deriv.mul(tape.ref_gradient(&_result));
        tape.mut_gradient(&lhs).add_assign(&d_grad_lhs);

        let d_grad_rhs = tape
            .ref_gradient(&_result)
            .mul(&rhs_deriv)
            .reduce_inner(|x, y| x + y);
        tape.mut_gradient(&_rhs).add_assign(&d_grad_rhs);
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
