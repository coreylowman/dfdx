use crate::prelude::*;

/// Broadcasts the last dimension of `rhs` to make it the same size of `lhs`.
///
/// The last dim ([Device::sum_last_dim()], and [Tensor::LastDimeReduced]) reduced version of a tensor from another tensor.
/// So the size of Rhs is smaller than Lhs.
///
/// Examples:
/// ```rust
/// use dfdx::prelude::*;
/// let a = Tensor2D::new([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
/// let b = Tensor1D::new([3.0, 4.0]);
/// let r = broadcast_inner_sub(a, &b);
/// assert_eq!(r.data(), &[[-3.0, -2.0, -1.0], [-1.0, 0.0, 1.0]]);
/// ```
pub fn broadcast_inner_sub<Lhs: Tensor<Dtype = f32>>(
    lhs: Lhs,
    rhs: &<Lhs::LastDimReduced as Tensor>::NoTape,
) -> Lhs {
    let result = Lhs::NoTape::new_boxed(Lhs::Device::sub(lhs.data(), rhs.data()));

    let _rhs = rhs.phantom();
    let _result = result.phantom();
    let (mut lhs, mut tape_holder) = lhs.split_tape_holder();
    tape_holder.add_backward_op(move |tape| {
        let result_grad = tape.ref_gradient(&_result);

        Lhs::Device::zip_map_assign(lhs.mut_data(), result_grad, &mut |l, r| *l = *r);

        // this is reduce_inner(result_grad * rhs_deriv, x + y), where rhs_deriv = -1.
        let d_grad_rhs = Lhs::Device::reduce_last_dim(result_grad, |x, y| x + y);

        Lhs::Device::add_assign(tape.mut_gradient(&lhs), lhs.data());

        //NOTE: sub_assign here to account for negative sign from rhs_deriv
        <Lhs::LastDimReduced as HasDevice>::Device::sub_assign(
            tape.mut_gradient(&_rhs),
            d_grad_rhs.as_ref(),
        );
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
