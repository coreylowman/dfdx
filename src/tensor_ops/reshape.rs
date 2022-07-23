use super::utils::move_tape_and_add_backward_op;
use crate::prelude::*;

/// Reshapes `t` into a differently shaped tensor. `T` and `R` must have the same
/// number of elements.
///
/// # Example
/// ```ignore
/// # use dfdx::prelude::*;
/// let t = Tensor1D::new([1.0, 2.0, 3.0, 4.0]);
/// let r: Tensor2D<2, 2> = reshape(t);
/// assert_eq!(r.data(), &[[1.0, 2.0], [3.0, 4.0]]);
/// ```
pub fn reshape<T, R>(t: T) -> R
where
    T: Tensor<Dtype = f32>,
    R: Tensor<Tape = T::Tape, Dtype = f32>,
    ConstEq<{ T::Array::NUM_ELEMENTS }, { R::Array::NUM_ELEMENTS }>: ConstTrue,
    ConstEq<{ R::Array::NUM_ELEMENTS }, { T::Array::NUM_ELEMENTS }>: ConstTrue,
{
    let mut result: R::NoTape = R::NoTape::zeros();
    copy(t.data(), result.mut_data());
    move_tape_and_add_backward_op(t, result, move |mut t, result, grads| {
        let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);
        copy(result_grad, t.mut_data());
        T::Device::add(t_grad, t.data());
    })
}

fn copy<Lhs: CountElements, Rhs: CountElements<Dtype = Lhs::Dtype>>(lhs: &Lhs, rhs: &mut Rhs)
where
    ConstEq<{ Lhs::NUM_ELEMENTS }, { Rhs::NUM_ELEMENTS }>: ConstTrue,
{
    let l = lhs.ref_first_elem() as *const Lhs::Dtype;
    let r = rhs.mut_first_elem() as *mut Lhs::Dtype;
    unsafe {
        std::ptr::copy_nonoverlapping(l, r, Lhs::NUM_ELEMENTS);
    }
}

pub trait ConstTrue {}

pub struct ConstEq<const A: usize, const B: usize>;
impl<const N: usize> ConstTrue for ConstEq<N, N> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_0d_reshape() {
        let a = Tensor0D::new(std::f32::consts::PI);
        let b: Tensor1D<1> = reshape(a.duplicate());
        assert_eq!(b.data(), &[std::f32::consts::PI]);

        let c: Tensor2D<1, 1> = reshape(a.duplicate());
        assert_eq!(c.data(), &[[std::f32::consts::PI]]);
    }

    #[test]
    fn test_valid_reshapes() {
        let _: Tensor1D<8> = reshape(Tensor2D::<2, 4>::zeros());
        let _: Tensor2D<2, 4> = reshape(Tensor3D::<2, 2, 2>::zeros());
        let _: Tensor3D<2, 2, 2> = reshape(Tensor2D::<2, 4>::zeros());
        let _: Tensor2D<3, 3> = reshape(Tensor1D::<9>::zeros());
    }

    #[test]
    fn test_1d_reshape() {
        let a = Tensor1D::new([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let b: Tensor2D<2, 3, OwnedTape> = reshape(a.trace());
        assert_eq!(b.data(), &[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);
        let gradients = b.exp().mean().backward();
        assert_eq!(
            gradients.ref_gradient(&a),
            &[0.18419516, 0.20356713, 0.22497648, 0.24863747, 0.2747869, 0.3036865]
        )
    }
}
