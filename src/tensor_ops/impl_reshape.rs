use super::utils::move_tape_and_add_backward_op;
use crate::prelude::*;

/// **Requires Nightly** Reshape `Self` into `T`.
pub trait Reshape<T> {
    /// Reshape `self` into `T`.
    fn reshape(self) -> T;
}

macro_rules! tensor_impl {
    ($src_ty:ident, [$($SrcVs:tt),*], $dst_ty:ident, [$($DstVs:tt),*], $assert_lhs:tt, $assert_rhs:tt) => {
impl<$(const $SrcVs: usize, )* $(const $DstVs: usize, )* H: Tape> Reshape<$dst_ty<$($DstVs, )* H>> for $src_ty<$($SrcVs, )* H>
where
    Assert<{ $assert_lhs == $assert_rhs }>: ConstTrue,
{
    fn reshape(self) -> $dst_ty<$($DstVs, )* H> {
        unsafe { reshape(self) }
    }
}
    };
}

macro_rules! impl_all_reshapes {
    ($src_ty:ident, [$($SrcVs:tt),*], $assert_lhs:tt) => {
        tensor_impl!($src_ty, [$($SrcVs),*], Tensor0D, [], $assert_lhs, (1));
        tensor_impl!($src_ty, [$($SrcVs),*], Tensor1D, [M], $assert_lhs, (M));
        tensor_impl!($src_ty, [$($SrcVs),*], Tensor2D, [M, N], $assert_lhs, (M * N));
        tensor_impl!($src_ty, [$($SrcVs),*], Tensor3D, [M, N, O], $assert_lhs, (M * N * O));
        tensor_impl!($src_ty, [$($SrcVs),*], Tensor4D, [M, N, O, P], $assert_lhs, (M * N * O * P));
    };
}

impl_all_reshapes!(Tensor0D, [], (1));
impl_all_reshapes!(Tensor1D, [A], (A));
impl_all_reshapes!(Tensor2D, [A, B], (A * B));
impl_all_reshapes!(Tensor3D, [A, B, C], (A * B * C));
impl_all_reshapes!(Tensor4D, [A, B, C, D], (A * B * C * D));

/// Reshapes `T` into `R`'s shape. This is unsafe because there are no compile
/// time guaruntees that `T` and `R` have the same number of elements.
unsafe fn reshape<T, R>(t: T) -> R
where
    T: Tensor<Dtype = f32>,
    R: Tensor<Dtype = f32, Tape = T::Tape>,
{
    let mut result = R::NoTape::zeros();
    copy_unsafe(t.data(), result.mut_data());
    move_tape_and_add_backward_op(t, result, move |mut t, result, grads| {
        let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);
        copy_unsafe(result_grad, t.mut_data());
        T::Device::add(t_grad, t.data());
    })
}

/// Unsafely copies `min(Lhs::NUM_ELEMENTS, Rhs::NUM_ELEMENTS` from `lhs` into `rhs`.
unsafe fn copy_unsafe<Lhs: CountElements, Rhs: CountElements<Dtype = Lhs::Dtype>>(
    lhs: &Lhs,
    rhs: &mut Rhs,
) {
    let l = lhs.ref_first_elem() as *const Lhs::Dtype;
    let r = rhs.mut_first_elem() as *mut Lhs::Dtype;
    std::ptr::copy_nonoverlapping(l, r, Lhs::NUM_ELEMENTS.min(Rhs::NUM_ELEMENTS));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_reshapes() {
        let _: Tensor1D<1> = Tensor0D::zeros().reshape();
        let _: Tensor1D<16> = Tensor1D::<16>::zeros().reshape();
        let _: Tensor1D<16> = Tensor2D::<2, 8>::zeros().reshape();
        let _: Tensor1D<16> = Tensor3D::<2, 2, 4>::zeros().reshape();
        let _: Tensor1D<16> = Tensor4D::<2, 2, 2, 2>::zeros().reshape();

        let _: Tensor2D<1, 1> = Tensor0D::zeros().reshape();
        let _: Tensor2D<2, 8> = Tensor1D::<16>::zeros().reshape();
        let _: Tensor2D<2, 8> = Tensor2D::<8, 2>::zeros().reshape();
        let _: Tensor2D<2, 8> = Tensor3D::<2, 2, 4>::zeros().reshape();
        let _: Tensor2D<2, 8> = Tensor4D::<2, 2, 2, 2>::zeros().reshape();

        let _: Tensor3D<1, 1, 1> = Tensor0D::zeros().reshape();
        let _: Tensor3D<2, 2, 4> = Tensor1D::<16>::zeros().reshape();
        let _: Tensor3D<2, 2, 4> = Tensor2D::<2, 8>::zeros().reshape();
        let _: Tensor3D<2, 2, 4> = Tensor3D::<4, 2, 2>::zeros().reshape();
        let _: Tensor3D<2, 2, 4> = Tensor4D::<2, 2, 2, 2>::zeros().reshape();

        let _: Tensor4D<1, 1, 1, 1> = Tensor0D::zeros().reshape();
        let _: Tensor4D<2, 2, 2, 2> = Tensor1D::<16>::zeros().reshape();
        let _: Tensor4D<2, 2, 2, 2> = Tensor2D::<2, 8>::zeros().reshape();
        let _: Tensor4D<2, 2, 2, 2> = Tensor3D::<4, 2, 2>::zeros().reshape();
        let _: Tensor4D<2, 2, 2, 2> = Tensor4D::<4, 1, 2, 2>::zeros().reshape();
    }

    #[test]
    fn test_1d_reshape() {
        let a = Tensor1D::new([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let b: Tensor2D<2, 3, OwnedTape> = a.trace().reshape();
        assert_eq!(b.data(), &[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);
        let gradients = backward(b.exp().mean());
        assert_eq!(
            gradients.ref_gradient(&a),
            &[0.18419516, 0.20356713, 0.22497648, 0.24863747, 0.2747869, 0.3036865]
        )
    }
}
