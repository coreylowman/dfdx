use super::utils::move_tape_and_add_backward_op;
use crate::prelude::*;

pub trait Reshape<T> {
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

tensor_impl!(Tensor0D, [], Tensor0D, [], (1), (1));
tensor_impl!(Tensor0D, [], Tensor1D, [M], (1), (M));
tensor_impl!(Tensor0D, [], Tensor2D, [M, N], (1), (M * N));
tensor_impl!(Tensor0D, [], Tensor3D, [M, N, O], (1), (M * N * O));
tensor_impl!(Tensor0D, [], Tensor4D, [M, N, O, P], (1), (M * N * O * P));

tensor_impl!(Tensor1D, [A], Tensor0D, [], (A), (1));
tensor_impl!(Tensor1D, [A], Tensor1D, [M], (A), (M));
tensor_impl!(Tensor1D, [A], Tensor2D, [M, N], (A), (M * N));
tensor_impl!(Tensor1D, [A], Tensor3D, [M, N, O], (A), (M * N * O));
tensor_impl!(Tensor1D, [A], Tensor4D, [M, N, O, P], (A), (M * N * O * P));

tensor_impl!(Tensor2D, [A, B], Tensor0D, [], (A * B), (1));
tensor_impl!(Tensor2D, [A, B], Tensor1D, [M], (A * B), (M));
tensor_impl!(Tensor2D, [A, B], Tensor2D, [M, N], (A * B), (M * N));
tensor_impl!(Tensor2D, [A, B], Tensor3D, [M, N, O], (A * B), (M * N * O));
tensor_impl!(
    Tensor2D,
    [A, B],
    Tensor4D,
    [M, N, O, P],
    (A * B),
    (M * N * O * P)
);

tensor_impl!(Tensor3D, [A, B, C], Tensor0D, [], (A * B * C), (1));
tensor_impl!(Tensor3D, [A, B, C], Tensor1D, [M], (A * B * C), (M));
tensor_impl!(Tensor3D, [A, B, C], Tensor2D, [M, N], (A * B * C), (M * N));
tensor_impl!(
    Tensor3D,
    [A, B, C],
    Tensor3D,
    [M, N, O],
    (A * B * C),
    (M * N * O)
);
tensor_impl!(
    Tensor3D,
    [A, B, C],
    Tensor4D,
    [M, N, O, P],
    (A * B * C),
    (M * N * O * P)
);

tensor_impl!(Tensor4D, [A, B, C, D], Tensor0D, [], (A * B * C * D), (1));
tensor_impl!(Tensor4D, [A, B, C, D], Tensor1D, [M], (A * B * C * D), (M));
tensor_impl!(
    Tensor4D,
    [A, B, C, D],
    Tensor2D,
    [M, N],
    (A * B * C * D),
    (M * N)
);
tensor_impl!(
    Tensor4D,
    [A, B, C, D],
    Tensor3D,
    [M, N, O],
    (A * B * C * D),
    (M * N * O)
);
tensor_impl!(
    Tensor4D,
    [A, B, C, D],
    Tensor4D,
    [M, N, O, P],
    (A * B * C * D),
    (M * N * O * P)
);

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

/// THIS FUNCTION DOES NOT CHECK IF ARRAY LENGTHS ARE EQUAL
unsafe fn copy_unsafe<Lhs: CountElements, Rhs: CountElements<Dtype = Lhs::Dtype>>(
    lhs: &Lhs,
    rhs: &mut Rhs,
) {
    let l = lhs.ref_first_elem() as *const Lhs::Dtype;
    let r = rhs.mut_first_elem() as *mut Lhs::Dtype;
    std::ptr::copy_nonoverlapping(l, r, Lhs::NUM_ELEMENTS);
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
        let gradients = b.exp().mean().backward();
        assert_eq!(
            gradients.ref_gradient(&a),
            &[0.18419516, 0.20356713, 0.22497648, 0.24863747, 0.2747869, 0.3036865]
        )
    }
}
