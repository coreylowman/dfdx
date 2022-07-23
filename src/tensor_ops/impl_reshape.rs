use crate::prelude::*;
use super::utils::move_tape_and_add_backward_op;

pub trait Reshape<T> {
    fn reshape(self) -> T;
}

macro_rules! tensor_impl {
    ($type1:ident, [$Vs1f:tt $(,$Vs1:tt)*], $type2:ident, [$Vs2f:tt $(,$Vs2:tt)*], $LEqStatement:tt, $REqStatement:tt) => {
impl<const $Vs1f: usize, $(const $Vs1: usize, )* const $Vs2f: usize, $(const $Vs2: usize, )*> Reshape<$type2<$Vs2f, $($Vs2, )*>> for $type1<$Vs1f, $($Vs1, )*>
where Assert<{$LEqStatement == $REqStatement}>: ConstTrue {
    fn reshape(self) -> $type2<$Vs2f, $($Vs2, )*> {
        let mut result: $type2<$Vs2f, $($Vs2, )* NoneTape> = $type2::zeros();
        copy_unsafe(self.data(), result.mut_data());
        move_tape_and_add_backward_op(self, result, move |mut t, result, grads| {
            let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);
            copy_unsafe(result_grad, t.mut_data());
            Cpu::add(t_grad, t.data());
        })
    }
}
    };
}

// 1D
tensor_impl!(Tensor1D, [A], Tensor2D, [B, C], (A), (B * C));
tensor_impl!(Tensor1D, [A], Tensor3D, [B, C, D], (A), (B * C * D));
tensor_impl!(Tensor1D, [A], Tensor4D, [B, C, D, E], (A), (B * C * D * E));
// 2D
tensor_impl!(Tensor2D, [A, B], Tensor1D, [C], (A * B), (C));
tensor_impl!(Tensor2D, [A, B], Tensor2D, [C, D], (A * B), (C * D));
tensor_impl!(Tensor2D, [A, B], Tensor3D, [C, D, E], (A * B), (C * D * E));
tensor_impl!(Tensor2D, [A, B], Tensor4D, [C, D, E, F], (A * B), (C * D * E * F));
// 3D
tensor_impl!(Tensor3D, [A, B, C], Tensor1D, [D], (A * B * C), (D));
tensor_impl!(Tensor3D, [A, B, C], Tensor2D, [D, E], (A * B * C), (D * E));
tensor_impl!(Tensor3D, [A, B, C], Tensor3D, [D, E, F], (A * B * C), (D * E * F));
tensor_impl!(Tensor3D, [A, B, C], Tensor4D, [D, E, F, G], (A * B * C), (D * E * F * G));
// 4D
tensor_impl!(Tensor4D, [A, B, C, D], Tensor1D, [E], (A * B * C * D), (E));
tensor_impl!(Tensor4D, [A, B, C, D], Tensor2D, [E, F], (A * B * C * D), (E * F));
tensor_impl!(Tensor4D, [A, B, C, D], Tensor3D, [E, F, G], (A * B * C * D), (E * F * G));
tensor_impl!(Tensor4D, [A, B, C, D], Tensor4D, [E, F, G, H], (A * B * C * D), (E * F * G * H));

/// THIS FUNCTION DOES NOT CHECK IF ARRAY LENGTHS ARE EQUAL
fn copy_unsafe<Lhs: CountElements, Rhs: CountElements<Dtype = Lhs::Dtype>>(lhs: &Lhs, rhs: &mut Rhs) {
    let l = lhs.ref_first_elem() as *const Lhs::Dtype;
    let r = rhs.mut_first_elem() as *mut Lhs::Dtype;
    unsafe {
        std::ptr::copy_nonoverlapping(l, r, Lhs::NUM_ELEMENTS);
    }
}