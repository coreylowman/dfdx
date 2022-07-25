use super::utils::move_tape_and_add_backward_op;
use crate::prelude::*;

pub trait Reshape<T> {
    fn reshape(self) -> T;
}

macro_rules! tensor_impl {
    ($type1:ident, [$Vs1f:tt $(,$Vs1:tt)*], $type2:ident, [$Vs2f:tt $(,$Vs2:tt)*], $LEqStatement:tt, $REqStatement:tt) => {
impl<const $Vs1f: usize, $(const $Vs1: usize, )* const $Vs2f: usize, $(const $Vs2: usize, )* T: Tape> Reshape<$type2<$Vs2f, $($Vs2, )* T>> for $type1<$Vs1f, $($Vs1, )* T>
where Assert<{$LEqStatement == $REqStatement}>: ConstTrue {
    fn reshape(self) -> $type2<$Vs2f, $($Vs2, )* T> {
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

// 0D
impl<T: Tape> Reshape<Tensor1D<1, T>> for Tensor0D<T> {
    fn reshape(self) -> Tensor1D<1, T> {
        let mut result: Tensor1D<1, NoneTape> = Tensor1D::zeros();
        copy_unsafe(self.data(), result.mut_data());
        move_tape_and_add_backward_op(self, result, move |mut t, result, grads| {
            let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);
            copy_unsafe(result_grad, t.mut_data());
            Cpu::add(t_grad, t.data());
        })
    }
}

impl<T: Tape> Reshape<Tensor2D<1, 1, T>> for Tensor0D<T> {
    fn reshape(self) -> Tensor2D<1, 1, T> {
        let mut result: Tensor2D<1, 1, NoneTape> = Tensor2D::zeros();
        copy_unsafe(self.data(), result.mut_data());
        move_tape_and_add_backward_op(self, result, move |mut t, result, grads| {
            let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);
            copy_unsafe(result_grad, t.mut_data());
            Cpu::add(t_grad, t.data());
        })
    }
}

impl<T: Tape> Reshape<Tensor3D<1, 1, 1, T>> for Tensor0D<T> {
    fn reshape(self) -> Tensor3D<1, 1, 1, T> {
        let mut result: Tensor3D<1, 1, 1, NoneTape> = Tensor3D::zeros();
        copy_unsafe(self.data(), result.mut_data());
        move_tape_and_add_backward_op(self, result, move |mut t, result, grads| {
            let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);
            copy_unsafe(result_grad, t.mut_data());
            Cpu::add(t_grad, t.data());
        })
    }
}

impl<T: Tape> Reshape<Tensor4D<1, 1, 1, 1, T>> for Tensor0D<T> {
    fn reshape(self) -> Tensor4D<1, 1, 1, 1, T> {
        let mut result: Tensor4D<1, 1, 1, 1, NoneTape> = Tensor4D::zeros();
        copy_unsafe(self.data(), result.mut_data());
        move_tape_and_add_backward_op(self, result, move |mut t, result, grads| {
            let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);
            copy_unsafe(result_grad, t.mut_data());
            Cpu::add(t_grad, t.data());
        })
    }
}

impl<T: Tape> Reshape<Tensor0D<T>> for Tensor1D<1, T> {
    fn reshape(self) -> Tensor0D<T> {
        let mut result: Tensor0D = Tensor0D::zeros();
        copy_unsafe(self.data(), result.mut_data());
        move_tape_and_add_backward_op(self, result, move |mut t, result, grads| {
            let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);
            copy_unsafe(result_grad, t.mut_data());
            Cpu::add(t_grad, t.data());
        })
    }
}

impl<T: Tape> Reshape<Tensor0D<T>> for Tensor2D<1, 1, T> {
    fn reshape(self) -> Tensor0D<T> {
        let mut result: Tensor0D = Tensor0D::zeros();
        copy_unsafe(self.data(), result.mut_data());
        move_tape_and_add_backward_op(self, result, move |mut t, result, grads| {
            let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);
            copy_unsafe(result_grad, t.mut_data());
            Cpu::add(t_grad, t.data());
        })
    }
}

impl<T: Tape> Reshape<Tensor0D<T>> for Tensor3D<1, 1, 1, T> {
    fn reshape(self) -> Tensor0D<T> {
        let mut result: Tensor0D = Tensor0D::zeros();
        copy_unsafe(self.data(), result.mut_data());
        move_tape_and_add_backward_op(self, result, move |mut t, result, grads| {
            let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);
            copy_unsafe(result_grad, t.mut_data());
            Cpu::add(t_grad, t.data());
        })
    }
}

impl<T: Tape> Reshape<Tensor0D<T>> for Tensor4D<1, 1, 1, 1, T> {
    fn reshape(self) -> Tensor0D<T> {
        let mut result: Tensor0D = Tensor0D::zeros();
        copy_unsafe(self.data(), result.mut_data());
        move_tape_and_add_backward_op(self, result, move |mut t, result, grads| {
            let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);
            copy_unsafe(result_grad, t.mut_data());
            Cpu::add(t_grad, t.data());
        })
    }
}

// 1D
tensor_impl!(Tensor1D, [A], Tensor2D, [B, C], (A), (B * C));
tensor_impl!(Tensor1D, [A], Tensor3D, [B, C, D], (A), (B * C * D));
tensor_impl!(Tensor1D, [A], Tensor4D, [B, C, D, E], (A), (B * C * D * E));
// 2D
tensor_impl!(Tensor2D, [A, B], Tensor1D, [C], (A * B), (C));
tensor_impl!(Tensor2D, [A, B], Tensor2D, [C, D], (A * B), (C * D));
tensor_impl!(Tensor2D, [A, B], Tensor3D, [C, D, E], (A * B), (C * D * E));
tensor_impl!(
    Tensor2D,
    [A, B],
    Tensor4D,
    [C, D, E, F],
    (A * B),
    (C * D * E * F)
);
// 3D
tensor_impl!(Tensor3D, [A, B, C], Tensor1D, [D], (A * B * C), (D));
tensor_impl!(Tensor3D, [A, B, C], Tensor2D, [D, E], (A * B * C), (D * E));
tensor_impl!(
    Tensor3D,
    [A, B, C],
    Tensor3D,
    [D, E, F],
    (A * B * C),
    (D * E * F)
);
tensor_impl!(
    Tensor3D,
    [A, B, C],
    Tensor4D,
    [D, E, F, G],
    (A * B * C),
    (D * E * F * G)
);
// 4D
tensor_impl!(Tensor4D, [A, B, C, D], Tensor1D, [E], (A * B * C * D), (E));
tensor_impl!(
    Tensor4D,
    [A, B, C, D],
    Tensor2D,
    [E, F],
    (A * B * C * D),
    (E * F)
);
tensor_impl!(
    Tensor4D,
    [A, B, C, D],
    Tensor3D,
    [E, F, G],
    (A * B * C * D),
    (E * F * G)
);
tensor_impl!(
    Tensor4D,
    [A, B, C, D],
    Tensor4D,
    [E, F, G, H],
    (A * B * C * D),
    (E * F * G * H)
);

/// THIS FUNCTION DOES NOT CHECK IF ARRAY LENGTHS ARE EQUAL
fn copy_unsafe<Lhs: CountElements, Rhs: CountElements<Dtype = Lhs::Dtype>>(
    lhs: &Lhs,
    rhs: &mut Rhs,
) {
    let l = lhs.ref_first_elem() as *const Lhs::Dtype;
    let r = rhs.mut_first_elem() as *mut Lhs::Dtype;
    unsafe {
        std::ptr::copy_nonoverlapping(l, r, Lhs::NUM_ELEMENTS);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_0d_reshape() {
        let a = Tensor0D::new(std::f32::consts::PI);
        let b: Tensor1D<1> = a.duplicate().reshape();
        assert_eq!(b.data(), &[std::f32::consts::PI]);

        let c: Tensor2D<1, 1> = a.duplicate().reshape();
        assert_eq!(c.data(), &[[std::f32::consts::PI]]);
    }

    #[test]
    fn test_valid_reshapes() {
        let _: Tensor1D<8> = Tensor2D::<2, 4>::zeros().reshape();
        let _: Tensor2D<2, 4> = Tensor3D::<2, 2, 2>::zeros().reshape();
        let _: Tensor3D<2, 2, 2> = Tensor2D::<2, 4>::zeros().reshape();
        let _: Tensor2D<3, 3> = Tensor1D::<9>::zeros().reshape();
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
