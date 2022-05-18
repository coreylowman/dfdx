use crate::prelude::*;

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: TapeHolder> std::ops::Neg for $typename<$($Vs, )* H>
where
    Cpu: Device<<Self as HasArrayType>::Array>
{
    type Output = Self;
    fn neg(self) -> Self::Output {
        let result = <Self::Output as Tensor>::NoTape::new_boxed(Cpu::map(self.data(), |v| -v));
        let mut deriv = Cpu::map(self.data(), |_| -1.0);
        let (t, mut tape_holder) = self.split_tape_holder();
        let _result = result.phantom();
        tape_holder.add_operation(move |tape| {
            Cpu::mul_assign(deriv.as_mut(), tape.ref_gradient(&_result));
            Cpu::add_assign(tape.mut_gradient(&t), deriv.as_ref());
        });
        result.with_tape_holder(tape_holder)
    }
}
    };
}

tensor_impl!(Tensor0D, []);
tensor_impl!(Tensor1D, [M]);
tensor_impl!(Tensor2D, [M, N]);
tensor_impl!(Tensor3D, [M, N, O]);
tensor_impl!(Tensor4D, [M, N, O, P]);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_0d_neg() {
        let a = Tensor0D::new(10.0);
        let r = -(a.trace());
        assert_eq!(r.data(), &-10.0);
        let gradients = r.backward();
        assert_eq!(gradients.ref_gradient(&a), &-1.0);
    }

    #[test]
    fn test_1d_neg() {
        let a: Tensor1D<3> = Tensor1D::new([-2.0, 0.0, 5.0]);
        let r = -(a.trace());
        assert_eq!(r.data(), &[2.0, 0.0, -5.0]);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&a), &[-1.0 / 3.0; 3]);
    }

    #[test]
    fn test_2d_neg() {
        let a: Tensor2D<2, 3> = Tensor2D::new([[-2.0, 0.0, 5.0], [1.0, 2.0, 3.0]]);
        let r = -(a.trace());
        assert_eq!(r.data(), &[[2.0, 0.0, -5.0], [-1.0, -2.0, -3.0]]);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&a), &[[-1.0 / 6.0; 3]; 2]);
    }

    #[test]
    fn test_3d_neg() {
        let a: Tensor3D<4, 2, 3> = Tensor3D::ones();
        let r = -(a.trace());
        assert_eq!(r.data(), &[[[-1.0; 3]; 2]; 4]);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&a), &[[[-1.0 / 24.0; 3]; 2]; 4]);
    }
}
