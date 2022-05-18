use crate::prelude::*;

pub trait HasSumLastMethod: Tensor {
    type Output: Tensor;
    fn sum_last(self) -> Self::Output;
}

macro_rules! sum_last_impl {
    ($typename:ident, [$($Vs:tt),*], $res:ident, [$($Zs:tt),*]) => {
impl<$(const $Vs: usize, )* H: TapeHolder> HasSumLastMethod for $typename<$($Vs, )* H> {
    type Output = $res<$($Zs, )* H>;
    fn sum_last(self) -> Self::Output {
        let result = <$res<$($Zs, )* H> as Tensor>::NoTape::new_boxed(Self::Device::reduce_inner(self.data(), |a, b| a + b));
        let mut deriv = Self::Device::map(self.data(), |_| 1.0);
        let (t, mut tape_holder) = self.split_tape_holder();
        let _result = result.phantom();
        tape_holder.add_operation(move |tape| {
            Self::Device::mul_assign(deriv.as_mut(), tape.ref_gradient(&_result));
            Self::Device::add_assign(tape.mut_gradient(&t), deriv.as_ref());
        });
        result.with_tape_holder(tape_holder)
    }
}
    };
}

sum_last_impl!(Tensor1D, [M], Tensor0D, []);
sum_last_impl!(Tensor2D, [M, N], Tensor1D, [M]);
sum_last_impl!(Tensor3D, [M, N, O], Tensor2D, [M, N]);
sum_last_impl!(Tensor4D, [M, N, O, P], Tensor3D, [M, N, O]);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_last_1d() {
        let t: Tensor1D<3> = Tensor1D::new([1.0, 2.0, 3.0]);
        let r: Tensor0D<WithTape> = t.trace().sum_last();
        assert_eq!(r.data(), &6.0);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&t), &[1.0; 3]);
    }

    #[test]
    fn test_sum_last_2d() {
        let t: Tensor2D<2, 3> = Tensor2D::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let r: Tensor1D<2, WithTape> = t.trace().sum_last();
        assert_eq!(r.data(), &[6.0, 15.0]);
        let gradients = r.mean().backward();
        assert_eq!(
            gradients.ref_gradient(&t),
            &[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
        );
    }

    #[test]
    fn test_sum_last_3d() {
        let t: Tensor3D<4, 2, 3> = Tensor3D::new([
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]],
            [[-3.0, 2.0, -1.0], [-6.0, 5.0, -4.0]],
            [[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]],
        ]);
        let r: Tensor2D<4, 2, WithTape> = t.trace().sum_last();
        assert_eq!(
            r.data(),
            &[[6.0, 15.0], [-6.0, -15.0], [-2.0, -5.0], [2.0, 5.0],]
        );
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&t), &[[[1.0 / 8.0; 3]; 2]; 4]);
    }
}
