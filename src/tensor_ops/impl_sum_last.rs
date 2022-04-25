use super::ops::add_unary_op;
use crate::prelude::*;
use ndarray::prelude::*;

pub trait HasSumLastMethod: Tensor {
    type Output: Tensor;
    fn sum_last(self) -> Self::Output;
}

macro_rules! sum_last_impl {
    ($typename:ident, [$($Vs:tt),*], [$($Zs:tt),*], $ax:expr) => {
impl<$(const $Vs: usize, )* H: TapeHolder> HasSumLastMethod for $typename<$($Vs, )* H> {
    type Output = $typename<$($Zs, )* H>;
    fn sum_last(self) -> Self::Output {
        let result = <$typename<$($Zs, )* H> as Tensor>::NoTape::new(self.data().sum_axis($ax).insert_axis($ax));
        let (t, mut tape_holder) = self.split_tape_holder();
        tape_holder.update_with(|tape| {
            add_unary_op(tape, (&t, &result), t.data().map(|_| 1.0))
        });
        result.with_tape_holder(tape_holder)
    }
}
    };
}

sum_last_impl!(Tensor1D, [M], [1], Axis(0));
sum_last_impl!(Tensor2D, [M, N], [M, 1], Axis(1));
sum_last_impl!(Tensor3D, [M, N, O], [M, N, 1], Axis(2));
sum_last_impl!(Tensor4D, [M, N, O, P], [M, N, O, 1], Axis(3));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_last_1d() {
        let t: Tensor1D<3> = Tensor1D::new(arr1(&[1.0, 2.0, 3.0]));
        let r: Tensor1D<1, WithTape> = t.with_tape().sum_last();
        assert_eq!(r.data(), arr1(&[6.0]));
        let gradients = backward(r.mean());
        assert_eq!(
            gradients
                .gradient_for(t.id())
                .clone()
                .into_shape((3,))
                .unwrap(),
            arr1(&[1.0; 3])
        );
    }

    #[test]
    fn test_sum_last_2d() {
        let t: Tensor2D<2, 3> = Tensor2D::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
        let r: Tensor2D<2, 1, WithTape> = t.with_tape().sum_last();
        assert_eq!(r.data(), arr2(&[[6.0], [15.0]]));
        let gradients = backward(r.mean());
        assert_eq!(
            gradients
                .gradient_for(t.id())
                .clone()
                .into_shape((2, 3))
                .unwrap(),
            arr2(&[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
        );
    }
}
