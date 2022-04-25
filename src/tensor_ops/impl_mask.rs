use super::ops::add_unary_op;
use crate::prelude::*;

pub fn value_mask<T: Tensor>(t: T, other: &T::NoTape, value: f32) -> T {
    let result = T::NoTape::new(
        ndarray::Zip::from(t.data())
            .and(other.data())
            .map_collect(|&a, b| if b.eq(&value) { value } else { a }),
    );
    let (t, mut tape_holder) = t.split_tape_holder();
    tape_holder.update_with(|tape| {
        let deriv = other.data().map(|v| if v.eq(&value) { 0.0 } else { 1.0 });
        add_unary_op(tape, (&t, &result), deriv)
    });
    result.with_tape_holder(tape_holder)
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: TapeHolder> $typename<$($Vs, )* H> {
    pub fn value_mask(self, mask: &$typename<$($Vs, )* NoTape>, value: f32) -> Self {
        value_mask(self, mask, value)
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
    use ndarray::prelude::*;

    #[test]
    fn test_mask_0d() {
        let t = Tensor0D::new(arr0(1.0));
        let m = Tensor0D::new(arr0(-1e10));
        let r = t.with_tape().value_mask(&m, -1e10);
        assert_eq!(r.data(), arr0(-1e10));
        let gradients = backward(r.mean());
        assert_eq!(
            gradients.gradient_for(t.id()).to_shape(t.shape()).unwrap(),
            arr0(0.0)
        );
    }

    #[test]
    fn test_mask_1d() {
        let t: Tensor1D<3> = Tensor1D::new(arr1(&[1.0, 2.0, 3.0]));
        let m: Tensor1D<3> = Tensor1D::new(arr1(&[-1e10, 0.0, -1e10]));
        let r = t.with_tape().value_mask(&m, -1e10);
        assert_eq!(r.data(), arr1(&[-1e10, 2.0, -1e10]));
        let gradients = backward(r.mean());
        assert_eq!(
            gradients.gradient_for(t.id()).to_shape(t.shape()).unwrap(),
            arr1(&[0.0, 1.0 / 3.0, 0.0])
        );
    }

    #[test]
    fn test_mask_2d() {
        let t: Tensor2D<2, 3> = Tensor2D::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
        let m: Tensor2D<2, 3> = Tensor2D::new(arr2(&[[-1e10, 0.0, -1e10], [1.0, -1e10, -1e9]]));
        let r = t.with_tape().value_mask(&m, -1e10);
        assert_eq!(r.data(), arr2(&[[-1e10, 2.0, -1e10], [4.0, -1e10, 6.0]]));
        let gradients = backward(r.mean());
        assert_eq!(
            gradients.gradient_for(t.id()).to_shape(t.shape()).unwrap(),
            arr2(&[[0.0, 1.0 / 6.0, 0.0], [1.0 / 6.0, 0.0, 1.0 / 6.0]])
        );
    }
}
