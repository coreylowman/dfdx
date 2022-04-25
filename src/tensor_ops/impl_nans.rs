use super::ops::add_unary_op;
use crate::prelude::*;

fn nans_to<T: Tensor>(t: T, value: f32) -> T {
    let result = T::NoTape::new(t.data().mapv(|v| if v.is_nan() { value } else { v }));
    let (t, mut tape_holder) = t.split_tape_holder();
    tape_holder.update_with(|tape| {
        let deriv = t.data().mapv(|v| if v.is_nan() { 0.0 } else { 1.0 });
        add_unary_op(tape, (&t, &result), deriv)
    });
    result.with_tape_holder(tape_holder)
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: TapeHolder> $typename<$($Vs, )* H> {
    pub fn nans_to(self, value: f32) -> Self {
        nans_to(self, value)
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
    fn test_nans_0d() {
        let t = Tensor0D::new(arr0(f32::NAN));
        let r = t.with_tape().nans_to(0.0);
        assert_eq!(r.data(), arr0(0.0));
        let gradients = backward(r.mean());
        assert_eq!(
            gradients.gradient_for(t.id()).to_shape(t.shape()).unwrap(),
            arr0(0.0)
        );
    }

    #[test]
    fn test_nans_1d() {
        let t: Tensor1D<4> = Tensor1D::new(arr1(&[1.0, f32::NAN, f32::NAN, 4.0]));
        let r = t.with_tape().nans_to(0.0);
        assert_eq!(r.data(), arr1(&[1.0, 0.0, 0.0, 4.0]));
        let gradients = backward(r.mean());
        assert_eq!(
            gradients.gradient_for(t.id()).to_shape(t.shape()).unwrap(),
            arr1(&[0.25, 0.0, 0.0, 0.25])
        );
    }

    #[test]
    fn test_nans_2d() {
        let t: Tensor2D<2, 3> =
            Tensor2D::new(arr2(&[[1.0, f32::NAN, 3.0], [f32::NAN, 4.0, f32::NAN]]));
        let r = t.with_tape().nans_to(0.0);
        assert_eq!(r.data(), arr2(&[[1.0, 0.0, 3.0], [0.0, 4.0, 0.0]]));
        let gradients = backward(r.mean());
        assert_eq!(
            gradients.gradient_for(t.id()).to_shape(t.shape()).unwrap(),
            arr2(&[[1.0 / 6.0, 0.0, 1.0 / 6.0], [0.0, 1.0 / 6.0, 0.0]])
        );
    }
}
