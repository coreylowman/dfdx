use crate::prelude::*;

fn nans_to<T: Tensor>(t: T, value: f32) -> T {
    let result = T::NoTape::new(t.data().map_elems(|v| if v.is_nan() { value } else { *v }));
    let (t, mut tape_holder) = t.split_tape_holder();
    let deriv: T::ArrayType = t.data().map_elems(|v| if v.is_nan() { 0.0 } else { 1.0 });
    let _t = t.phantom();
    let _result = result.phantom();
    tape_holder.add_operation(move |tape| {
        let d_grad = deriv.mul(tape.gradient(&_result));
        tape.mut_gradient(&_t).add_assign(&d_grad);
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

    #[test]
    fn test_nans_0d() {
        let t = Tensor0D::new(f32::NAN);
        let r = t.with_tape().nans_to(0.0);
        assert_eq!(r.data(), &0.0);
        let gradients = backward(r.mean());
        assert_eq!(gradients.gradient(&t), &0.0);
    }

    #[test]
    fn test_nans_1d() {
        let t: Tensor1D<4> = Tensor1D::new([1.0, f32::NAN, f32::NAN, 4.0]);
        let r = t.with_tape().nans_to(0.0);
        assert_eq!(r.data(), &[1.0, 0.0, 0.0, 4.0]);
        let gradients = backward(r.mean());
        assert_eq!(gradients.gradient(&t), &[0.25, 0.0, 0.0, 0.25]);
    }

    #[test]
    fn test_nans_2d() {
        let t: Tensor2D<2, 3> = Tensor2D::new([[1.0, f32::NAN, 3.0], [f32::NAN, 4.0, f32::NAN]]);
        let r = t.with_tape().nans_to(0.0);
        assert_eq!(r.data(), &[[1.0, 0.0, 3.0], [0.0, 4.0, 0.0]]);
        let gradients = backward(r.mean());
        assert_eq!(
            gradients.gradient(&t),
            &[[1.0 / 6.0, 0.0, 1.0 / 6.0], [0.0, 1.0 / 6.0, 0.0]]
        );
    }

    #[test]
    fn test_nans_3d() {
        todo!();
    }
}
