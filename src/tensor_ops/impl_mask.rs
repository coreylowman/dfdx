use crate::prelude::*;

/// Sets `t` to `value` anywhere `mask` equals value
///
/// # Examples
/// ```rust
/// # use dfdx::prelude::*;
/// let t: Tensor1D<3> = Tensor1D::new([1.0, 2.0, 3.0]);
/// let m: Tensor1D<3> = Tensor1D::new([-1e10, 0.0, -1e10]);
/// let r = t.trace().value_mask(&m, -1e10);
/// assert_eq!(r.data(), &[-1e10, 2.0, -1e10]);
/// ```
pub fn value_mask<T: Tensor<Dtype = f32>>(t: T, other: &T::NoTape, value: T::Dtype) -> T {
    let result = T::NoTape::new_boxed(T::Device::zip_map(t.data(), other.data(), |x, y| {
        if y == &value {
            value
        } else {
            *x
        }
    }));
    let (mut t, mut tape_holder) = t.split_tape_holder();
    T::Device::map_into(other.data(), t.mut_data(), |x| (x != &value) as i32 as f32);
    let _result = result.phantom();
    tape_holder.add_backward_op(move |tape| {
        T::Device::mul_assign(t.mut_data(), tape.ref_gradient(&_result));
        T::Device::add_assign(tape.mut_gradient(&t), t.data());
    });
    result.with_tape_holder(tape_holder)
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: TapeHolder> $typename<$($Vs, )* H> {
    /// Calls [value_mask] on self
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

    #[test]
    fn test_mask_0d() {
        let t = Tensor0D::new(1.0);
        let m = Tensor0D::new(-1e10);
        let r = t.trace().value_mask(&m, -1e10);
        assert_eq!(r.data(), &-1e10);
        let gradients = backward(r.mean());
        assert_eq!(gradients.ref_gradient(&t), &0.0);
    }

    #[test]
    fn test_mask_1d() {
        let t: Tensor1D<3> = Tensor1D::new([1.0, 2.0, 3.0]);
        let m: Tensor1D<3> = Tensor1D::new([-1e10, 0.0, -1e10]);
        let r = t.trace().value_mask(&m, -1e10);
        assert_eq!(r.data(), &[-1e10, 2.0, -1e10]);
        let gradients = backward(r.mean());
        assert_eq!(gradients.ref_gradient(&t), &[0.0, 1.0 / 3.0, 0.0]);
    }

    #[test]
    fn test_mask_2d() {
        let t: Tensor2D<2, 3> = Tensor2D::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let m: Tensor2D<2, 3> = Tensor2D::new([[-1e10, 0.0, -1e10], [1.0, -1e10, -1e9]]);
        let r = t.trace().value_mask(&m, -1e10);
        assert_eq!(r.data(), &[[-1e10, 2.0, -1e10], [4.0, -1e10, 6.0]]);
        let gradients = backward(r.mean());
        assert_eq!(
            gradients.ref_gradient(&t),
            &[[0.0, 1.0 / 6.0, 0.0], [1.0 / 6.0, 0.0, 1.0 / 6.0]]
        );
    }
}
