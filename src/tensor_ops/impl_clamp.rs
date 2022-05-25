use crate::prelude::*;

/// Clamps all values in `t` to between `min` and `max`
///
/// # Examples
/// ```rust
/// # use dfdx::prelude::*;
/// let t = Tensor1D::new([-1.0, -0.5, 0.0, 0.5, 1.0]);
/// let r = t.clamp(-0.5, 0.5);
/// assert_eq!(r.data(), &[-0.5, -0.5, 0.0, 0.5, 0.5]);
/// ```
pub fn clamp<T: Tensor<Dtype = f32>>(t: T, min: T::Dtype, max: T::Dtype) -> T {
    let result = T::NoTape::new_boxed(T::Device::map(t.data(), |x| x.clamp(min, max)));
    let (mut t, mut tape_holder) = t.split_tape_holder();
    let _result = result.phantom();
    tape_holder.add_operation(move |tape| {
        T::Device::zip_map_assign(t.mut_data(), tape.ref_gradient(&_result), &mut |l, r| {
            *l = if min <= *l && *l <= max { *r } else { 0.0 }
        });
        T::Device::add_assign(tape.mut_gradient(&t), t.data());
    });
    result.with_tape_holder(tape_holder)
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: TapeHolder> $typename<$($Vs, )* H> {
    /// Calls [clamp] on self
    pub fn clamp(self, min: f32, max: f32) -> Self {
        clamp(self, min, max)
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
    fn test_clamp_0d() {
        let t = Tensor0D::new(1.0);
        let r = t.trace().clamp(0.0, 1.0);
        assert_eq!(r.data(), &1.0);
        let gradients = backward(r.mean());
        assert_eq!(gradients.ref_gradient(&t), &1.0);
    }

    #[test]
    fn test_clamp_1d() {
        let t = Tensor1D::new([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]);
        let r = t.trace().clamp(-0.5, 0.25);
        assert_eq!(r.data(), &[-0.5, -0.5, -0.25, 0.0, 0.25, 0.25, 0.25]);
        let gradients = backward(r.mean());
        const V: f32 = 1.0 / 7.0;
        assert_eq!(gradients.ref_gradient(&t), &[0.0, V, V, V, V, 0.0, 0.0]);
    }

    #[test]
    fn test_clamp_2d() {
        let t: Tensor2D<2, 3> = Tensor2D::new([[-1.0, 0.0, 1.0], [-2.0, 2.0, 1.1]]);
        let r = t.trace().clamp(-1.0, 1.0);
        assert_eq!(r.data(), &[[-1.0, 0.0, 1.0], [-1.0, 1.0, 1.0]]);
        let gradients = backward(r.mean());
        assert_eq!(gradients.ref_gradient(&t), &[[1.0 / 6.0; 3], [0.0; 3]]);
    }
}
