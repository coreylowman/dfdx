use crate::prelude::*;

/// Calls [Device::sum_last_dim()] on the underlying array.
/// Result [Tensor] has smaller number of dimensions.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = Tensor2D::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
/// let r: Tensor1D<2> = sum_last_dim(t);
/// assert_eq!(r.data(), &[6.0, 15.0]);
/// ```
pub fn sum_last_dim<T: Tensor<Dtype = f32>>(t: T) -> T::LastDimReduced {
    let result = <T::LastDimReduced as Tensor>::NoTape::new_boxed(T::Device::reduce_last_dim(
        t.data(),
        |a, b| a + b,
    ));
    let (mut t, mut tape_holder) = t.split_tape_holder();
    let _result = result.phantom();
    tape_holder.add_operation(move |tape| {
        T::Device::zip_map_assign(t.mut_data(), tape.ref_gradient(&_result), |l, r| *l = *r);
        T::Device::add_assign(tape.mut_gradient(&t), t.data());
    });
    result.with_tape_holder(tape_holder)
}

macro_rules! sum_last_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: TapeHolder> $typename<$($Vs, )* H> {
    /// Calls [sum_last_dim()] on `self`.
    pub fn sum_last_dim(self) -> <Self as Tensor>::LastDimReduced {
        sum_last_dim(self)
    }
}
    };
}

sum_last_impl!(Tensor0D, []);
sum_last_impl!(Tensor1D, [M]);
sum_last_impl!(Tensor2D, [M, N]);
sum_last_impl!(Tensor3D, [M, N, O]);
sum_last_impl!(Tensor4D, [M, N, O, P]);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_last_0d() {
        let t = Tensor0D::new(3.14);
        let r: Tensor0D<WithTape> = t.trace().sum_last_dim();
        assert_eq!(r.data(), &3.14);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&t), &1.0);
    }

    #[test]
    fn test_sum_last_1d() {
        let t: Tensor1D<3> = Tensor1D::new([1.0, 2.0, 3.0]);
        let r: Tensor0D<WithTape> = t.trace().sum_last_dim();
        assert_eq!(r.data(), &6.0);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&t), &[1.0; 3]);
    }

    #[test]
    fn test_sum_last_2d() {
        let t: Tensor2D<2, 3> = Tensor2D::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let r: Tensor1D<2, WithTape> = t.trace().sum_last_dim();
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
        let r: Tensor2D<4, 2, WithTape> = t.trace().sum_last_dim();
        assert_eq!(
            r.data(),
            &[[6.0, 15.0], [-6.0, -15.0], [-2.0, -5.0], [2.0, 5.0],]
        );
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&t), &[[[1.0 / 8.0; 3]; 2]; 4]);
    }
}
