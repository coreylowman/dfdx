use crate::prelude::*;

/// Negates all values in `t`.
///
/// # Examples
///
/// ```rust
/// # use dfdx::prelude::*;
/// let a: Tensor1D<3> = Tensor1D::new([-2.0, 0.0, 5.0]);
/// let r = -a; // or negate(a);
/// assert_eq!(r.data(), &[2.0, 0.0, -5.0]);
/// ```
pub fn negate<T: Tensor<Dtype = f32>>(t: T) -> T {
    let result = T::NoTape::new_boxed(T::Device::map(t.data(), |v| -v));
    let (mut t, mut tape) = t.split_tape();
    let _result = result.phantom();
    tape.add_backward_op(move |grads| {
        T::Device::zip_map_assign(t.mut_data(), grads.ref_gradient(&_result), &mut |l, r| {
            *l = -r
        });
        T::Device::add_assign(grads.mut_gradient(&t), t.data());
    });
    result.put_tape(tape)
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: Tape> std::ops::Neg for $typename<$($Vs, )* H>
{
    type Output = Self;

    /// Calls [negate] on `self`.
    fn neg(self) -> Self::Output {
        negate(self)
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
