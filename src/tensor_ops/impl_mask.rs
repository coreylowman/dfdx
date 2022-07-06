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
    let mut result = T::NoTape::zeros();
    T::Device::foreach_mrr(result.mut_data(), t.data(), other.data(), &mut |r, t, o| {
        *r = if o == &value { value } else { *t }
    });

    // store derivative in t
    let (mut t, mut tape) = t.split_tape();
    T::Device::foreach_mr(t.mut_data(), other.data(), &mut |t, o| {
        *t = if o == &value { 0.0 } else { 1.0 }
    });

    let _result = result.phantom();
    tape.add_backward_op(move |grads| {
        let (t_grad, result_grad) = grads.mut_and_ref(&t, &_result);
        T::Device::addmul(t_grad, t.data(), result_grad);
    });
    result.put_tape(tape)
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: Tape> $typename<$($Vs, )* H> {
    /// Calls [value_mask()] on self
    pub fn value_mask(self, mask: &$typename<$($Vs, )* NoneTape>, value: f32) -> Self {
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
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&t), &0.0);
    }

    #[test]
    fn test_mask_1d() {
        let t: Tensor1D<3> = Tensor1D::new([1.0, 2.0, 3.0]);
        let m: Tensor1D<3> = Tensor1D::new([-1e10, 0.0, -1e10]);
        let r = t.trace().value_mask(&m, -1e10);
        assert_eq!(r.data(), &[-1e10, 2.0, -1e10]);
        // NOTE: .exp() so we cover the case where .mask() has to use result grad
        let gradients = r.exp().mean().backward();
        assert_eq!(gradients.ref_gradient(&t), &[0.0, 2.463019, 0.0]);
    }

    #[test]
    fn test_mask_2d() {
        let t: Tensor2D<2, 3> = Tensor2D::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let m: Tensor2D<2, 3> = Tensor2D::new([[-1e10, 0.0, -1e10], [1.0, -1e10, -1e9]]);
        let r = t.trace().value_mask(&m, -1e10);
        assert_eq!(r.data(), &[[-1e10, 2.0, -1e10], [4.0, -1e10, 6.0]]);
        let gradients = r.mean().backward();
        assert_eq!(
            gradients.ref_gradient(&t),
            &[[0.0, 1.0 / 6.0, 0.0], [1.0 / 6.0, 0.0, 1.0 / 6.0]]
        );
    }
}
