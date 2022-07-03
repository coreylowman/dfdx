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
    let (t, mut tape) = t.split_tape();
    let _result = result.phantom();
    tape.add_backward_op(move |grads| {
        let (t_grad, result_grad) = grads.mut_and_ref(&t, &_result);
        T::Device::foreach_mrr(t_grad, t.data(), result_grad, &mut |g, t, r| {
            *g += if min <= *t && *t <= max { *r } else { 0.0 }
        });
    });
    result.put_tape(tape)
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: Tape> $typename<$($Vs, )* H> {
    /// Calls [clamp()] on self
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
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&t), &1.0);
    }

    #[test]
    fn test_clamp_1d() {
        let t = Tensor1D::new([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]);
        let r = t.trace().clamp(-0.5, 0.25);
        assert_eq!(r.data(), &[-0.5, -0.5, -0.25, 0.0, 0.25, 0.25, 0.25]);
        // NOTE: .exp() so we cover case where .clamp() needs to use result's grad
        let gradients = r.exp().mean().backward();
        assert_eq!(
            gradients.ref_gradient(&t),
            &[0.0, 0.08664724, 0.11125726, 0.14285715, 0.1834322, 0.0, 0.0]
        );
    }

    #[test]
    fn test_clamp_2d() {
        let t: Tensor2D<2, 3> = Tensor2D::new([[-1.0, 0.0, 1.0], [-2.0, 2.0, 1.1]]);
        let r = t.trace().clamp(-1.0, 1.0);
        assert_eq!(r.data(), &[[-1.0, 0.0, 1.0], [-1.0, 1.0, 1.0]]);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&t), &[[1.0 / 6.0; 3], [0.0; 3]]);
    }
}
