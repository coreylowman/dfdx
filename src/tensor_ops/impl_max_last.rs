use super::utils::move_tape_and_add_backward_op;
use crate::prelude::*;

/// `t.max(-1)`. Reduces the last dimension of the tensor by gathering the maximum value from that dimension.
/// Resulting [Tensor] has the last dimension removed (e.g. a 2d tensor will become 1d).
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = Tensor2D::new([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
/// let r: Tensor1D<2> = max_last_dim(t);
/// assert_eq!(r.data(), &[3.0, -1.0]);
/// ```
///
/// This is equivalent to calling `t.max(-1)[0]` in pytorch.
pub fn max_last_dim<T: Tensor<Dtype = f32>>(mut t: T) -> T::LastDimReduced {
    let result = <T::LastDimReduced as Tensor>::NoTape::new_boxed(T::Device::reduce_last_dim(
        t.data(),
        &mut f32::max,
    ));

    // store derivative in t
    T::Device::foreach_mb(t.mut_data(), Broadcast(result.data()), &mut |l, r| {
        *l = if l == r { 1.0 } else { 0.0 }
    });

    move_tape_and_add_backward_op(t, result, move |t, result, grads| {
        let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);
        T::Device::foreach_mrb(t_grad, t.data(), Broadcast(result_grad), &mut |g, t, r| {
            *g += t * r;
        });
    })
}

macro_rules! max_last_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: Tape> $typename<$($Vs, )* H> {
    /// Calls [max_last_dim()] on `self`.
    pub fn max_last_dim(self) -> <Self as Tensor>::LastDimReduced {
        max_last_dim(self)
    }
}
    };
}

max_last_impl!(Tensor0D, []);
max_last_impl!(Tensor1D, [M]);
max_last_impl!(Tensor2D, [M, N]);
max_last_impl!(Tensor3D, [M, N, O]);
max_last_impl!(Tensor4D, [M, N, O, P]);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_last_0d() {
        let t = Tensor0D::new(2.0);
        let r: Tensor0D<OwnedTape> = t.trace().max_last_dim();
        assert_eq!(r.data(), &2.0);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&t), &1.0);
    }

    #[test]
    fn test_max_last_1d() {
        let t: Tensor1D<3> = Tensor1D::new([1.0, 2.0, 3.0]);
        let r: Tensor0D<OwnedTape> = t.trace().max_last_dim();
        assert_eq!(r.data(), &3.0);
        // NOTE: .exp() so we make sure its using result grad properly
        let gradients = r.exp().backward();
        assert_eq!(gradients.ref_gradient(&t), &[0.0, 0.0, 20.085537]);
    }

    #[test]
    fn test_max_last_2d() {
        let t: Tensor2D<2, 3> = Tensor2D::new([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
        let r: Tensor1D<2, OwnedTape> = t.trace().max_last_dim();
        assert_eq!(r.data(), &[3.0, -1.0]);
        let gradients = r.mean().backward();
        assert_eq!(
            gradients.ref_gradient(&t),
            &[[0.0, 0.0, 0.5], [0.5, 0.0, 0.0]]
        );
    }

    #[test]
    fn test_max_last_3d() {
        let t: Tensor3D<4, 2, 3> = Tensor3D::new([
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]],
            [[-3.0, 2.0, -1.0], [-6.0, 5.0, -4.0]],
            [[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]],
        ]);
        let r: Tensor2D<4, 2, OwnedTape> = t.trace().max_last_dim();
        assert_eq!(
            r.data(),
            &[[3.0, 6.0], [-1.0, -4.0], [2.0, 5.0], [3.0, 6.0]]
        );
        let gradients = r.mean().backward();
        assert_eq!(
            gradients.ref_gradient(&t),
            &[
                [[0.0, 0.0, 0.125], [0.0, 0.0, 0.125]],
                [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0]],
                [[0.0, 0.125, 0.0], [0.0, 0.125, 0.0]],
                [[0.0, 0.0, 0.125], [0.0, 0.0, 0.125]]
            ]
        );
    }
}
