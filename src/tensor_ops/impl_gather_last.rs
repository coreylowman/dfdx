use super::utils::move_tape_and_add_backward_op;
use crate::prelude::*;

/// `t.gather(-1, indices)`. Reduces the last dimension of the tensor by gathering the value specified by `indices`.
/// Resulting [Tensor] has the last dimension removed (e.g. a 2d tensor will become 1d).
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = Tensor2D::new([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
/// let r: Tensor1D<2> = gather_last_dim(t, &[2, 1]);
/// assert_eq!(r.data(), &[3.0, -2.0]);
/// ```
///
/// This is equivalent to calling `t.gather(-1, indices)` in pytorch.
pub fn gather_last_dim<T: Reduce1<-1>>(mut t: T, indices: &T::ReducingIndices) -> T::Reduced
where
    T::Array: MultiDimensional,
    T::Device: ForEachLast<<T::Reduced as HasArrayType>::Array, T::Array, T::ReducingIndices>
        + FillElements<<T::Array as MultiDimensional>::LastDim>,
{
    // gather indices
    let mut result = <T::Reduced as Tensor>::NoTape::zeros();
    T::Device::foreachlast_brb(
        BroadcastMut(result.mut_data()),
        t.data(),
        Broadcast(indices),
        &mut |r, t, i| {
            *r = t[*i];
        },
    );

    // store derivative in t
    T::Device::foreachlast_mb(t.mut_data(), Broadcast(indices), &mut |t, i| {
        T::Device::fill(t, &mut |v| *v = 0.0);
        t[*i] = 1.0;
    });

    move_tape_and_add_backward_op(t, result, move |mut t, result, grads| {
        let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);
        T::DeviceR::foreach_br(t.mut_data(), result_grad, &mut |d, r| {
            *d *= r;
        });
        T::Device::foreach_mr(t_grad, t.data(), &mut |g, dr| {
            *g += dr;
        })
    })
}

macro_rules! gather_last_impl {
    ($T:ident, [$($Ts:tt),*], $I:ty) => {
impl<$(const $Ts: usize, )* H: Tape> $T<$($Ts, )* H> {
    /// Calls [gather_last_dim()] on `self`.
    pub fn gather_last_dim(self, indices: &$I) -> <Self as Reduce1<-1>>::Reduced {
        gather_last_dim(self, indices)
    }
}
    };
}

gather_last_impl!(Tensor1D, [M], usize);
gather_last_impl!(Tensor2D, [M, N], [usize; M]);
gather_last_impl!(Tensor3D, [M, N, O], [[usize; N]; M]);
gather_last_impl!(Tensor4D, [M, N, O, P], [[[usize; O]; N]; M]);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gather_last_1d() {
        let t: Tensor1D<3> = Tensor1D::new([1.0, 2.0, 3.0]);
        let r: Tensor0D<OwnedTape> = gather_last_dim(t.trace(), &2);
        assert_eq!(r.data(), &3.0);
        // NOTE: .exp() so we make sure its using result grad properly
        let gradients = r.exp().backward();
        assert_eq!(gradients.ref_gradient(&t), &[0.0, 0.0, 20.085537]);
    }

    #[test]
    fn test_gather_last_2d() {
        let t: Tensor2D<2, 3> = Tensor2D::new([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
        let r: Tensor1D<2, OwnedTape> = gather_last_dim(t.trace(), &[1, 2]);
        assert_eq!(r.data(), &[2.0, -3.0]);
        let gradients = r.mean().backward();
        assert_eq!(
            gradients.ref_gradient(&t),
            &[[0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]
        );
    }

    #[test]
    fn test_gather_last_3d() {
        let t: Tensor3D<4, 2, 3> = Tensor3D::new([
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]],
            [[-3.0, 2.0, -1.0], [-6.0, 5.0, -4.0]],
            [[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]],
        ]);
        let r: Tensor2D<4, 2, OwnedTape> =
            gather_last_dim(t.trace(), &[[0, 1], [2, 2], [1, 1], [0, 0]]);
        assert_eq!(
            r.data(),
            &[[1.0, 5.0], [-3.0, -6.0], [2.0, 5.0], [1.0, 4.0]]
        );
        let gradients = r.mean().backward();
        assert_eq!(
            gradients.ref_gradient(&t),
            &[
                [[0.125, 0.0, 0.0], [0.0, 0.125, 0.0]],
                [[0.0, 0.0, 0.125], [0.0, 0.0, 0.125]],
                [[0.0, 0.125, 0.0], [0.0, 0.125, 0.0]],
                [[0.125, 0.0, 0.0], [0.125, 0.0, 0.0]]
            ]
        );
    }
}
