use crate::prelude::*;

impl ZipMapElements<f32, usize> for Cpu {
    /// TODO make this private to this module somehow
    /// NOTE: This is extremely specific to [gather_last_dim()]. This is used for setting the
    /// derivative in that function.
    fn zip_map_assign<F: FnMut(&mut f32, &usize)>(l: &mut f32, _: &usize, _: &mut F) {
        *l = 1.0;
    }
}

impl<const M: usize> ZipMapElements<[f32; M], usize> for Cpu {
    /// TODO make this private to this module somehow
    /// NOTE: this is extremely specific to [gather_last_dim()]. This is used for setting
    /// the derivative in that function.
    fn zip_map_assign<F: FnMut(&mut f32, &usize)>(l: &mut [f32; M], r: &usize, _: &mut F) {
        for (i, l_i) in l.iter_mut().enumerate() {
            *l_i = if i == *r { 1.0 } else { 0.0 };
        }
    }
}

/// Reduces the last dimension of the tensor by gathering the value specified by `indices`.
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
pub fn gather_last_dim<T: Tensor<Dtype = f32>, I>(t: T, indices: &I) -> T::LastDimReduced
where
    I: CountElements<Dtype = usize>,
    T::Device: GatherElements<T::Array, Indices = I, Gathered = <T::LastDimReduced as HasArrayType>::Array>
        + Device<T::Array>
        + ZipMapElements<T::Array, I>,
{
    let result =
        <T::LastDimReduced as Tensor>::NoTape::new_boxed(T::Device::gather(t.data(), indices));
    let (mut t, mut tape) = t.split_tape();
    T::Device::zip_map_assign(t.mut_data(), indices, &mut |_, _| {});
    let _result = result.phantom();
    tape.add_backward_op(move |grads| {
        T::Device::mul_assign(t.mut_data(), grads.ref_gradient(&_result));
        T::Device::add_assign(grads.mut_gradient(&t), t.data());
    });
    result.put_tape(tape)
}

macro_rules! gather_last_impl {
    ($T:ident, [$($Ts:tt),*], $I:ty) => {
impl<$(const $Ts: usize, )* H: Tape> $T<$($Ts, )* H> {
    /// Calls [gather_last_dim()] on `self`.
    pub fn gather_last_dim(self, indices: &$I) -> <Self as Tensor>::LastDimReduced {
        gather_last_dim(self, indices)
    }
}
    };
}

gather_last_impl!(Tensor0D, [], usize);
gather_last_impl!(Tensor1D, [M], usize);
gather_last_impl!(Tensor2D, [M, N], [usize; M]);
gather_last_impl!(Tensor3D, [M, N, O], [[usize; N]; M]);
gather_last_impl!(Tensor4D, [M, N, O, P], [[[usize; O]; N]; M]);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gather_last_0d() {
        let t = Tensor0D::new(2.0);
        let r: Tensor0D<OwnsTape> = gather_last_dim(t.trace(), &0);
        assert_eq!(r.data(), &2.0);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&t), &1.0);
    }

    #[test]
    fn test_gather_last_1d() {
        let t: Tensor1D<3> = Tensor1D::new([1.0, 2.0, 3.0]);
        let r: Tensor0D<OwnsTape> = gather_last_dim(t.trace(), &2);
        assert_eq!(r.data(), &3.0);
        // NOTE: .exp() so we make sure its using result grad properly
        let gradients = r.exp().backward();
        assert_eq!(gradients.ref_gradient(&t), &[0.0, 0.0, 20.085537]);
    }

    #[test]
    fn test_gather_last_2d() {
        let t: Tensor2D<2, 3> = Tensor2D::new([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
        let r: Tensor1D<2, OwnsTape> = gather_last_dim(t.trace(), &[1, 2]);
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
        let r: Tensor2D<4, 2, OwnsTape> =
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
