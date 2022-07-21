use super::utils::move_tape_and_add_backward_op;
use crate::prelude::*;
use rand::Rng;
use rand_distr::{Distribution, Standard};

/// Randomly drops out elements from `t` with probability `p`, and multiplies all elements by `1 / (1 - p)`.
///
/// If the `t: T` passed in does **not** have a tape, then no dropout is applied. See [Tape::OWNS_TAPE].
///
/// Described in paper: [Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/abs/1207.0580)
pub fn dropout<T: Tensor<Dtype = f32>, R: Rng>(t: T, p: f32, rng: &mut R) -> T {
    if !T::Tape::OWNS_TAPE {
        // This is the branch where `t` doesn't own the tape, so we don't have to drop out anything.
        t
    } else {
        // `t` owns the tape in this branch, so apply dropout randomly.
        let rinvp = (1.0 - p).recip();
        let deriv = T::Device::filled(&mut |d| {
            let val: f32 = Standard.sample(rng);
            *d = if val < p { 0.0 } else { rinvp };
        });
        let mut result = T::NoTape::zeros();
        T::Device::addmul(result.mut_data(), t.data(), deriv.as_ref());

        move_tape_and_add_backward_op(t, result, move |t, result, grads| {
            let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);
            T::Device::addmul(t_grad, deriv.as_ref(), result_grad);
        })
    }
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: Tape> $typename<$($Vs, )* H> {
    /// Calls [dropout()] on `self`.
    pub fn dropout<R: Rng>(self, p: f32, rng: &mut R) -> Self {
        dropout(self, p, rng)
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
    use rand::{prelude::StdRng, SeedableRng};

    #[test]
    fn test_dropout_all_0d() {
        let mut rng = StdRng::seed_from_u64(0);
        let t: Tensor0D = Tensor0D::new(3.0);
        let r = t.trace().dropout(1.0, &mut rng);
        assert_eq!(r.data(), &0.0);
        let gradients = r.backward();
        assert_eq!(gradients.ref_gradient(&t), &0.0);
    }

    #[test]
    fn test_dropout_none_0d() {
        let mut rng = StdRng::seed_from_u64(0);
        let t: Tensor0D = Tensor0D::new(3.0);
        let r = t.trace().dropout(0.0, &mut rng);
        assert_eq!(r.data(), &3.0);
        let gradients = r.backward();
        assert_eq!(gradients.ref_gradient(&t), &1.0);
    }

    #[test]
    fn test_dropout_no_tape_0d() {
        let mut rng = StdRng::seed_from_u64(0);
        let t: Tensor0D = Tensor0D::new(3.0);
        let r = t.dropout(1.0, &mut rng);
        assert_eq!(r.data(), &3.0);
    }

    #[test]
    fn test_dropout_1d() {
        let mut rng = StdRng::seed_from_u64(3);
        let t = Tensor1D::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        let r = t.trace().dropout(0.5, &mut rng);
        assert_eq!(r.data(), &[2.0, 0.0, 6.0, 0.0, 0.0]);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&t), &[0.4, 0.0, 0.4, 0.0, 0.0]);
    }

    #[test]
    fn test_dropout_2d() {
        let mut rng = StdRng::seed_from_u64(0);
        let t = Tensor2D::new([[0.05, 0.1, 0.2], [0.3, 0.4, 0.5]]);
        let r = t.trace().dropout(0.6, &mut rng);
        assert_eq!(
            r.data(),
            &[[0.12500001, 0.25000003, 0.0], [0.7500001, 1.0000001, 0.0]]
        );
        // NOTE: .exp() so we ensure result grad is used properly
        let gradients = r.exp().mean().backward();
        assert_eq!(
            gradients.ref_gradient(&t),
            &[[0.47214523, 0.5350107, 0.0], [0.88208354, 1.1326177, 0.0]]
        );
    }

    #[test]
    fn test_dropout_3d() {
        let mut rng = StdRng::seed_from_u64(0);
        let t: Tensor3D<4, 2, 3> = Tensor3D::ones();
        let r = t.trace().dropout(0.2, &mut rng);
        assert_eq!(
            r.data(),
            &[
                [[1.25, 1.25, 1.25], [1.25, 1.25, 0.0]],
                [[1.25, 1.25, 1.25], [1.25, 0.0, 1.25]],
                [[0.0, 1.25, 1.25], [1.25, 1.25, 1.25]],
                [[1.25, 1.25, 1.25], [1.25, 1.25, 0.0]]
            ]
        );
        let gradients = r.mean().backward();
        const V: f32 = 0.052083336;
        assert_eq!(
            gradients.ref_gradient(&t),
            &[
                [[V, V, V], [V, V, 0.0]],
                [[V, V, V], [V, 0.0, V]],
                [[0.0, V, V], [V, V, V]],
                [[V, V, V], [V, V, 0.0]]
            ]
        );
    }
}
