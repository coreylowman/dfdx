use crate::prelude::*;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::Standard;

/// Does nothing if no tape is in `t`. Zeros elements with probability `p` and scales all elements by `1 / (1 - p)`.
/// See [Tape::OWNS_TAPE].
///
/// Described in paper: [Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/abs/1207.0580)
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// # use rand::prelude::*;
/// let mut rng = StdRng::seed_from_u64(4);
/// let t = tensor([1.0, 2.0, 3.0, 4.0]);
///
/// // no tape in t, this won't do anything
/// let a = dropout(t.clone(), 0.5, &mut rng);
/// assert_eq!(a.data(), t.data());
///
/// // now t has the tape, dropout!
/// let a = dropout(t.trace(), 0.5, &mut rng);
/// assert_eq!(a.data(), &[2.0, 4.0, 0.0, 8.0]);
/// ```
///
/// ### Implementation details:
///
/// To reduce memory usage, this function first samples a u64 seed from `rng`,
/// and then instantiates two identical [StdRng] with that seed. These rngs
/// are used in both the forward pass and backward pass to generate identical
/// random numbers, so the masking is the same for both.
pub fn dropout<T: Tensor<Dtype = f32>, R: Rng>(t: T, p: f32, rng: &mut R) -> T {
    if !T::Tape::OWNS_TAPE {
        // This is the branch where `t` doesn't own the tape, so we don't have to drop out anything.
        t
    } else {
        // `t` owns the tape in this branch, so apply dropout randomly.
        let seed: u64 = rng.gen();
        let mut fwd_rng = StdRng::seed_from_u64(seed);
        let mut bwd_rng = StdRng::seed_from_u64(seed);
        crate::tensor_ops::utils::map(
            t,
            move |x| {
                let val: f32 = fwd_rng.sample(Standard);
                if val < p {
                    0.0
                } else {
                    x / (1.0 - p)
                }
            },
            move |_| {
                let val: f32 = bwd_rng.sample(Standard);
                if val < p {
                    0.0
                } else {
                    1.0 / (1.0 - p)
                }
            },
        )
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
    use crate::tests::assert_close;
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
    fn test_dropout_1d_with_non_positive_values() {
        let mut rng = StdRng::seed_from_u64(3);
        let t = Tensor1D::new([0.0, 2.0, -3.0, -4.0, 0.0]);
        let r = t.trace().dropout(0.5, &mut rng);
        assert_eq!(r.data(), &[0.0, 0.0, 0.0, -8.0, 0.0]);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&t), &[0.0, 0.0, 0.0, 0.4, 0.0]);
    }

    #[test]
    fn test_dropout_2d() {
        let mut rng = StdRng::seed_from_u64(0);
        let t = Tensor2D::new([[0.05, 0.1, -0.2], [0.3, -0.4, 0.5]]);
        let r = t.trace().dropout(0.6, &mut rng);
        assert_close(r.data(), &[[0.125, 0.25, -0.5], [0.0, 0.0, 1.25]]);
        // NOTE: .exp() so we ensure result grad is used properly
        let gradients = r.exp().mean::<_, AllAxes>().backward();
        assert_eq!(
            gradients.ref_gradient(&t),
            &[[0.47214523, 0.5350107, 0.2527211], [0.0, 0.0, 1.4543099]]
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
                [[1.25, 1.25, 1.25], [1.25, 1.25, 1.25]],
                [[1.25, 1.25, 1.25], [1.25, 0.0, 1.25]],
                [[1.25, 1.25, 0.0], [1.25, 0.0, 1.25]],
                [[1.25, 1.25, 1.25], [1.25, 1.25, 1.25]]
            ]
        );
        let gradients = r.mean::<_, AllAxes>().backward();
        const V: f32 = 0.052083336;
        assert_eq!(
            gradients.ref_gradient(&t),
            &[
                [[V, V, V], [V, V, V]],
                [[V, V, V], [V, 0.0, V]],
                [[V, V, 0.0], [V, 0.0, V]],
                [[V, V, V], [V, V, V]]
            ]
        );
    }
}
