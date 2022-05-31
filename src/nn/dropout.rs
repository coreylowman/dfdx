use crate::prelude::*;
use rand::{prelude::StdRng, Rng, SeedableRng};
use rand_distr::Distribution;
use std::{cell::RefCell, ops::DerefMut};

/// A [Module<Tensor>] that calls [dropout()] in [Module::forward()] with probability `self.p`.
///
/// This also implements [Module<(Tensor, Rng)>] if you want to pass in an [Rng] externally, though
/// this may be harder to use and infect other modules.
///
/// [Default] is implemented as `p=0.5` and seeds.
///
/// Implementation details:
/// This stores the [Rng] in a [RefCell] to maintain compatibility with forward taking
/// a non-mutable reference to self.
///
/// Example:
///
/// ```rust
/// # use dfdx::prelude::*;
/// type MlpWithDropout = (
///     Dropout,
///     Linear<5, 10>,
///     ReLU,
///     Dropout,
/// );
///
/// // or
/// let my_mlp = (
///     Dropout::p(0.5),
///     Linear::<5, 10>::default(),
///     ReLU::default(),
///     Dropout::p(0.1),
/// );
/// ```
#[derive(Clone, Debug)]
pub struct Dropout {
    pub p: f32,
    rng: RefCell<StdRng>,
}

impl Dropout {
    /// Constructs [Dropout] with `p` and `rng`.
    pub fn new(p: f32, rng_seed: u64) -> Self {
        Self {
            p,
            rng: RefCell::new(StdRng::seed_from_u64(rng_seed)),
        }
    }

    /// Constructs [Dropout] with `p` and `StdRng::seed_from_u64(0)`.
    pub fn p(p: f32) -> Self {
        Self {
            p,
            rng: RefCell::new(StdRng::seed_from_u64(0)),
        }
    }
}

impl Default for Dropout {
    /// Sets `self.p` to `0.5`, and seeds [StdRng] with 0.
    fn default() -> Self {
        Self::new(0.5, 0)
    }
}

impl CanUpdateWithGradients for Dropout {
    fn update<G: GradientProvider>(&mut self, _: &mut G) {}
}

impl Randomize<f32> for Dropout {
    fn randomize<R: Rng, D: Distribution<f32>>(&mut self, _: &mut R, _: &D) {}
}

impl<T: Tensor<Dtype = f32>> Module<T> for Dropout {
    type Output = T;

    /// Calls [dropout()] using `self.rng`.
    fn forward(&self, input: T) -> Self::Output {
        let mut rng = self.rng.borrow_mut();
        dropout(input, self.p, rng.deref_mut())
    }
}

impl<R: Rng + SeedableRng, T: Tensor<Dtype = f32>> Module<(T, R)> for Dropout {
    type Output = (T, R);

    /// Calls [dropout()] using `input.1`.
    fn forward(&self, input: (T, R)) -> Self::Output {
        let (t, mut rng) = input;
        let t = dropout(t, self.p, &mut rng);
        (t, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dropout_internal_rng_reproduce() {
        let d1 = Dropout::p(0.5);
        let d2 = Dropout::p(0.5);
        let t: Tensor1D<100> = Tensor1D::ones();
        let r1 = d1.forward(t.trace());
        let r2 = d2.forward(t.trace());
        let r1_2 = d1.forward(t.trace());
        assert_eq!(r1.data(), r2.data());
        assert!(r1.data() != r1_2.data());
    }

    #[test]
    fn test_dropout_external_rng() {
        let rng = StdRng::seed_from_u64(0);
        let d = Dropout::p(0.5);
        let t: Tensor1D<100> = Tensor1D::ones();
        let (r, _rng) = d.forward((t.trace(), rng));
        assert!(t.data() != r.data());
    }

    #[test]
    fn test_dropout_no_tape() {
        let dropout = Dropout::p(0.5);
        let t: Tensor1D<100> = Tensor1D::ones();
        let r = dropout.forward(t.clone());
        assert_eq!(t.data(), r.data());
    }

    #[test]
    fn test_dropout_tape() {
        let dropout = Dropout::p(0.5);
        let t: Tensor1D<100> = Tensor1D::ones();
        let r = dropout.forward(t.trace());
        assert!(t.data() != r.data());
    }
}
