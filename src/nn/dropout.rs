use crate::prelude::*;
use rand::{prelude::StdRng, Rng, SeedableRng};

/// A [Module<Tensor>] that calls [dropout()] in [ModuleMut::forward_mut()] with probability `1.0 / N`.
/// It is the identity function in [Module::forward()]. Note that [dropout()] does not do anything
/// for tensors with [NoneTape], even in [ModuleMut].
///
/// Generics
/// - `N`: p is set as `1.0 / N`
///
/// Examples
/// ```rust
/// # use dfdx::prelude::*;
/// let mut dropout: DropoutOneIn<2> = Default::default();
/// let t: Tensor2D<2, 5> = Tensor2D::ones();
/// let r = dropout.forward_mut(t.trace());
/// assert_eq!(r.data(), &[[2.0, 2.0, 2.0, 0.0, 0.0], [2.0, 2.0, 0.0, 0.0, 2.0]]);
/// ```
#[derive(Clone, Debug)]
pub struct DropoutOneIn<const N: usize> {
    rng: StdRng,
}

impl<const N: usize> Default for DropoutOneIn<N> {
    /// Seeds [StdRng] with a new seed every time this is called. The seed comes from the [UniqueId] constructor.
    fn default() -> Self {
        let seed = unique_id().as_u64();
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl<const N: usize> CanUpdateWithGradients for DropoutOneIn<N> {
    /// Does nothing.
    fn update<G: GradientProvider>(&mut self, _: &mut G, _: &mut UnusedTensors) {}
}

impl<const N: usize> ResetParams for DropoutOneIn<N> {
    /// Does nothing.
    fn reset_params<R: Rng>(&mut self, _: &mut R) {}
}

impl<const N: usize> SaveToNpz for DropoutOneIn<N> {}
impl<const N: usize> LoadFromNpz for DropoutOneIn<N> {}

impl<const N: usize, T: Tensor<Dtype = f32, NoTape = T>> Module<T> for DropoutOneIn<N> {
    type Output = T;

    /// Identity function
    fn forward(&self, input: T) -> Self::Output {
        input
    }
}

impl<const N: usize, T: Tensor<Dtype = f32, OwnedTape = T>> ModuleMut<T> for DropoutOneIn<N> {
    type Output = T;

    /// Calls [dropout()] with `p=1/N` using `self.rng`.
    fn forward_mut(&mut self, input: T) -> Self::Output {
        dropout(input, 1.0 / N as f32, &mut self.rng)
    }
}

/// A [Module<Tensor>] that calls [dropout()] in [ModuleMut::forward_mut()] with probability `self.p`.
/// It is the identity function in [Module::forward()].
///
/// This also implements [Module<(Tensor, Rng)>] if you want to pass in an [Rng] externally, though
/// this may be harder to use and infect other modules.
///
/// [Default] is implemented as `p=0.5` and seeds.
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
    rng: StdRng,
}

impl Dropout {
    /// Constructs [Dropout] with `p` and `rng`.
    pub fn new(p: f32, rng_seed: u64) -> Self {
        Self {
            p,
            rng: StdRng::seed_from_u64(rng_seed),
        }
    }

    /// Constructs [Dropout] with `p` and a different seed every call.
    pub fn p(p: f32) -> Self {
        let seed = unique_id().as_u64();
        Self {
            p,
            rng: StdRng::seed_from_u64(seed),
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
    /// Does nothing.
    fn update<G: GradientProvider>(&mut self, _: &mut G, _: &mut UnusedTensors) {}
}

impl ResetParams for Dropout {
    /// Does nothing.
    fn reset_params<R: rand::Rng>(&mut self, _: &mut R) {}
}

impl SaveToNpz for Dropout {}
impl LoadFromNpz for Dropout {}

impl<T: Tensor<Dtype = f32, NoTape = T>> Module<T> for Dropout {
    type Output = T;

    /// Identity function
    fn forward(&self, input: T) -> Self::Output {
        input
    }
}
impl<T: Tensor<Dtype = f32, OwnedTape = T>> ModuleMut<T> for Dropout {
    type Output = T;

    /// Identity function
    fn forward_mut(&mut self, input: T) -> Self::Output {
        dropout(input, self.p, &mut self.rng)
    }
}

impl<R: Rng, T: Tensor<Dtype = f32, NoTape = T>> Module<(T, R)> for Dropout {
    type Output = (T, R);

    /// Identity function
    fn forward(&self, input: (T, R)) -> Self::Output {
        input
    }
}

impl<R: Rng, T: Tensor<Dtype = f32, OwnedTape = T>> ModuleMut<(T, R)> for Dropout {
    type Output = (T, R);

    /// Calls [dropout()] using `input.1`.
    fn forward_mut(&mut self, (t, mut rng): (T, R)) -> Self::Output {
        let t = dropout(t, self.p, &mut rng);
        (t, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dropout_internal_rng_reproduce() {
        let mut d1 = Dropout::new(0.5, 0);
        let mut d2 = Dropout::new(0.5, 0);
        let t: Tensor1D<100> = Tensor1D::ones();
        let r1 = d1.forward_mut(t.trace());
        let r2 = d2.forward_mut(t.trace());
        let r1_2 = d1.forward_mut(t.trace());
        assert_eq!(r1.data(), r2.data());
        assert_ne!(r1.data(), t.data());
        assert_ne!(r1.data(), r1_2.data());
        assert_ne!(r1_2.data(), t.data());
    }

    #[test]
    fn test_dropout_external_rng() {
        let rng = StdRng::seed_from_u64(0);
        let mut d = Dropout::p(0.5);
        let t: Tensor1D<100> = Tensor1D::ones();
        let (r, rng) = d.forward((t.clone(), rng));
        assert_eq!(t.data(), r.data());

        let (r, _rng) = d.forward_mut((t.trace(), rng));
        assert_ne!(t.data(), r.data());
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
        let mut dropout = Dropout::p(0.5);
        let t: Tensor1D<100> = Tensor1D::ones();
        let r = dropout.forward_mut(t.trace());
        assert!(t.data() != r.data());
    }
}
