use super::*;
use crate::gradients::*;
use crate::tensor::*;
use crate::tensor_ops::dropout;
use crate::unique_id::unique_id;
use rand::prelude::*;

/// Does nothing as a [Module], and calls [dropout()] as [ModuleMut] with probability `1.0 / N`.
///
/// To prevent programmer error, [Module] and [ModuleMut] are only implemented for specific tapes:
/// 1. [Module] requires that the input tensor has a [NoneTape]. i.e. that gradients are not being
///    tracked.
/// 2. [ModuleMut] requires that the tensor has a [OwnedTape]. i.e. that the gradients are being
///    tracked
///
/// That means the following will fail to compile:
///
/// 1. Using [Module] with [OwnedTape] **fails to compile**
/// ```compile_fail
/// # use dfdx::prelude::*;
/// let dropout: DropoutOneIn<2> = Default::default();
/// let x: Tensor1D<5, OwnedTape> = Tensor1D::zeros().trace();
/// dropout.forward(x);
/// ```
///
/// 2. Using [ModuleMut] with [NoneTape] **fails to compile**
/// ```compile_fail
/// # use dfdx::prelude::*;
/// let mut dropout: DropoutOneIn<2> = Default::default();
/// let x: Tensor1D<5, NoneTape> = TensorCreator::zeros();
/// dropout.forward_mut(x);
/// ```
///
/// Generics:
/// - `N`: p is set as `1.0 / N`
///
/// Examples:
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
    /// Seeds [StdRng] with a new seed every time this is called. The seed is initialized
    /// with a deterministic value.
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

impl<const N: usize, T: Tensor<Dtype = f32, Tape = NoneTape>> Module<T> for DropoutOneIn<N> {
    type Output = T;
    /// Does nothing
    fn forward(&self, input: T) -> Self::Output {
        input
    }
}

impl<const N: usize, T: Tensor<Dtype = f32, Tape = OwnedTape>> ModuleMut<T> for DropoutOneIn<N> {
    type Output = T;
    /// Calls [dropout()] with `p=1/N` using `self.rng`.
    fn forward_mut(&mut self, input: T) -> Self::Output {
        dropout(input, 1.0 / N as f32, &mut self.rng)
    }
}

/// Does nothing as a [Module], and calls [dropout()] as [ModuleMut] with probability `1.0 / N`.
///
/// To prevent programmer error, [Module] and [ModuleMut] are only implemented for specific tapes:
/// 1. [Module] requires that the input tensor has a [NoneTape]. i.e. that gradients are not being
///    tracked.
/// 2. [ModuleMut] requires that the tensor has a [OwnedTape]. i.e. that the gradients are being
///    tracked
///
/// That means the following will fail to compile:
///
/// 1. Using [Module] with [OwnedTape] **fails to compile**
/// ```compile_fail
/// # use dfdx::prelude::*;
/// let dropout: Dropout = Default::default();
/// let x: Tensor1D<5, OwnedTape> = Tensor1D::zeros().trace();
/// dropout.forward(x);
/// ```
///
/// 2. Using [ModuleMut] with [NoneTape] **fails to compile**
/// ```compile_fail
/// # use dfdx::prelude::*;
/// let mut dropout: Dropout = Default::default();
/// let x: Tensor1D<5, NoneTape> = TensorCreator::zeros();
/// dropout.forward_mut(x);
/// ```
///
/// Generics:
/// - `N`: p is set as `1.0 / N`
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let mut dropout = Dropout::new(0.5, 0);
/// let t: Tensor2D<2, 5> = Tensor2D::ones();
/// let r = dropout.forward_mut(t.trace());
/// assert_eq!(r.data(), &[[2.0, 2.0, 2.0, 0.0, 0.0], [2.0, 2.0, 0.0, 0.0, 2.0]]);
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

impl<T: Tensor<Dtype = f32, Tape = NoneTape>> Module<T> for Dropout {
    type Output = T;
    /// Does nothing.
    fn forward(&self, input: T) -> Self::Output {
        input
    }
}

impl<T: Tensor<Dtype = f32, Tape = OwnedTape>> ModuleMut<T> for Dropout {
    type Output = T;
    /// Calls [dropout()]
    fn forward_mut(&mut self, input: T) -> Self::Output {
        dropout(input, self.p, &mut self.rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrays::HasArrayData;

    #[test]
    fn test_dropout_internal_rng_reproduce() {
        let mut d1 = Dropout::new(0.5, 0);
        let mut d2 = Dropout::new(0.5, 0);
        let t: Tensor1D<100> = Tensor1D::ones();
        let r1 = d1.forward_mut(t.trace());
        let r2 = d2.forward_mut(t.trace());
        let r1_2 = d1.forward_mut(t.trace());
        assert_eq!(r1.data(), r2.data());
        assert!(r1.data() != r1_2.data());
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
