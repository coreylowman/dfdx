use crate::{
    gradients::{NoneTape, OwnedTape},
    shapes::*,
    tensor::Tensor,
    tensor_ops::{dropout, Device},
};

use super::{Module, ModuleMut, ZeroSizedModule};

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
#[derive(Clone, Debug, Default)]
pub struct DropoutOneIn<const N: usize>;

impl<const N: usize> ZeroSizedModule for DropoutOneIn<N> {}

impl<const N: usize, S: Shape, E: Dtype, D: Device<E>> Module<Tensor<S, E, D, NoneTape>>
    for DropoutOneIn<N>
{
    type Output = Tensor<S, E, D, NoneTape>;
    /// Does nothing
    fn forward(&self, input: Tensor<S, E, D, NoneTape>) -> Self::Output {
        input
    }
}

impl<const N: usize, S: Shape, E: Dtype, D: Device<E>> ModuleMut<Tensor<S, E, D, OwnedTape<D>>>
    for DropoutOneIn<N>
{
    type Output = Tensor<S, E, D, OwnedTape<D>>;
    /// Calls [dropout()] with `p=1/N` using `self.rng`.
    fn forward_mut(&mut self, input: Tensor<S, E, D, OwnedTape<D>>) -> Self::Output {
        dropout(input, 1.0 / N as f32)
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
}

impl Default for Dropout {
    /// Sets `self.p` to `0.5`, and seeds [StdRng] with 0.
    fn default() -> Self {
        Self { p: 0.5 }
    }
}

impl ZeroSizedModule for Dropout {}

impl<S: Shape, E: Dtype, D: Device<E>> Module<Tensor<S, E, D, NoneTape>> for Dropout {
    type Output = Tensor<S, E, D, NoneTape>;
    /// Does nothing.
    fn forward(&self, input: Tensor<S, E, D, NoneTape>) -> Self::Output {
        input
    }
}

impl<S: Shape, E: Dtype, D: Device<E>> ModuleMut<Tensor<S, E, D, OwnedTape<D>>> for Dropout {
    type Output = Tensor<S, E, D, OwnedTape<D>>;
    /// Calls [dropout()]
    fn forward_mut(&mut self, input: Tensor<S, E, D, OwnedTape<D>>) -> Self::Output {
        dropout(input, self.p)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        shapes::Rank1,
        tensor::{AsArray, OnesTensor},
        tests::build_test_device,
    };

    use super::*;

    #[test]
    fn test_dropout_internal_rng_reproduce() {
        let dev = build_test_device!();
        let mut d1 = Dropout { p: 0.5 };
        let mut d2 = Dropout { p: 0.5 };
        let t = dev.ones::<Rank1<100>>();
        let r1 = d1.forward_mut(t.trace());
        let r2 = d2.forward_mut(t.trace());
        let r1_2 = d1.forward_mut(t.trace());
        assert_ne!(r1.array(), r2.array());
        assert_ne!(r1.array(), r1_2.array());
    }

    #[test]
    fn test_dropout_no_tape() {
        let dev = build_test_device!();
        let dropout = Dropout { p: 0.5 };
        let t = dev.ones::<Rank1<100>>();
        let r = dropout.forward(t.clone());
        assert_eq!(t.array(), r.array());
    }

    #[test]
    fn test_dropout_tape() {
        let dev = build_test_device!();
        let mut dropout = Dropout { p: 0.5 };
        let t = dev.ones::<Rank1<100>>();
        let r = dropout.forward_mut(t.trace());
        assert_ne!(t.array(), r.array());
    }
}
