use crate::{gradients::*, shapes::*, tensor::Tensor, tensor_ops::*};

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
/// # let dev: Cpu = Default::default();
/// let dropout: DropoutOneIn<2> = BuildModule::build(&dev);
/// dropout.forward(dev.zeros::<Rank1<5>>().trace());
/// ```
///
/// 2. Using [ModuleMut] with [NoneTape] **fails to compile**
/// ```compile_fail
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let mut dropout: DropoutOneIn<2> = Default::default();
/// dropout.forward_mut(dev.zeros::<Rank1<5>>());
/// ```
///
/// Generics:
/// - `N`: p is set as `1.0 / N`
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let mut dropout: DropoutOneIn<2> = Default::default();
/// let x: Tensor<Rank2<2, 5>, f32, _> = dev.ones();
/// let r = dropout.forward_mut(x.trace());
/// assert_eq!(r.array(), [[2.0, 2.0, 2.0, 0.0, 0.0], [2.0, 2.0, 0.0, 0.0, 2.0]]);
/// ```
#[derive(Clone, Debug, Default)]
pub struct DropoutOneIn<const N: usize>;

impl<const N: usize> ZeroSizedModule for DropoutOneIn<N> {}

impl<const N: usize, S: Shape, E: Dtype, D: Device<E>> Module<Tensor<S, E, D, NoneTape>>
    for DropoutOneIn<N>
{
    type Output = Tensor<S, E, D, NoneTape>;
    /// Does nothing
    type Error = D::Err;

    fn try_forward(&self, input: Tensor<S, E, D, NoneTape>) -> Result<Self::Output, D::Err> {
        Ok(input)
    }
}

impl<const N: usize, S: Shape, E: Dtype, D: Device<E>> ModuleMut<Tensor<S, E, D, OwnedTape<E, D>>>
    for DropoutOneIn<N>
{
    type Output = Tensor<S, E, D, OwnedTape<E, D>>;
    type Error = D::Err;

    /// Calls [dropout()] with `p=1/N` using `self.rng`.
    fn try_forward_mut(
        &mut self,
        input: Tensor<S, E, D, OwnedTape<E, D>>,
    ) -> Result<Self::Output, D::Err> {
        input.try_dropout(E::ONE / E::from_usize(N).unwrap())
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
/// # let dev: Cpu = Default::default();
/// let dropout: Dropout = Default::default();
/// dropout.forward(dev.zeros::<Rank1<5>>().trace());
/// ```
///
/// 2. Using [ModuleMut] with [NoneTape] **fails to compile**
/// ```compile_fail
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let mut dropout: Dropout = Default::default();
/// dropout.forward_mut(dev.zeros::<Rank1<5>>());
/// ```
///
/// Generics:
/// - `N`: p is set as `1.0 / N`
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let mut dropout = Dropout { p: 0.5 };
/// let x: Tensor<Rank2<2, 5>, f32, _> = dev.ones();
/// let r = dropout.forward_mut(x.trace());
/// assert_eq!(r.array(), [[2.0, 2.0, 2.0, 0.0, 0.0], [2.0, 2.0, 0.0, 0.0, 2.0]]);
/// ```
#[derive(Clone, Debug)]
pub struct Dropout {
    pub p: f32,
}

impl Default for Dropout {
    /// Sets `self.p` to `0.5`
    fn default() -> Self {
        Self { p: 0.5 }
    }
}

impl ZeroSizedModule for Dropout {}

impl<S: Shape, E: Dtype, D: Device<E>> Module<Tensor<S, E, D, NoneTape>> for Dropout {
    type Output = Tensor<S, E, D, NoneTape>;
    type Error = D::Err;

    /// Does nothing.
    fn try_forward(&self, input: Tensor<S, E, D, NoneTape>) -> Result<Self::Output, D::Err> {
        Ok(input)
    }
}

impl<S: Shape, E: Dtype, D: Device<E>> ModuleMut<Tensor<S, E, D, OwnedTape<E, D>>> for Dropout {
    type Output = Tensor<S, E, D, OwnedTape<E, D>>;
    type Error = D::Err;

    /// Calls [dropout()]
    fn try_forward_mut(
        &mut self,
        input: Tensor<S, E, D, OwnedTape<E, D>>,
    ) -> Result<Self::Output, D::Err> {
        input.try_dropout(E::from_f32(self.p).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        shapes::Rank1,
        tensor::{AsArray, OnesTensor},
        tests::*,
    };

    use super::*;

    #[test]
    fn test_dropout_internal_rng_reproduce() {
        let dev: TestDevice = Default::default();
        let mut d1 = Dropout { p: 0.5 };
        let mut d2 = Dropout { p: 0.5 };
        let t: Tensor<Rank1<100>, TestDtype, _> = dev.ones();
        let r1 = d1.forward_mut(t.trace());
        let r2 = d2.forward_mut(t.trace());
        let r1_2 = d1.forward_mut(t.trace());
        assert_ne!(r1.array(), r2.array());
        assert_ne!(r1.array(), r1_2.array());
    }

    #[test]
    fn test_dropout_no_tape() {
        let dev: TestDevice = Default::default();
        let dropout = Dropout { p: 0.5 };
        let t: Tensor<Rank1<100>, TestDtype, _> = dev.ones();
        let r = dropout.forward(t.clone());
        assert_eq!(t.array(), r.array());
    }

    #[test]
    fn test_dropout_tape() {
        let dev: TestDevice = Default::default();
        let mut dropout = Dropout { p: 0.5 };
        let t: Tensor<Rank1<100>, TestDtype, _> = dev.ones();
        let r = dropout.forward_mut(t.trace());
        assert_ne!(t.array(), r.array());
    }
}
