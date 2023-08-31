use dfdx::prelude::*;

use crate::*;

/// Calls [dfdx::tensor_ops::dropout()] with `p = 1.0 / N` in [Module::forward_mut()], and does nothing in  [Module::forward()].
///
/// Generics:
/// - `N`: p is set as `1.0 / N`
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// # use dfdx_nn::*;
/// # let dev: Cpu = Default::default();
/// let mut dropout: DropoutOneIn<2> = Default::default();
/// let grads = dropout.alloc_grads();
/// let x: Tensor<Rank2<2, 5>, f32, _> = dev.ones();
/// let r = dropout.forward_mut(x.trace(grads));
/// assert_eq!(r.array(), [[2.0, 0.0, 2.0, 0.0, 2.0], [0.0, 2.0, 0.0, 2.0, 2.0]]);
/// ```
#[derive(Clone, Debug, Default, CustomModule)]
pub struct DropoutOneIn<const N: usize>;

impl<const N: usize, S: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<S, E, D, T>>
    for DropoutOneIn<N>
{
    type Output = Tensor<S, E, D, T>;
    type Error = D::Err;

    /// Does nothing
    fn try_forward(&self, input: Tensor<S, E, D, T>) -> Result<Self::Output, D::Err> {
        assert!(
            !T::OWNS_TAPE,
            "DropoutOneIn::try_forward input must not be traced."
        );
        Ok(input)
    }

    /// Applies dropout to the input tensor.
    fn try_forward_mut(&mut self, x: Tensor<S, E, D, T>) -> Result<Self::Output, Self::Error> {
        assert!(
            T::OWNS_TAPE,
            "DropoutOneIn::try_forward_mut input must be traced."
        );
        x.try_dropout(1.0 / N as f64)
    }
}

/// Calls [dfdx::tensor_ops::dropout()] in [Module::forward_mut()], and does nothing in  [Module::forward()].
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// # use dfdx_nn::*;
/// # let dev: Cpu = Default::default();
/// let mut dropout = Dropout { p: 0.5 };
/// let grads = dropout.alloc_grads();
/// let x: Tensor<Rank2<2, 5>, f32, _> = dev.ones();
/// let r = dropout.forward_mut(x.trace(grads));
/// assert_eq!(r.array(), [[2.0, 0.0, 2.0, 0.0, 2.0], [0.0, 2.0, 0.0, 2.0, 2.0]]);
/// ```
#[derive(Clone, Debug, CustomModule)]
pub struct Dropout {
    pub p: f64,
}

impl Default for Dropout {
    /// Sets `self.p` to `0.5`
    fn default() -> Self {
        Self { p: 0.5 }
    }
}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<S, E, D, T>> for Dropout {
    type Output = Tensor<S, E, D, T>;
    type Error = D::Err;

    /// Does nothing
    fn try_forward(&self, input: Tensor<S, E, D, T>) -> Result<Self::Output, D::Err> {
        assert!(
            !T::OWNS_TAPE,
            "Dropout::try_forward input must not be traced."
        );
        Ok(input)
    }

    /// Applies dropout to the input tensor.
    fn try_forward_mut(&mut self, x: Tensor<S, E, D, T>) -> Result<Self::Output, Self::Error> {
        assert!(
            T::OWNS_TAPE,
            "Dropout::try_forward_mut input must be traced."
        );
        x.try_dropout(self.p)
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::*;

    use super::*;

    #[test]
    fn test_dropout_internal_rng_reproduce() {
        let dev: TestDevice = Default::default();
        let mut d1 = Dropout { p: 0.5 };
        let mut d2 = Dropout { p: 0.5 };
        let t: Tensor<Rank1<100>, TestDtype, _> = dev.ones();
        let r1 = d1.forward_mut(t.leaky_trace());
        let r2 = d2.forward_mut(t.leaky_trace());
        let r1_2 = d1.forward_mut(t.leaky_trace());
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
        let r = dropout.forward_mut(t.leaky_trace());
        assert_ne!(t.array(), r.array());
    }

    #[test]
    #[should_panic = "Dropout::try_forward input must not be traced."]
    fn test_dropout_forward_with_tape() {
        let dev: TestDevice = Default::default();
        let dropout = Dropout { p: 0.5 };
        let t: Tensor<Rank1<100>, TestDtype, _> = dev.ones();
        let _ = dropout.forward(t.leaky_trace());
    }

    #[test]
    #[should_panic = "Dropout::try_forward_mut input must be traced."]
    fn test_dropout_forward_mut_without_tape() {
        let dev: TestDevice = Default::default();
        let mut dropout = Dropout { p: 0.5 };
        let t: Tensor<Rank1<100>, TestDtype, _> = dev.ones();
        let _ = dropout.forward_mut(t);
    }
}
