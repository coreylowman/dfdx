use dfdx::{
    shapes::{Dtype, Shape},
    tensor::{Tape, Tensor},
    tensor_ops::Device,
};

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
        assert!(!T::OWNS_TAPE);
        Ok(input)
    }

    /// Applies dropout to the input tensor.
    fn try_forward_mut(&mut self, x: Tensor<S, E, D, T>) -> Result<Self::Output, Self::Error> {
        assert!(T::OWNS_TAPE);
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
        assert!(!T::OWNS_TAPE);
        Ok(input)
    }

    /// Applies dropout to the input tensor.
    fn try_forward_mut(&mut self, x: Tensor<S, E, D, T>) -> Result<Self::Output, Self::Error> {
        assert!(T::OWNS_TAPE);
        x.try_dropout(self.p)
    }
}
