use crate::{Bias1DConfig, MatMulConfig, Sequential};

use dfdx::shapes::{Const, Dim};

/// A linear transformation of the form `weight * x + bias`, where `weight` is a matrix, `x` is a vector or matrix,
/// and `bias` is a vector.
///
/// Generics:
/// - `I` The "input" size of vectors & matrices.
/// - `O` The "output" size of vectors & matrices.
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let arch: LinearConstConfig<5, 2> = Default::default();
/// let model = dev.build_module_ext::<f32>(arch);
/// // single item forward
/// let _: Tensor<Rank1<2>, f32, _> = model.forward(dev.zeros::<Rank1<5>>());
/// // batched forward
/// let _: Tensor<Rank2<10, 2>, f32, _> = model.forward(dev.zeros::<Rank2<10, 5>>());
/// ```
#[derive(Default, Debug, Clone, Copy, Sequential)]
#[built(Linear)]
pub struct LinearConfig<I: Dim, O: Dim> {
    pub matmul: MatMulConfig<I, O>,
    pub bias: Bias1DConfig<O>,
}

/// Compile time sugar alias around [LinearConfig].
pub type LinearConstConfig<const I: usize, const O: usize> = LinearConfig<Const<I>, Const<O>>;

impl<I: Dim, O: Dim> LinearConfig<I, O> {
    pub fn new(inp: I, out: O) -> Self {
        Self {
            matmul: MatMulConfig { inp, out },
            bias: Bias1DConfig(out),
        }
    }
}
