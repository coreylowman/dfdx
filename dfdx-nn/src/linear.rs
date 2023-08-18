use crate::{Bias1DConfig, MatMulConfig, Sequential};

use dfdx::shapes::{Const, Dim};

#[derive(Default, Debug, Clone, Copy, Sequential)]
#[built(Linear)]
pub struct LinearConfig<I: Dim, O: Dim> {
    pub matmul: MatMulConfig<I, O>,
    pub bias: Bias1DConfig<O>,
}

pub type LinearConstConfig<const I: usize, const O: usize> = LinearConfig<Const<I>, Const<O>>;

impl<I: Dim, O: Dim> LinearConfig<I, O> {
    pub fn new(inp: I, out: O) -> Self {
        Self {
            matmul: MatMulConfig { inp, out },
            bias: Bias1DConfig(out),
        }
    }
}
