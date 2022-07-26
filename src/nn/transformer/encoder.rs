use rand::Rng;

use crate::prelude::*;

pub type TransformerEncoderBlock<const M: usize, const I: usize, const K: usize, const H: usize> = (
    Residual<MultiHeadAttention<M, M, K, M, H>>,
    LayerNorm1D<M>,
    Residual<(Linear<M, I>, ReLU, Linear<I, M>)>,
    LayerNorm1D<M>,
);

/// A transformer encoder.
///
/// # Generics
/// - `M` The embedding size of token vectors.
/// - `I` The inner size of the feedforward layers.
/// - `L` The number of layers.
/// - `H` The number of heads for self attention.
/// TODO: Doctests
#[derive(Debug, Clone)]
pub struct TransformerEncoder<const M: usize, const I: usize, const L: usize, const H: usize>
where
    Assert<{ M % H == 0 }>: ConstTrue,
{
    pub blocks: Repeated<TransformerEncoderBlock<M, I, M, H>, L>,
}

impl<const M: usize, const I: usize, const L: usize, const H: usize> Default
    for TransformerEncoder<M, I, L, H>
where
    Assert<{ M % H == 0 }>: ConstTrue,
    [TransformerEncoderBlock<M, I, M, H>; L]: Default,
{
    fn default() -> Self {
        Self {
            blocks: Default::default(),
        }
    }
}

impl<const M: usize, const I: usize, const L: usize, const H: usize> ResetParams
    for TransformerEncoder<M, I, L, H>
where
    Assert<{ M % H == 0 }>: ConstTrue,
{
    fn reset_params<R: Rng>(&mut self, rng: &mut R) {
        self.blocks.reset_params(rng);
    }
}

impl<const M: usize, const I: usize, const L: usize, const H: usize> CanUpdateWithGradients
    for TransformerEncoder<M, I, L, H>
where
    Assert<{ M % H == 0 }>: ConstTrue,
{
    fn update<G: GradientProvider>(&mut self, grads: &mut G) {
        self.blocks.update(grads);
    }
}

impl<const S: usize, const M: usize, const I: usize, const L: usize, const H: usize>
    Module<Tensor2D<S, M>> for TransformerEncoder<M, I, L, H>
where
    Assert<{ M % H == 0 }>: ConstTrue,
    Assert<{ S * M == S * H * (M / H) }>: ConstTrue,
    Assert<{ H * S * (M / H) == S * M }>: ConstTrue,
    Assert<{ S * M == H * S * (M / H) }>: ConstTrue,
{
    type Output = Tensor2D<S, M>;

    fn forward(&self, input: Tensor2D<S, M>) -> Self::Output {
        self.blocks.forward(input)
    }
}

impl<
        const B: usize,
        const S: usize,
        const M: usize,
        const I: usize,
        const H: usize,
        const L: usize,
    > Module<Tensor3D<B, S, M>> for TransformerEncoder<M, I, L, H>
where
    Assert<{ M % H == 0 }>: ConstTrue,
    Assert<{ B * S * M == B * S * H * (M / H) }>: ConstTrue,
    Assert<{ B * H * S * (M / H) == B * S * M }>: ConstTrue,
    Assert<{ B * S * M == B * H * S * (M / H) }>: ConstTrue,
{
    type Output = Tensor3D<B, S, M>;

    fn forward(&self, input: Tensor3D<B, S, M>) -> Self::Output {
        self.blocks.forward(input)
    }
}
