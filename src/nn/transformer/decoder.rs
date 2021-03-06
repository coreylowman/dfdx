use rand::Rng;

use crate::prelude::*;

/// A transformer decoder block. Different than the normal transformer block as this self attention accepts an additional sequence from the encoder.
///
/// # Generics
/// - `M` The embedding size of token vectors from decoder.
/// - `N` The embedding size of token vectors from encoder.
/// - `K` The size of the keys in self attention.
/// - `H` The number of attention heads.
/// TODO: Doctests
#[derive(Default, Debug)]
pub struct TransformerDecoderBlock<
    const M: usize,
    const N: usize,
    const I: usize,
    const K: usize,
    const H: usize,
> where
    Assert<{ M % H == 0 }>: ConstTrue,
    Assert<{ K % H == 0 }>: ConstTrue,
{
    pub attn: MultiHeadAttention<M, N, K, M, H>,
    pub ff: (
        LayerNorm1D<M>,
        Residual<(Linear<M, I>, ReLU, Linear<I, M>)>,
        LayerNorm1D<M>,
    ),
}

impl<const M: usize, const N: usize, const I: usize, const K: usize, const H: usize> ResetParams
    for TransformerDecoderBlock<M, N, I, K, H>
where
    Assert<{ M % H == 0 }>: ConstTrue,
    Assert<{ K % H == 0 }>: ConstTrue,
{
    fn reset_params<R: Rng>(&mut self, rng: &mut R) {
        self.attn.reset_params(rng);
        self.ff.reset_params(rng);
    }
}

impl<const M: usize, const N: usize, const I: usize, const K: usize, const H: usize>
    CanUpdateWithGradients for TransformerDecoderBlock<M, N, I, K, H>
where
    Assert<{ M % H == 0 }>: ConstTrue,
    Assert<{ K % H == 0 }>: ConstTrue,
{
    fn update<G: GradientProvider>(&mut self, grads: &mut G) {
        self.attn.update(grads);
        self.ff.update(grads);
    }
}

impl<
        const M: usize,
        const N: usize,
        const I: usize,
        const K: usize,
        const H: usize,
        const S1: usize,
        const S2: usize,
        T: Tape,
    > Module<(Tensor2D<S1, M, T>, Tensor2D<S2, N>)> for TransformerDecoderBlock<M, N, I, K, H>
where
    Assert<{ M % H == 0 }>: ConstTrue,
    Assert<{ K % H == 0 }>: ConstTrue,
    Assert<{ S1 * K == H * S1 * (K / H) }>: ConstTrue,
    Assert<{ S2 * K == H * S2 * (K / H) }>: ConstTrue,
    Assert<{ S2 * M == H * S2 * (M / H) }>: ConstTrue,
    Assert<{ H * S1 * (M / H) == S1 * M }>: ConstTrue,
{
    type Output = Tensor2D<S1, M, T>;

    fn forward(&self, (input, from_enc): (Tensor2D<S1, M, T>, Tensor2D<S2, N>)) -> Self::Output {
        let (input, input_tape) = input.split_tape();
        self.ff.forward(add(
            self.attn
                .forward((input.duplicate().put_tape(input_tape), from_enc)),
            &input,
        ))
    }
}

impl<
        const M: usize,
        const N: usize,
        const I: usize,
        const K: usize,
        const H: usize,
        const S1: usize,
        const S2: usize,
        const B: usize,
        T: Tape,
    > Module<(Tensor3D<B, S1, M, T>, Tensor3D<B, S2, N>)> for TransformerDecoderBlock<M, N, I, K, H>
where
    Assert<{ M % H == 0 }>: ConstTrue,
    Assert<{ K % H == 0 }>: ConstTrue,
    Assert<{ B * S1 * K == B * H * S1 * (K / H) }>: ConstTrue,
    Assert<{ B * S2 * K == B * H * S2 * (K / H) }>: ConstTrue,
    Assert<{ B * S2 * M == B * H * S2 * (M / H) }>: ConstTrue,
    Assert<{ B * H * S1 * (M / H) == B * S1 * M }>: ConstTrue,
{
    type Output = Tensor3D<B, S1, M, T>;

    fn forward(
        &self,
        (input, from_enc): (Tensor3D<B, S1, M, T>, Tensor3D<B, S2, N>),
    ) -> Self::Output {
        let (input, input_tape) = input.split_tape();
        let x = self
            .attn
            .forward((input.duplicate().put_tape(input_tape), from_enc));
        self.ff.forward(add(x, &input))
    }
}

/// A transformer decoder.
///
/// # Generics
/// - `M` The embedding size of token vectors.
/// - `N` The size of encoder vectors.
/// - `I` The inner size of the feedforward layers.
/// - `L` The number of layers.
/// - `H` The number of heads for self attention.
/// TODO: Doctests
#[derive(Debug)]
pub struct TransformerDecoder<
    const M: usize,
    const N: usize,
    const I: usize,
    const L: usize,
    const H: usize,
> where
    Assert<{ M % H == 0 }>: ConstTrue,
{
    pub blocks: [TransformerDecoderBlock<M, N, I, M, H>; L],
}

impl<const M: usize, const N: usize, const I: usize, const L: usize, const H: usize> Default
    for TransformerDecoder<M, N, I, L, H>
where
    Assert<{ M % H == 0 }>: ConstTrue,
    [TransformerDecoderBlock<M, N, I, M, H>; L]: Default,
{
    fn default() -> Self {
        Self {
            blocks: Default::default(),
        }
    }
}

impl<const M: usize, const N: usize, const I: usize, const L: usize, const H: usize> ResetParams
    for TransformerDecoder<M, N, I, L, H>
where
    Assert<{ M % H == 0 }>: ConstTrue,
{
    fn reset_params<R: Rng>(&mut self, rng: &mut R) {
        for block in &mut self.blocks {
            block.reset_params(rng);
        }
    }
}

impl<const M: usize, const N: usize, const I: usize, const L: usize, const H: usize>
    CanUpdateWithGradients for TransformerDecoder<M, N, I, L, H>
where
    Assert<{ M % H == 0 }>: ConstTrue,
{
    fn update<G: GradientProvider>(&mut self, grads: &mut G) {
        for block in &mut self.blocks {
            block.update(grads);
        }
    }
}

impl<
        const M: usize,
        const N: usize,
        const I: usize,
        const L: usize,
        const H: usize,
        const S1: usize,
        const S2: usize,
        T: Tape,
    > Module<(Tensor2D<S1, M, T>, Tensor2D<S2, N, T>)> for TransformerDecoder<M, N, I, L, H>
where
    Assert<{ M % H == 0 }>: ConstTrue,
    Assert<{ S1 * M == H * S1 * (M / H) }>: ConstTrue,
    Assert<{ S2 * M == H * S2 * (M / H) }>: ConstTrue,
    Assert<{ H * S1 * (M / H) == S1 * M }>: ConstTrue,
{
    type Output = Tensor2D<S1, M, T>;

    fn forward(&self, input: (Tensor2D<S1, M, T>, Tensor2D<S2, N, T>)) -> Self::Output {
        let (mut x, from_enc) = input;
        for block in &self.blocks {
            x = block.forward((x, from_enc.duplicate()));
        }
        x
    }
}

impl<
        const M: usize,
        const N: usize,
        const I: usize,
        const L: usize,
        const H: usize,
        const S1: usize,
        const S2: usize,
        const B: usize,
        T: Tape,
    > Module<(Tensor3D<B, S1, M, T>, Tensor3D<B, S2, N>)> for TransformerDecoder<M, N, I, L, H>
where
    Assert<{ M % H == 0 }>: ConstTrue,
    Assert<{ B * S1 * M == B * H * S1 * (M / H) }>: ConstTrue,
    Assert<{ B * S2 * M == B * H * S2 * (M / H) }>: ConstTrue,
    Assert<{ B * H * S1 * (M / H) == B * S1 * M }>: ConstTrue,
{
    type Output = Tensor3D<B, S1, M, T>;

    fn forward(&self, input: (Tensor3D<B, S1, M, T>, Tensor3D<B, S2, N>)) -> Self::Output {
        let (mut x, from_enc) = input;
        for block in &self.blocks {
            x = block.forward((x, from_enc.duplicate()));
        }
        x
    }
}
