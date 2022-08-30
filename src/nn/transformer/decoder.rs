use rand::Rng;

use crate::prelude::*;

/// **Requires Nightly** A transformer decoder block. Different than the normal transformer block
/// as this self attention accepts an additional sequence from the encoder.
///
/// # Generics
/// - `M` The embedding size of token vectors from decoder.
/// - `N` The embedding size of token vectors from encoder.
/// - `K` The size of the keys in self attention.
/// - `H` The number of attention heads.
/// TODO: Doctests
#[derive(Default, Debug)]
pub struct TransformerDecoderBlock<
    const MODEL_DIM: usize,
    const NUM_HEADS: usize,
    const FF_DIM: usize,
> {
    pub self_attn: MultiHeadAttention<MODEL_DIM, NUM_HEADS>,
    pub l1: LayerNorm1D<MODEL_DIM>,
    pub mha_attn: MultiHeadAttention<MODEL_DIM, NUM_HEADS>,
    pub ff: FF<MODEL_DIM, FF_DIM>,
}

type FF<const M: usize, const F: usize> = (
    LayerNorm1D<M>,
    Residual<(Linear<M, F>, ReLU, Linear<F, M>)>,
    LayerNorm1D<M>,
);

impl<const MODEL_DIM: usize, const NUM_HEADS: usize, const FF_DIM: usize> ResetParams
    for TransformerDecoderBlock<MODEL_DIM, NUM_HEADS, FF_DIM>
{
    fn reset_params<R: Rng>(&mut self, rng: &mut R) {
        self.self_attn.reset_params(rng);
        self.l1.reset_params(rng);
        self.mha_attn.reset_params(rng);
        self.ff.reset_params(rng);
    }
}

impl<const MODEL_DIM: usize, const NUM_HEADS: usize, const FF_DIM: usize> CanUpdateWithGradients
    for TransformerDecoderBlock<MODEL_DIM, NUM_HEADS, FF_DIM>
{
    fn update<G: GradientProvider>(&mut self, grads: &mut G, unused: &mut UnusedTensors) {
        self.self_attn.update(grads, unused);
        self.l1.update(grads, unused);
        self.mha_attn.update(grads, unused);
        self.ff.update(grads, unused);
    }
}

impl<const M: usize, const H: usize, const F: usize, Tgt, Mem> Module<(Tgt, Mem)>
    for TransformerDecoderBlock<M, H, F>
where
    Tgt: Tensor<Dtype = f32>,
    Mem: Tensor<Dtype = f32, NoTape = Mem>,
    MultiHeadAttention<M, H>: Module<(Tgt, Tgt::NoTape, Tgt::NoTape), Output = Tgt>
        + Module<(Tgt, Mem, Mem), Output = Tgt>,
    LayerNorm1D<M>: Module<Tgt, Output = Tgt>,
    FF<M, F>: Module<Tgt, Output = Tgt>,
{
    type Output = Tgt;

    fn forward(&self, (tgt, mem): (Tgt, Mem)) -> Self::Output {
        let (tgt, tape) = tgt.split_tape();
        let x = self.self_attn.forward((
            tgt.duplicate().put_tape(tape),
            tgt.duplicate(),
            tgt.duplicate(),
        ));
        let x = add(x, &tgt);
        let x = self.l1.forward(x);

        let x_ = x.duplicate();
        let x = self.mha_attn.forward((x, mem.duplicate(), mem));
        let x = add(x, &x_);
        self.ff.forward(x)
    }
}

/// **Requires Nightly** A transformer decoder.
///
/// # Generics
/// - `M` The embedding size of token vectors.
/// - `N` The size of encoder vectors.
/// - `I` The inner size of the feedforward layers.
/// - `L` The number of layers.
/// - `H` The number of heads for self attention.
/// TODO: Doctests
#[derive(Debug, Default)]
pub struct TransformerDecoder<
    const MODEL_DIM: usize,
    const NUM_HEADS: usize,
    const FF_DIM: usize,
    const NUM_LAYERS: usize,
>(pub Repeated<TransformerDecoderBlock<MODEL_DIM, NUM_HEADS, FF_DIM>, NUM_LAYERS>);

impl<const M: usize, const H: usize, const F: usize, const L: usize> ResetParams
    for TransformerDecoder<M, H, F, L>
{
    fn reset_params<R: Rng>(&mut self, rng: &mut R) {
        self.0.reset_params(rng);
    }
}

impl<const M: usize, const H: usize, const F: usize, const L: usize> CanUpdateWithGradients
    for TransformerDecoder<M, H, F, L>
{
    fn update<G: GradientProvider>(&mut self, grads: &mut G, unused: &mut UnusedTensors) {
        self.0.update(grads, unused);
    }
}

impl<const M: usize, const H: usize, const F: usize, const L: usize, Tgt, Mem> Module<(Tgt, Mem)>
    for TransformerDecoder<M, H, F, L>
where
    Mem: Tensor<NoTape = Mem>,
    TransformerDecoderBlock<M, H, F>: Module<(Tgt, Mem), Output = Tgt>,
{
    type Output = Tgt;

    fn forward(&self, (mut x, mem): (Tgt, Mem)) -> Self::Output {
        for block in self.0.modules.iter() {
            x = block.forward((x, mem.duplicate()));
        }
        x
    }
}
