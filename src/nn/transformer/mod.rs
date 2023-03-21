mod decoder;
mod encoder;
mod mha;

pub use decoder::*;
pub use encoder::*;
pub use mha::*;

use num_traits::Float;
use rand_distr::uniform::SampleUniform;

use crate::{shapes::*, tensor::*, tensor_ops::*};

use super::*;

pub mod builder {
    #[derive(Debug, Clone)]
    pub struct Transformer<
        const MODEL_DIM: usize,
        const NUM_HEADS: usize,
        const NUM_ENCODER_LAYERS: usize,
        const NUM_DECODER_LAYERS: usize,
        const FF_DIM: usize,
    >;

    pub use super::decoder::builder::{TransformerDecoder, TransformerDecoderBlock};
    pub use super::encoder::builder::{TransformerEncoder, TransformerEncoderBlock};
    pub use super::mha::builder::MultiHeadAttention;
}

impl<const M: usize, const H: usize, const A: usize, const B: usize, const F: usize, E, D>
    BuildOnDevice<D, E> for builder::Transformer<M, H, A, B, F>
where
    E: Dtype,
    D: Device<E>,
    Transformer<M, H, A, B, F, E, D>: BuildModule<D, E>,
{
    type Built = Transformer<M, H, A, B, F, E, D>;
    fn try_build_on_device(device: &D) -> Result<Self::Built, <D>::Err> {
        Self::Built::try_build(device)
    }
}

/// Transformer architecture as described in
/// [Attention is all you need](https://arxiv.org/abs/1706.03762).
///
/// This is comprised of a [TransformerEncoder] and a [TransformerDecoder].
///
/// Generics:
/// - `MODEL_DIM`: Size of the input features to the encoder/decoder.
/// - `NUM_HEADS`: Number of heads for [MultiHeadAttention].
/// - `NUM_ENCODER_LAYERS`: Number of [TransformerEncoderBlock] to use
/// - `NUM_DECODER_LAYERS`: Number of [TransformerDecoderBlock] to use
/// - `FF_DIM`: Feedforward hidden dimension for both encoder/decoder
///
/// **Pytorch equivalent**:
/// ```python
/// torch.nn.Transformer(
///     d_model=MODEL_DIM,
///     nhead=NUM_HEADS,
///     num_encoder_layers=NUM_ENCODER_LAYERS,
///     num_decoder_layers=NUM_DECODER_LAYERS,
///     dim_feedforward=FF_DIM,
///     batch_first=True,
/// )
/// ```
#[derive(Debug, Clone)]
pub struct Transformer<
    const MODEL_DIM: usize,
    const NUM_HEADS: usize,
    const NUM_ENCODER_LAYERS: usize,
    const NUM_DECODER_LAYERS: usize,
    const FF_DIM: usize,
    E: Dtype,
    D: DeviceStorage,
> {
    pub encoder: TransformerEncoder<MODEL_DIM, NUM_HEADS, FF_DIM, NUM_ENCODER_LAYERS, E, D>,
    pub decoder: TransformerDecoder<MODEL_DIM, NUM_HEADS, FF_DIM, NUM_DECODER_LAYERS, E, D>,
}

impl<const M: usize, const H: usize, const A: usize, const B: usize, const F: usize, E, D>
    TensorCollection<E, D> for Transformer<M, H, A, B, F, E, D>
where
    E: Dtype + Float + SampleUniform,
    D: Device<E>,
{
    type To<E2: Dtype, D2: Device<E2>> = Transformer<M, H, A, B, F, E2, D2>;

    fn iter_tensors<V: ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            (
                Self::module("encoder", |s| &s.encoder, |s| &mut s.encoder),
                Self::module("decoder", |s| &s.decoder, |s| &mut s.decoder),
            ),
            |(encoder, decoder)| Transformer { encoder, decoder },
        )
    }
}

impl<
        const M: usize,
        const H: usize,
        const EL: usize,
        const DL: usize,
        const F: usize,
        E: Dtype,
        D: Device<E>,
        Src: SplitTape,
        Tgt: PutTape<Src::Tape>,
    > Module<(Src, Tgt)> for Transformer<M, H, EL, DL, F, E, D>
where
    TransformerEncoder<M, H, F, EL, E, D>: Module<Src, Output = Src, Error = D::Err>,
    TransformerDecoder<M, H, F, DL, E, D>: Module<
        (<Tgt as PutTape<Src::Tape>>::Output, Src::NoTape),
        Output = <Tgt as PutTape<Src::Tape>>::Output,
        Error = D::Err,
    >,
{
    type Output = <Tgt as PutTape<Src::Tape>>::Output;
    type Error = D::Err;

    fn try_forward(&self, (src, tgt): (Src, Tgt)) -> Result<Self::Output, D::Err> {
        let (mem, tape) = self.encoder.try_forward(src)?.split_tape();
        self.decoder.try_forward((tgt.put_tape(tape), mem))
    }
}

impl<const M: usize, const H: usize, const A: usize, const B: usize, const F: usize, E, D>
    NonMutableModule for Transformer<M, H, A, B, F, E, D>
where
    E: Dtype,
    D: Device<E>,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{optim::*, tests::*};

    #[test]
    fn test_forward() {
        let dev = TestDevice::seed_from_u64(0);
        type Model = builder::Transformer<16, 4, 3, 3, 8>;
        let mut t = dev.build_module::<Model, TestDtype>();

        // unbatched
        let src = dev.sample_normal::<Rank2<7, 16>>();
        let tgt = dev.sample_normal::<Rank2<9, 16>>();
        let _: Tensor<Rank2<9, 16>, _, _, _> = t.forward_mut((src, tgt));

        // batched
        let src = dev.sample_normal::<Rank3<4, 12, 16>>();
        let tgt = dev.sample_normal::<Rank3<4, 6, 16>>();
        let _: Tensor<Rank3<4, 6, 16>, _, _, _> = t.forward_mut((src, tgt));
    }

    #[test]
    fn test_backward() {
        let dev = TestDevice::seed_from_u64(0);
        type Model = builder::Transformer<16, 4, 3, 3, 8>;
        let mut t = dev.build_module::<Model, TestDtype>();

        let src = dev.sample_normal::<Rank3<4, 12, 16>>();
        let tgt = dev.sample_normal::<Rank3<4, 6, 16>>();
        let out: Tensor<Rank3<4, 6, 16>, _, _, _> = t.forward_mut((src.leaky_trace(), tgt));
        let g = out.mean().backward();

        let mut opt = Sgd::new(&t, Default::default());
        opt.update(&mut t, &g).expect("");
    }
}
