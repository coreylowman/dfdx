mod decoder;
mod encoder;
mod mha;

pub use decoder::*;
pub use encoder::*;
pub use mha::*;

use crate::{
    optim::{GradientUpdate, ParamUpdater, UnusedTensors},
    shapes::Dtype,
    tensor::{DeviceStorage, PutTape, SplitTape},
    tensor_ops::Device,
};

use super::{BuildModule, Module, ModuleMut, ResetParams, ToDevice};

/// **Requires Nightly** Transformer architecture as described in
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
>;

impl<const M: usize, const H: usize, const A: usize, const B: usize, const F: usize, D>
    BuildModule<D, f32> for Transformer<M, H, A, B, F>
where
    D: Device<f32>,
{
    type Built = DeviceTransformer<M, H, A, B, F, f32, D>;
    fn try_build(device: &D) -> Result<Self::Built, <D>::Err> {
        Ok(Self::Built {
            encoder: TransformerEncoder::try_build(device)?,
            decoder: TransformerDecoder::try_build(device)?,
        })
    }
}

#[derive(Debug, Clone)]
pub struct DeviceTransformer<
    const M: usize,
    const H: usize,
    const EL: usize,
    const DL: usize,
    const F: usize,
    E: Dtype,
    D: DeviceStorage,
> {
    pub encoder: DeviceEncoder<M, H, F, EL, E, D>,
    pub decoder: DeviceDecoder<M, H, F, DL, E, D>,
}

impl<const M: usize, const H: usize, const A: usize, const B: usize, const F: usize, D>
    ResetParams<D, f32> for DeviceTransformer<M, H, A, B, F, f32, D>
where
    D: Device<f32>,
{
    fn try_reset_params(&mut self) -> Result<(), <D>::Err> {
        self.encoder.try_reset_params()?;
        self.decoder.try_reset_params()?;
        Ok(())
    }
}

impl<const M: usize, const H: usize, const A: usize, const B: usize, const F: usize, D>
    GradientUpdate<D, f32> for DeviceTransformer<M, H, A, B, F, f32, D>
where
    D: Device<f32>,
{
    fn update<U>(&mut self, updater: &mut U, unused: &mut UnusedTensors) -> Result<(), <D>::Err>
    where
        U: ParamUpdater<D, f32>,
    {
        self.encoder.update(updater, unused)?;
        self.decoder.update(updater, unused)?;
        Ok(())
    }
}

impl<const M: usize, const H: usize, const A: usize, const B: usize, const F: usize, D1, D2>
    ToDevice<D2> for DeviceTransformer<M, H, A, B, F, f32, D1>
where
    D1: Device<f32>,
    D2: Device<f32>,
{
    type Output = DeviceTransformer<M, H, A, B, F, f32, D2>;

    fn to_device(&self, device: &D2) -> Self::Output {
        DeviceTransformer {
            encoder: self.encoder.to_device(device),
            decoder: self.decoder.to_device(device),
        }
    }
}

impl<
        const M: usize,
        const H: usize,
        const EL: usize,
        const DL: usize,
        const F: usize,
        D: Device<f32>,
        Src: SplitTape,
        Tgt: PutTape<Src::Tape>,
    > Module<(Src, Tgt)> for DeviceTransformer<M, H, EL, DL, F, f32, D>
where
    DeviceEncoder<M, H, F, EL, f32, D>: Module<Src, Output = Src>,
    DeviceDecoder<M, H, F, DL, f32, D>: Module<
        (<Tgt as PutTape<Src::Tape>>::Output, Src::NoTape),
        Output = <Tgt as PutTape<Src::Tape>>::Output,
    >,
{
    type Output = <Tgt as PutTape<Src::Tape>>::Output;

    fn forward(&self, (src, tgt): (Src, Tgt)) -> Self::Output {
        let (mem, tape) = self.encoder.forward(src).split_tape();
        self.decoder.forward((tgt.put_tape(tape), mem))
    }
}

impl<const M: usize, const H: usize, const A: usize, const B: usize, const F: usize, D, T>
    ModuleMut<T> for DeviceTransformer<M, H, A, B, F, f32, D>
where
    D: Device<f32>,
    Self: Module<T>,
{
    type Output = <Self as Module<T>>::Output;
    fn forward_mut(&mut self, t: T) -> Self::Output {
        self.forward(t)
    }
}

#[cfg(feature = "nightly")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{nn::tests::SimpleUpdater, shapes::*, tensor::*, tensor_ops::*, tests::TestDevice};

    #[test]
    fn test_forward() {
        let dev = TestDevice::seed_from_u64(0);
        type Model = Transformer<16, 4, 3, 3, 8>;
        let mut t = Model::build(&dev);

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
        type Model = Transformer<16, 4, 3, 3, 8>;
        let mut t = Model::build(&dev);

        let src = dev.sample_normal::<Rank3<4, 12, 16>>();
        let tgt = dev.sample_normal::<Rank3<4, 6, 16>>();
        let out: Tensor<Rank3<4, 6, 16>, _, _, _> = t.forward_mut((src.trace(), tgt));
        let g = out.mean().backward();

        let mut gs = SimpleUpdater(g);
        let mut unused: UnusedTensors = Default::default();
        t.update(&mut gs, &mut unused).unwrap();
        assert!(unused.is_empty());
    }
}
