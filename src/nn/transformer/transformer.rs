use crate::gradients::{CanUpdateWithGradients, GradientProvider, UnusedTensors};
use crate::prelude::*;
use std::io::{Read, Seek, Write};
use zip::{result::ZipResult, ZipArchive, ZipWriter};

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
#[derive(Debug, Default, Clone)]
pub struct Transformer<
    const MODEL_DIM: usize,
    const NUM_HEADS: usize,
    const NUM_ENCODER_LAYERS: usize,
    const NUM_DECODER_LAYERS: usize,
    const FF_DIM: usize,
> {
    encoder: TransformerEncoder<MODEL_DIM, NUM_HEADS, FF_DIM, NUM_ENCODER_LAYERS>,
    decoder: TransformerDecoder<MODEL_DIM, NUM_HEADS, FF_DIM, NUM_DECODER_LAYERS>,
}

impl<const M: usize, const H: usize, const E: usize, const D: usize, const F: usize> ResetParams
    for Transformer<M, H, E, D, F>
{
    fn reset_params<R: rand::Rng>(&mut self, rng: &mut R) {
        self.encoder.reset_params(rng);
        self.decoder.reset_params(rng);
    }
}

impl<const M: usize, const H: usize, const E: usize, const D: usize, const F: usize>
    CanUpdateWithGradients for Transformer<M, H, E, D, F>
{
    fn update<G: GradientProvider>(&mut self, grads: &mut G, unused: &mut UnusedTensors) {
        self.encoder.update(grads, unused);
        self.decoder.update(grads, unused);
    }
}

impl<const M: usize, const H: usize, const E: usize, const D: usize, const F: usize, Src, Tgt>
    Module<(Src, Tgt)> for Transformer<M, H, E, D, F>
where
    Src: Tensor<Dtype = f32>,
    Tgt: Tensor<Dtype = f32> + PutTape<Src::Tape>,
    TransformerEncoder<M, H, F, E>: Module<Src, Output = Src>,
    TransformerDecoder<M, H, F, D>: Module<
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

impl<const M: usize, const H: usize, const E: usize, const D: usize, const F: usize> SaveToNpz
    for Transformer<M, H, E, D, F>
{
    fn write<W: Write + Seek>(&self, pre: &str, w: &mut ZipWriter<W>) -> ZipResult<()> {
        self.encoder.write(&format!("{pre}encoder."), w)?;
        self.decoder.write(&format!("{pre}decoder."), w)?;
        Ok(())
    }
}

impl<const M: usize, const H: usize, const E: usize, const D: usize, const F: usize> LoadFromNpz
    for Transformer<M, H, E, D, F>
{
    fn read<R: Read + Seek>(&mut self, pre: &str, r: &mut ZipArchive<R>) -> Result<(), NpzError> {
        self.encoder.read(&format!("{pre}encoder."), r)?;
        self.decoder.read(&format!("{pre}decoder."), r)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::tests::SimpleGradients;
    use rand::{rngs::StdRng, thread_rng, SeedableRng};
    use tempfile::NamedTempFile;

    #[test]
    fn test_forward() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut t: Transformer<16, 4, 3, 3, 8> = Default::default();
        t.reset_params(&mut rng);

        // unbatched
        let src: Tensor2D<7, 16> = TensorCreator::randn(&mut rng);
        let tgt: Tensor2D<9, 16> = TensorCreator::randn(&mut rng);
        let _: Tensor2D<9, 16> = t.forward((src, tgt));

        // batched
        let src: Tensor3D<4, 12, 16> = TensorCreator::randn(&mut rng);
        let tgt: Tensor3D<4, 6, 16> = TensorCreator::randn(&mut rng);
        let _: Tensor3D<4, 6, 16> = t.forward((src, tgt));
    }

    #[test]
    fn test_backward() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut t: Transformer<16, 4, 3, 3, 8> = Default::default();
        t.reset_params(&mut rng);

        let src: Tensor3D<4, 12, 16> = TensorCreator::randn(&mut rng);
        let tgt: Tensor3D<4, 6, 16> = TensorCreator::randn(&mut rng);
        let out: Tensor3D<4, 6, 16, _> = t.forward((src.trace(), tgt));
        let g = backward(out.mean());

        let mut gs = SimpleGradients(g);
        let mut unused: UnusedTensors = Default::default();
        t.update(&mut gs, &mut unused);

        assert!(unused.is_empty());
    }

    #[test]
    fn test_save_load() {
        let mut rng = thread_rng();

        let mut saved: Transformer<16, 4, 3, 4, 8> = Default::default();
        saved.reset_params(&mut rng);

        let file = NamedTempFile::new().expect("failed to create tempfile");
        saved.save(file.path()).expect("");

        let mut loaded: Transformer<16, 4, 3, 4, 8> = Default::default();
        loaded.load(file.path()).expect("");

        let src: Tensor3D<4, 12, 16> = TensorCreator::randn(&mut rng);
        let tgt: Tensor3D<4, 6, 16> = TensorCreator::randn(&mut rng);

        let y1 = saved.forward((src.clone(), tgt.clone()));
        let y2 = loaded.forward((src.clone(), tgt.clone()));

        assert_eq!(y1.data(), y2.data());
    }
}
