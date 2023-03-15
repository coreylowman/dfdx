use num_traits::Float;
use rand_distr::uniform::SampleUniform;

use crate::{
    nn::modules::*,
    shapes::Dtype,
    tensor::{DeviceStorage, PutTape, SplitTape},
    tensor_ops::Device,
};

use super::mha::MultiHeadAttention;

/// A transformer encoder.
///
/// Generics
/// - `MODEL_DIM`: The size of query/key/value tensors. Given to [MultiHeadAttention].
/// - `NUM_HEADS`: The number of heads in [MultiHeadAttention].
/// - `FF_DIM`: The size of the hidden layer in
///   the feedforward network in [TransformerEncoderBlock].
/// - `NUM_LAYERS`: The number of [TransformerEncoderBlock] to use.
pub type TransformerEncoder<
    const MODEL_DIM: usize,
    const NUM_HEADS: usize,
    const FF_DIM: usize,
    const NUM_LAYERS: usize,
    E,
    D,
> = Repeated<TransformerEncoderBlock<MODEL_DIM, NUM_HEADS, FF_DIM, E, D>, NUM_LAYERS>;

pub mod builder {
    #[derive(Debug)]
    pub struct TransformerEncoder<
        const MODEL_DIM: usize,
        const NUM_HEADS: usize,
        const FF_DIM: usize,
        const NUM_LAYERS: usize,
    >;

    #[derive(Debug)]
    pub struct TransformerEncoderBlock<
        const MODEL_DIM: usize,
        const NUM_HEADS: usize,
        const FF_DIM: usize,
    >;
}

impl<const M: usize, const H: usize, const F: usize, const L: usize, E: Dtype, D: Device<E>>
    BuildOnDevice<D, E> for builder::TransformerEncoder<M, H, F, L>
where
    TransformerEncoder<M, H, F, L, E, D>: BuildModule<D, E>,
{
    type Built = TransformerEncoder<M, H, F, L, E, D>;
    fn try_build_on_device(device: &D) -> Result<Self::Built, <D>::Err> {
        Self::Built::try_build(device)
    }
}

impl<const M: usize, const H: usize, const F: usize, E: Dtype, D: Device<E>> BuildOnDevice<D, E>
    for builder::TransformerEncoderBlock<M, H, F>
where
    TransformerEncoderBlock<M, H, F, E, D>: BuildModule<D, E>,
{
    type Built = TransformerEncoderBlock<M, H, F, E, D>;
    fn try_build_on_device(device: &D) -> Result<Self::Built, <D>::Err> {
        Self::Built::try_build(device)
    }
}

/// A single transformer encoder block
///
/// Generics
/// - `MODEL_DIM`: The size of query/key/value tensors. Given to [MultiHeadAttention].
/// - `NUM_HEADS`: The number of heads in [MultiHeadAttention].
/// - `FF_DIM`: The size of the hidden layer in the feedforward network.
///
/// **Pytorch equivalent**:
/// ```python
/// encoder = torch.nn.TransformerEncoderLayer(
///    EMBED_DIM, NUM_HEADS, dim_feedforward=FF_DIM, batch_first=True, dropout=0.0
/// )
/// ```
#[derive(Clone, Debug)]
pub struct TransformerEncoderBlock<
    const MODEL_DIM: usize,
    const NUM_HEADS: usize,
    const FF_DIM: usize,
    E: Dtype,
    D: DeviceStorage,
> {
    pub self_attn: MultiHeadAttention<MODEL_DIM, NUM_HEADS, MODEL_DIM, MODEL_DIM, E, D>,
    pub norm1: LayerNorm1D<MODEL_DIM, E, D>,
    pub ff: FF<MODEL_DIM, FF_DIM, E, D>,
    pub norm2: LayerNorm1D<MODEL_DIM, E, D>,
}

type FF<const M: usize, const F: usize, E, D> =
    Residual<(Linear<M, F, E, D>, ReLU, Linear<F, M, E, D>)>;

impl<const M: usize, const H: usize, const F: usize, E, D: Device<E>> TensorCollection<E, D>
    for TransformerEncoderBlock<M, H, F, E, D>
where
    E: Dtype + Float + SampleUniform,
{
    type To<E2: Dtype, D2: Device<E2>> = TransformerEncoderBlock<M, H, F, E2, D2>;

    fn iter_tensors<V: ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            (
                Self::module("self_attn", |s| &s.self_attn, |s| &mut s.self_attn),
                Self::module("norm1", |s| &s.norm1, |s| &mut s.norm1),
                Self::module("ff", |s| &s.ff, |s| &mut s.ff),
                Self::module("norm2", |s| &s.norm2, |s| &mut s.norm2),
            ),
            |(self_attn, norm1, ff, norm2)| TransformerEncoderBlock {
                self_attn,
                norm1,
                ff,
                norm2,
            },
        )
    }
}

impl<const M: usize, const H: usize, const F: usize, E: Dtype, D: Device<E>, Src> Module<Src>
    for TransformerEncoderBlock<M, H, F, E, D>
where
    Src: SplitTape + std::ops::Add<Src::NoTape, Output = Src>,
    MultiHeadAttention<M, H, M, M, E, D>: Module<Src, Output = Src, Error = D::Err>,
    LayerNorm1D<M, E, D>: Module<Src, Output = Src, Error = D::Err>,
    FF<M, F, E, D>: Module<Src, Output = Src, Error = D::Err>,
{
    type Output = Src;
    type Error = D::Err;

    fn try_forward(&self, src: Src) -> Result<Self::Output, D::Err> {
        let (src, tape) = src.split_tape();
        let x = self.self_attn.try_forward(src.clone().put_tape(tape))?;
        let x = x + src;
        let x = self.norm1.try_forward(x)?;
        let x = self.ff.try_forward(x)?;
        self.norm2.try_forward(x)
    }
}

impl<const M: usize, const H: usize, const F: usize, E: Dtype, D: Device<E>> NonMutableModule
    for TransformerEncoderBlock<M, H, F, E, D>
{
}

#[cfg(test)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::{shapes::Rank3, tensor::*, tests::*};

    #[test]
    fn test_encoder_block_forward() {
        let dev = TestDevice::seed_from_u64(2);

        const BATCH: usize = 3;
        const SEQ_LEN: usize = 5;
        const EMBED_DIM: usize = 9;
        const NUM_HEADS: usize = 3;
        const FF_DIM: usize = 16;

        type Dtype = f32;

        let encoder = dev
            .build_module::<builder::TransformerEncoderBlock<EMBED_DIM, NUM_HEADS, FF_DIM>, Dtype>(
            );

        let x: Tensor<Rank3<BATCH, SEQ_LEN, EMBED_DIM>, Dtype, _> = dev.sample_normal();
        let y = encoder.forward(x);

        // This expected y was generated by:
        // 1. saving `encoder` parameters, `x` and `y` to a npz files
        // 2. Running pytorch with the same values
        // 3. printing out the output
        // See https://github.com/coreylowman/dfdx/wiki/Exporting-MultiHeadAttention-to-pytorch-for-unit-tests
        #[rustfmt::skip]
        assert_close(
            &y.array(),
            &[
                [
                    [0.83316803, 0.85057360, 0.37431455, 1.48506296,-0.38405111,-1.89352179,-1.07049453,-0.50913972, 0.31408834],
                    [-0.57205188, 0.64078861,-0.56589824, 0.67155081, 0.65419787, 0.28409126,-1.75282931, 1.68111539,-1.04096484],
                    [-0.01414229, 1.34985816, 0.09684382, 0.13165890,-1.39875984,-1.61741352, 1.28747427, 0.75574619,-0.59126562],
                    [0.12542287, 2.60457349, 0.21064451,-0.81285846,-0.15861531,-0.87273139,-0.81707120,-0.17004849,-0.10931605],
                    [-1.54970682,-0.77183282, 1.37495196,-0.69562960,-0.66684282, 0.24720824, 1.38581741,-0.35962212, 1.03565681],
                ],
                [
                    [-0.15229249,-0.90768278,-0.85165489, 0.12768827, 1.61459768, 1.25826979,-0.46860829, 0.87496787,-1.49528503],
                    [-1.35595357, 1.13305736,-0.08542954, 1.01601434,-0.04678532,-1.69470263, 0.76144469,-0.68443829, 0.95679283],
                    [-1.49877191, 0.64559501, 0.33383703, 1.73698330,-0.14289393, 1.17869902,-1.01659226,-0.61038357,-0.62647283],
                    [0.78263682, 0.78481543,-0.16064386, 1.03396618, 1.49144781,-1.55002558,-1.11833119,-0.62120575,-0.64265978],
                    [-1.58957553, 1.75000548, 0.01272983, 0.11212827,-0.34744453,-1.45086825, 0.95842224, 0.50071126, 0.05389150],
                ],
                [
                    [-1.13160479,-0.21202824, 0.25907388,-0.64313424,-0.76302397,-0.16797650,-0.75345570, 2.01765633, 1.39449334],
                    [-0.16463053,-0.73241645,-0.69120175, 0.13771832, 0.72443259,-2.06525135, 1.02475107, 1.40244913, 0.36414924],
                    [0.38766465,-0.19543301,-1.80767059, 1.11545098, 0.21692322,-1.22834778, 0.13580292, 1.63094711,-0.25533777],
                    [1.22877085, 0.05472810, 0.65142977, 0.73869365,-0.74706972,-1.29277837, 1.07350135, 0.06228387,-1.76955938],
                    [-0.01733636,-1.57447529, 0.79691470, 1.00687420, 1.65637493,-0.75668150,-0.54616517, 0.45799020,-1.02349579],
                ],
            ],
        );
    }
}
