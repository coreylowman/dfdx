use crate::{
    nn::{modules::*, *},
    optim::{GradientUpdate, ParamUpdater, UnusedTensors},
    shapes::Dtype,
    tensor::{PutTape, SplitTape},
    tensor_ops::Device,
};

use super::mha::MultiHeadAttention;

/// **Requires Nightly** A transformer encoder.
///
/// Generics
/// - `MODEL_DIM`: The size of query/key/value tensors. Given to [MultiHeadAttention].
/// - `NUM_HEADS`: The number of heads in [MultiHeadAttention].
/// - `FF_DIM`: The size of the hidden layer in
///   the feedforward network in [TransformerEncoderBlock].
/// - `NUM_LAYERS`: The number of [TransformerEncoderBlock] to use.
/// TODO: Doctests
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

impl<const M: usize, const H: usize, const F: usize, const L: usize, D: Device<f32>>
    BuildOnDevice<D, f32> for builder::TransformerEncoder<M, H, F, L>
{
    type Built = TransformerEncoder<M, H, F, L, f32, D>;
    fn try_build_on_device(device: &D) -> Result<Self::Built, <D>::Err> {
        Self::Built::try_build(device)
    }
}

impl<const M: usize, const H: usize, const F: usize, D: Device<f32>> BuildOnDevice<D, f32>
    for builder::TransformerEncoderBlock<M, H, F>
{
    type Built = TransformerEncoderBlock<M, H, F, f32, D>;
    fn try_build_on_device(device: &D) -> Result<Self::Built, <D>::Err> {
        Self::Built::try_build(device)
    }
}

/// **Requires Nightly** A single transformer encoder block
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
/// TODO: Doctests
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

impl<const M: usize, const H: usize, const F: usize, D: Device<f32>> BuildModule<D, f32>
    for TransformerEncoderBlock<M, H, F, f32, D>
{
    fn try_build(device: &D) -> Result<Self, <D>::Err> {
        Ok(Self {
            self_attn: BuildModule::try_build(device)?,
            norm1: BuildModule::try_build(device)?,
            ff: BuildModule::try_build(device)?,
            norm2: BuildModule::try_build(device)?,
        })
    }
}

impl<const M: usize, const H: usize, const F: usize, D: Device<f32>> ResetParams<D, f32>
    for TransformerEncoderBlock<M, H, F, f32, D>
{
    fn try_reset_params(&mut self) -> Result<(), <D>::Err> {
        self.self_attn.try_reset_params()?;
        self.norm1.try_reset_params()?;
        self.ff.try_reset_params()?;
        self.norm2.try_reset_params()?;
        Ok(())
    }
}

impl<const M: usize, const H: usize, const F: usize, D: Device<f32>> GradientUpdate<D, f32>
    for TransformerEncoderBlock<M, H, F, f32, D>
{
    fn update<U>(&mut self, updater: &mut U, unused: &mut UnusedTensors) -> Result<(), <D>::Err>
    where
        U: ParamUpdater<D, f32>,
    {
        self.self_attn.update(updater, unused)?;
        self.norm1.update(updater, unused)?;
        self.ff.update(updater, unused)?;
        self.norm2.update(updater, unused)?;
        Ok(())
    }
}

impl<const M: usize, const H: usize, const F: usize, D1: Device<f32>, D2: Device<f32>> ToDevice<D2>
    for TransformerEncoderBlock<M, H, F, f32, D1>
{
    type Output = TransformerEncoderBlock<M, H, F, f32, D2>;
    fn to_device(&self, device: &D2) -> Self::Output {
        TransformerEncoderBlock {
            self_attn: self.self_attn.to_device(device),
            norm1: self.norm1.to_device(device),
            ff: self.ff.to_device(device),
            norm2: self.norm2.to_device(device),
        }
    }
}

impl<const M: usize, const H: usize, const F: usize, D: Device<f32>, Src> Module<Src>
    for TransformerEncoderBlock<M, H, F, f32, D>
where
    Src: SplitTape + std::ops::Add<Src::NoTape, Output = Src>,
    MultiHeadAttention<M, H, M, M, f32, D>: Module<Src, Output = Src>,
    LayerNorm1D<M, f32, D>: Module<Src, Output = Src>,
    FF<M, F, f32, D>: Module<Src, Output = Src>,
{
    type Output = Src;

    fn forward(&self, src: Src) -> Self::Output {
        let (src, tape) = src.split_tape();
        let x = self.self_attn.forward(src.clone().put_tape(tape));
        let x = x + src;
        let x = self.norm1.forward(x);
        let (x, tape) = x.split_tape();
        let x_residual = x.clone();
        let x = self.ff.forward(x.put_tape(tape));
        self.norm2.forward(x + x_residual)
    }
}

impl<const M: usize, const H: usize, const F: usize, D: Device<f32>, T> ModuleMut<T>
    for TransformerEncoderBlock<M, H, F, f32, D>
where
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
    use crate::{
        shapes::Rank3,
        tensor::{AsArray, SampleTensor},
        tests::*,
    };

    #[test]
    fn test_encoder_block_forward() {
        let dev = TestDevice::seed_from_u64(2);

        const BATCH: usize = 3;
        const SEQ_LEN: usize = 5;
        const EMBED_DIM: usize = 9;
        const NUM_HEADS: usize = 3;
        const FF_DIM: usize = 16;

        let encoder =
            builder::TransformerEncoderBlock::<EMBED_DIM, NUM_HEADS, FF_DIM>::build_on_device(&dev);

        let x = dev.sample_normal::<Rank3<BATCH, SEQ_LEN, EMBED_DIM>>();
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
                    [1.1020464, 0.70063215, 0.14642186, 1.6225557, -0.4766779, -1.6740295, -1.0135144, -0.70404243, 0.2966079], 
                    [-0.6346884, 0.55069435, -0.697153, 0.5799327, 0.5051933, 0.5023419, -1.6387398, 1.8183587, -0.98594], 
                    [0.013820761, 1.2537417, 0.10195519, 0.12989187, -1.5102644, -1.4867771, 1.4450823, 0.66643316, -0.6138834], 
                    [0.42696413, 2.6044195, 0.1249334, -0.8968443, -0.34174177, -0.6291483, -0.7388344, -0.32556844, -0.22417991], 
                    [-1.4780447, -0.9885726, 1.4425869, -0.6340557, -0.65442955, 0.39416587, 1.2612454, -0.36597, 1.0230747]
                ], 
                [
                    [-0.3735986, -0.93679315, -0.7623837, 0.061064582, 1.5543288, 1.3882414, -0.39559835, 0.9060303, -1.4412912], 
                    [-1.2308724, 1.1551111, -0.32503325, 1.042616, -0.21810767, -1.4823375, 0.92051923, -0.89431936, 1.0324237], 
                    [-1.3670719, 0.50169045, 0.11029812, 1.7554268, -0.20988941, 1.4115741, -0.9992623, -0.7181457, -0.48462012], 
                    [0.9449202, 0.6039523, -0.3057507, 1.0897537, 1.5412592, -1.3200078, -1.1866173, -0.69592535, -0.6715845], 
                    [-1.608161, 1.6817892, -0.017305028, 0.10595742, -0.45335528, -1.3680971, 1.1419259, 0.43018824, 0.08705761]
                ], 
                [
                    [-1.1326666, -0.44324997, 0.1956163, -0.6573088, -0.7423244, 0.057818394, -0.67033637, 2.0784216, 1.3140298], 
                    [-0.24817339, -0.9648294, -0.638977, 0.07365655, 0.79484904, -1.8809983, 0.9881894, 1.5242727, 0.35201037], 
                    [0.47322243, -0.33990225, -1.9786029, 0.96823835, 0.15409008, -0.95221484, 0.18269716, 1.6764445, -0.1839726], 
                    [1.2580589, -0.21497832, 0.6631624, 0.8229629, -0.78036994, -1.0463971, 1.0627513, 0.08246568, -1.8476558], 
                    [-0.19245562, -1.675238, 0.8498067, 0.99485075, 1.6175559, -0.5155067, -0.6052279, 0.4992601, -0.9730453]
                ]
            ]
        );
    }
}
