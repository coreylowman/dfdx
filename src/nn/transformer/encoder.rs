use crate::{
    nn::*,
    optim::{GradientUpdate, ParamUpdater, UnusedTensors},
    shapes::Dtype,
    tensor::{PutTape, SplitTape},
    tensor_ops::Device,
};

use super::{mha::DeviceMHA, MultiHeadAttention};

/// **Requires Nightly** A transformer encoder.
///
/// Generics
/// - `MODEL_DIM`: The size of query/key/value tensors. Given to [MultiHeadAttention].
/// - `NUM_HEADS`: The number of heads in [MultiHeadAttention].
/// - `FF_DIM`: The size of the hidden layer in
///   the feedforward network in [TransformerEncoderBlock].
/// - `NUM_LAYERS`: The number of [TransformerEncoderBlock] to use.
/// TODO: Doctests
pub type DeviceEncoder<const M: usize, const H: usize, const F: usize, const L: usize, E, D> =
    Repeated<DeviceEncoderBlock<M, H, F, E, D>, L>;

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
pub struct TransformerEncoderBlock<
    const MODEL_DIM: usize,
    const NUM_HEADS: usize,
    const FF_DIM: usize,
>;

impl<const M: usize, const H: usize, const F: usize, D: Device<f32>> BuildModule<D, f32>
    for TransformerEncoderBlock<M, H, F>
{
    type Built = DeviceEncoderBlock<M, H, F, f32, D>;
    fn try_build(device: &D) -> Result<Self::Built, <D>::Err> {
        Ok(Self::Built {
            self_attn: MultiHeadAttention::try_build(device)?,
            norm1: LayerNorm1D::try_build(device)?,
            ff: FF::try_build(device)?,
            norm2: LayerNorm1D::try_build(device)?,
        })
    }
}

type FF<const M: usize, const F: usize> = Residual<(Linear<M, F>, ReLU, Linear<F, M>)>;

type DeviceFF<const M: usize, const F: usize, E, D> =
    Residual<(DeviceLinear<M, F, E, D>, ReLU, DeviceLinear<F, M, E, D>)>;

#[derive(Clone, Debug)]
pub struct DeviceEncoderBlock<
    const M: usize,
    const H: usize,
    const F: usize,
    E: Dtype,
    D: DeviceStorage,
> {
    pub self_attn: DeviceMHA<M, H, M, M, E, D>,
    pub norm1: DeviceLayerNorm1D<M, E, D>,
    pub ff: DeviceFF<M, F, E, D>,
    pub norm2: DeviceLayerNorm1D<M, E, D>,
}

impl<const M: usize, const H: usize, const F: usize, D: Device<f32>> ResetParams<D, f32>
    for DeviceEncoderBlock<M, H, F, f32, D>
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
    for DeviceEncoderBlock<M, H, F, f32, D>
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
    for DeviceEncoderBlock<M, H, F, f32, D1>
{
    type Output = DeviceEncoderBlock<M, H, F, f32, D2>;
    fn to_device(&self, device: &D2) -> Self::Output {
        DeviceEncoderBlock {
            self_attn: self.self_attn.to_device(device),
            norm1: self.norm1.to_device(device),
            ff: self.ff.to_device(device),
            norm2: self.norm2.to_device(device),
        }
    }
}

impl<const M: usize, const H: usize, const F: usize, D: Device<f32>, Src> Module<Src>
    for DeviceEncoderBlock<M, H, F, f32, D>
where
    Src: SplitTape + std::ops::Add<Src::NoTape, Output = Src>,
    DeviceMHA<M, H, M, M, f32, D>: Module<Src, Output = Src>,
    DeviceLayerNorm1D<M, f32, D>: Module<Src, Output = Src>,
    DeviceFF<M, F, f32, D>: Module<Src, Output = Src>,
{
    type Output = Src;

    fn forward(&self, src: Src) -> Self::Output {
        let (src, tape) = src.split_tape();
        let x = self.self_attn.forward(src.clone().put_tape(tape));
        let x = x + src;
        let x = self.norm1.forward(x);
        let x = self.ff.forward(x);
        self.norm2.forward(x)
    }
}

impl<const M: usize, const H: usize, const F: usize, D: Device<f32>, T> ModuleMut<T>
    for DeviceEncoderBlock<M, H, F, f32, D>
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
            TransformerEncoderBlock::<EMBED_DIM, NUM_HEADS, FF_DIM>::build_on_device(&dev);

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
