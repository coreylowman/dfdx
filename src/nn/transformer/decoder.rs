use crate::{
    nn::{modules::*, *},
    optim::{GradientUpdate, ParamUpdater, UnusedTensors},
    shapes::Dtype,
    tensor::{PutTape, SplitTape},
    tensor_ops::Device,
};

use super::mha::MultiHeadAttention;

pub mod builder {
    #[derive(Clone, Debug)]
    pub struct TransformerDecoder<
        const MODEL_DIM: usize,
        const NUM_HEADS: usize,
        const FF_DIM: usize,
        const NUM_LAYERS: usize,
    >;

    #[derive(Clone, Debug)]
    pub struct TransformerDecoderBlock<
        const MODEL_DIM: usize,
        const NUM_HEADS: usize,
        const FF_DIM: usize,
    >;
}

impl<const M: usize, const H: usize, const F: usize, const L: usize, D: Device<f32>>
    BuildOnDevice<D, f32> for builder::TransformerDecoder<M, H, F, L>
{
    type Built = TransformerDecoder<M, H, F, L, f32, D>;
    fn try_build_on_device(device: &D) -> Result<Self::Built, D::Err> {
        Self::Built::try_build(device)
    }
}

impl<const M: usize, const N: usize, const F: usize, D: Device<f32>> BuildOnDevice<D, f32>
    for builder::TransformerDecoderBlock<M, N, F>
{
    type Built = TransformerDecoderBlock<M, N, F, f32, D>;
    fn try_build_on_device(device: &D) -> Result<Self::Built, <D>::Err> {
        Self::Built::try_build(device)
    }
}

/// **Requires Nightly** A transformer decoder.
///
/// Generics
/// - `MODEL_DIM`: The size of query/key/value tensors. Given to [MultiHeadAttention].
/// - `NUM_HEADS`: The number of heads in [MultiHeadAttention].
/// - `FF_DIM`: The size of the hidden layer in
///   the feedforward network in [TransformerDecoderBlock].
/// - `NUM_LAYERS`: The number of [TransformerDecoderBlock] to use.
/// TODO: Doctests
#[derive(Clone, Debug)]
pub struct TransformerDecoder<
    const MODEL_DIM: usize,
    const NUM_HEADS: usize,
    const FF_DIM: usize,
    const NUM_LAYERS: usize,
    E: Dtype,
    D: DeviceStorage,
>(pub Repeated<TransformerDecoderBlock<MODEL_DIM, NUM_HEADS, FF_DIM, E, D>, NUM_LAYERS>);

impl<const M: usize, const H: usize, const F: usize, const L: usize, D: Device<f32>>
    BuildModule<D, f32> for TransformerDecoder<M, H, F, L, f32, D>
{
    fn try_build(device: &D) -> Result<Self, D::Err> {
        Ok(Self(BuildModule::try_build(device)?))
    }
}

impl<const M: usize, const H: usize, const F: usize, const L: usize, D: Device<f32>>
    ResetParams<D, f32> for TransformerDecoder<M, H, F, L, f32, D>
{
    fn try_reset_params(&mut self) -> Result<(), D::Err> {
        self.0.try_reset_params()
    }
}

impl<const M: usize, const H: usize, const F: usize, const L: usize, D: Device<f32>>
    GradientUpdate<D, f32> for TransformerDecoder<M, H, F, L, f32, D>
{
    fn update<U>(&mut self, updater: &mut U, unused: &mut UnusedTensors) -> Result<(), D::Err>
    where
        U: ParamUpdater<D, f32>,
    {
        self.0.update(updater, unused)
    }
}

impl<
        const M: usize,
        const H: usize,
        const F: usize,
        const L: usize,
        D1: Device<f32>,
        D2: Device<f32>,
    > ToDevice<D2> for TransformerDecoder<M, H, F, L, f32, D1>
{
    type Output = TransformerDecoder<M, H, F, L, f32, D2>;

    fn to_device(&self, device: &D2) -> Self::Output {
        TransformerDecoder(self.0.to_device(device))
    }
}

impl<const M: usize, const H: usize, const F: usize, const L: usize, D, Tgt, Mem: Clone>
    Module<(Tgt, Mem)> for TransformerDecoder<M, H, F, L, f32, D>
where
    D: Device<f32>,
    TransformerDecoderBlock<M, H, F, f32, D>: Module<(Tgt, Mem), Output = Tgt>,
{
    type Output = Tgt;
    fn forward(&self, (mut tgt, mem): (Tgt, Mem)) -> Self::Output {
        for block in self.0.modules.iter() {
            tgt = block.forward((tgt, mem.clone()));
        }
        tgt
    }
}

impl<const M: usize, const H: usize, const F: usize, const L: usize, D: Device<f32>, T> ModuleMut<T>
    for TransformerDecoder<M, H, F, L, f32, D>
where
    Self: Module<T>,
{
    type Output = <Self as Module<T>>::Output;

    fn forward_mut(&mut self, t: T) -> Self::Output {
        self.forward(t)
    }
}

/// **Requires Nightly** A transformer decoder block. Different than the normal transformer block
/// as this self attention accepts an additional sequence from the encoder.
///
/// Generics
/// - `MODEL_DIM`: The size of query/key/value tensors. Given to [MultiHeadAttention].
/// - `NUM_HEADS`: The number of heads in [MultiHeadAttention].
/// - `FF_DIM`: The size of the hidden layer in the feedforward network.
///
/// **Pytorch equivalent**:
/// ```python
/// decoder = torch.nn.TransformerDecoderLayer(
///    EMBED_DIM, NUM_HEADS, dim_feedforward=FF_DIM, batch_first=True, dropout=0.0
/// )
/// ```
/// TODO: Doctests
#[derive(Clone, Debug)]
pub struct TransformerDecoderBlock<
    const MODEL_DIM: usize,
    const NUM_HEADS: usize,
    const FF_DIM: usize,
    E: Dtype,
    D: DeviceStorage,
> {
    pub self_attn: MultiHeadAttention<MODEL_DIM, NUM_HEADS, MODEL_DIM, MODEL_DIM, E, D>,
    pub norm1: LayerNorm1D<MODEL_DIM, E, D>,
    pub mh_attn: MultiHeadAttention<MODEL_DIM, NUM_HEADS, MODEL_DIM, MODEL_DIM, E, D>,
    pub norm2: LayerNorm1D<MODEL_DIM, E, D>,
    pub ff: FF<MODEL_DIM, FF_DIM, E, D>,
    pub norm3: LayerNorm1D<MODEL_DIM, E, D>,
}

type FF<const M: usize, const F: usize, E, D> =
    Residual<(Linear<M, F, E, D>, ReLU, Linear<F, M, E, D>)>;

impl<const M: usize, const N: usize, const F: usize, D: Device<f32>> BuildModule<D, f32>
    for TransformerDecoderBlock<M, N, F, f32, D>
{
    fn try_build(device: &D) -> Result<Self, <D>::Err> {
        Ok(Self {
            self_attn: BuildModule::try_build(device)?,
            norm1: BuildModule::try_build(device)?,
            mh_attn: BuildModule::try_build(device)?,
            norm2: BuildModule::try_build(device)?,
            ff: BuildModule::try_build(device)?,
            norm3: BuildModule::try_build(device)?,
        })
    }
}

impl<const M: usize, const N: usize, const F: usize, D: Device<f32>> ResetParams<D, f32>
    for TransformerDecoderBlock<M, N, F, f32, D>
{
    fn try_reset_params(&mut self) -> Result<(), D::Err> {
        self.self_attn.try_reset_params()?;
        self.norm1.try_reset_params()?;
        self.mh_attn.try_reset_params()?;
        self.norm2.try_reset_params()?;
        self.ff.try_reset_params()?;
        self.norm3.try_reset_params()?;
        Ok(())
    }
}

impl<const M: usize, const H: usize, const F: usize, D: Device<f32>> GradientUpdate<D, f32>
    for TransformerDecoderBlock<M, H, F, f32, D>
{
    fn update<U>(&mut self, updater: &mut U, unused: &mut UnusedTensors) -> Result<(), <D>::Err>
    where
        U: ParamUpdater<D, f32>,
    {
        self.self_attn.update(updater, unused)?;
        self.norm1.update(updater, unused)?;
        self.mh_attn.update(updater, unused)?;
        self.norm2.update(updater, unused)?;
        self.ff.update(updater, unused)?;
        self.norm3.update(updater, unused)?;
        Ok(())
    }
}

impl<const M: usize, const H: usize, const F: usize, D1: Device<f32>, D2: Device<f32>> ToDevice<D2>
    for TransformerDecoderBlock<M, H, F, f32, D1>
{
    type Output = TransformerDecoderBlock<M, H, F, f32, D2>;

    fn to_device(&self, device: &D2) -> Self::Output {
        TransformerDecoderBlock {
            self_attn: self.self_attn.to_device(device),
            norm1: self.norm1.to_device(device),
            mh_attn: self.mh_attn.to_device(device),
            norm2: self.norm2.to_device(device),
            ff: self.ff.to_device(device),
            norm3: self.norm3.to_device(device),
        }
    }
}

impl<const M: usize, const H: usize, const F: usize, D: Device<f32>, Tgt, Mem> Module<(Tgt, Mem)>
    for TransformerDecoderBlock<M, H, F, f32, D>
where
    Tgt: SplitTape + std::ops::Add<Tgt::NoTape, Output = Tgt>,
    Mem: Clone,
    MultiHeadAttention<M, H, M, M, f32, D>:
        Module<Tgt, Output = Tgt> + Module<(Tgt, Mem, Mem), Output = Tgt>,
    LayerNorm1D<M, f32, D>: Module<Tgt, Output = Tgt>,
    FF<M, F, f32, D>: Module<Tgt, Output = Tgt>,
{
    type Output = Tgt;

    fn forward(&self, (tgt, mem): (Tgt, Mem)) -> Self::Output {
        let (tgt, tape) = tgt.split_tape();
        let x = self.self_attn.forward(tgt.clone().put_tape(tape));
        let x = x + tgt;
        let x = self.norm1.forward(x);

        let (x, tape) = x.split_tape();
        let x_residual = x.clone();
        let x = self.mh_attn.forward((x.put_tape(tape), mem.clone(), mem));
        let x = x + x_residual;
        let x = self.norm2.forward(x);

        let (x, tape) = x.split_tape();
        let x_residual = x.clone();
        let x = self.ff.forward(x.put_tape(tape));
        self.norm3.forward(x + x_residual)
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
    fn test_decoder_block_forward() {
        let dev = TestDevice::seed_from_u64(2);

        const BATCH: usize = 4;
        const S1: usize = 8;
        const S2: usize = 6;
        const EMBED_DIM: usize = 12;
        const NUM_HEADS: usize = 6;
        const FF_DIM: usize = 2;

        let decoder =
            builder::TransformerDecoderBlock::<EMBED_DIM, NUM_HEADS, FF_DIM>::build_on_device(&dev);

        let tgt = dev.sample_normal::<Rank3<BATCH, S1, EMBED_DIM>>();
        let mem = dev.sample_normal::<Rank3<BATCH, S2, EMBED_DIM>>();
        let y = decoder.forward((tgt, mem));

        // This expected y was generated by:
        // 1. saving `decoder` parameters, `tgt`, `mem` and `y` to a npz files
        // 2. Running pytorch with the same values
        // 3. printing out the output
        // See https://github.com/coreylowman/dfdx/wiki/Exporting-MultiHeadAttention-to-pytorch-for-unit-tests
        #[rustfmt::skip]
        assert_close(
            &y.array(),
            &[
                [
                    [-1.8282166, 0.4905981, -0.087768994, -1.7342027, 1.0992093, -0.0798184, 1.8210326, 0.51643336, -0.0900775, 0.14202805, -0.64658207, 0.39736462], 
                    [-1.6180319, 0.41205788, 2.3084683, 0.29447594, -1.4795923, 0.025965191, -0.073491804, -0.63043165, 0.24433309, 0.97292364, -0.4441398, -0.012536508], 
                    [-0.21012637, -1.0700799, 1.1185564, 0.11702357, 2.4426334, -0.44077793, 0.2791867, -0.01996643, -0.3875589, -1.6609113, 0.3541146, -0.52209365], 
                    [-0.18832761, -1.4596629, 1.1257894, -0.2821726, 1.9980078, 0.21567026, 0.62414205, -0.9968184, -0.63679254, -0.4882773, 1.1436064, -1.0551643], 
                    [1.4696937, -0.9947407, -0.6724616, 1.4575161, 0.20573238, 0.4842603, -0.47080624, -0.51556194, 0.1098688, -2.1217275, 0.9965636, 0.051663037], 
                    [-0.6896891, 1.3658246, 1.2250992, 0.7056474, -1.608127, -0.40726888, 0.77198154, -0.5834845, -0.715882, -1.159938, 1.4177737, -0.32193714], 
                    [-0.5677555, -1.4149233, -0.042877033, 0.42163137, -1.4898248, 0.18696952, -0.34932703, 0.025438592, 1.7562915, -0.09288755, 1.9663084, -0.39904404], 
                    [-0.29730833, 1.4479588, 0.48051843, -2.0483263, 0.19551115, -1.1314168, 0.50667816, -0.41707638, 1.8668332, -0.12400002, -0.34949628, -0.12987565]
                ],
                [
                    [1.0474187, -0.36726877, 0.6062487, 1.2266891, 0.23859267, -1.2492886, 0.6870168, -0.41522908, 0.1120279, -2.0997283, 1.1530482, -0.93952775], 
                    [-0.9826481, -1.6384554, 1.3260157, -0.640091, 1.6127218, -0.037127804, 1.0104855, -0.6337795, 0.9092963, -0.40781853, 0.4580551, -0.97665435], 
                    [0.5360814, 0.49061117, 0.4866183, -2.0523639, 0.1988024, -1.2422996, 1.661322, 0.49859166, 0.3697553, -0.5680553, 0.78520685, -1.1642704], 
                    [-0.7726515, 0.93001896, -0.66617763, 0.4694671, -0.8629417, -1.8444319, 0.592756, 1.0716428, 1.9540994, -0.5891582, -0.036607213, -0.24601617], 
                    [0.047874976, -0.6871031, 1.8607649, -0.8823096, -0.2810341, -2.1582916, 0.26303598, 0.8947123, 0.8470898, 0.7815403, -0.10450425, -0.5817757], 
                    [1.7058957, 1.2483338, 0.39164263, -1.1604545, -0.924757, 0.11749809, -0.7216374, -0.67561686, -1.0829433, 0.49353305, 1.4408779, -0.8323721], 
                    [0.22543618, 0.33764997, -0.35637832, -1.272513, 2.4268339, -0.69349843, -0.81749564, 1.1467252, -0.7818217, 0.75159174, -0.6975957, -0.2689344], 
                    [0.20468958, -1.6314541, 1.2027377, 0.028924545, 1.7649139, 0.4768978, 0.53304714, -1.4178132, -0.4016031, 0.38177067, 0.22049235, -1.362603]
                ], 
                [
                    [-0.84798545, -0.5011886, 0.66881865, 1.2499812, -0.71572924, 0.64659107, -0.6125077, 1.4220015, -0.07497008, -2.1808815, -0.018554857, 0.9644248], 
                    [0.94609004, 0.17612238, -0.6704886, -0.84303737, -1.5406336, 1.0308661, 0.9653437, -0.37604672, 1.9540386, -1.2240725, -0.21073216, -0.20744993], 
                    [0.84163773, -1.4423486, 0.36422232, 1.4976592, 0.18822181, -1.3313874, -0.63579834, 0.1270687, 1.2412841, -1.652398, 0.48612878, 0.3157095], 
                    [-0.8337842, 2.1203926, -0.049839802, 0.2749188, 1.582959, -0.15599866, -0.9533963, -0.59190226, -1.2471876, 0.46127546, 0.39930797, -1.0067449], 
                    [-0.4358268, -1.1307701, 2.875104, -0.2916147, 0.84277123, -0.7373065, -0.78446764, -0.05593639, 0.097207345, -0.34416202, 0.2818471, -0.31684548], 
                    [-0.47762555, -0.51836544, -0.21069026, -1.1951331, -1.4180412, -0.8607104, -0.37695575, 0.2937455, 1.2657883, 1.8674875, 0.2533957, 1.377105], 
                    [-0.63986486, -0.7149309, 1.2318974, -0.6326741, -0.7989294, 0.17683157, -0.9319836, -0.85409164, -0.27413294, 2.4560745, 0.88110965, 0.10069457], 
                    [-0.28522006, -0.64838225, 1.5290003, 0.9294214, -1.4483466, 0.7511488, 0.12228248, 0.2166715, -0.47714162, 1.5517772, -0.6593426, -1.5818686]
                ], 
                [
                    [-1.2329057, 0.44552323, 1.551554, -1.5302522, 0.85586065, 0.8786212, 0.45335567, 0.36507848, -1.3898871, 0.3460387, 0.4536145, -1.1966012], 
                    [-0.43035677, -0.6110903, 2.6479146, -0.20168099, -0.5650418, -0.9760975, 0.6419022, 0.407723, -0.037293993, 0.46435288, -1.5138568, 0.17352544], 
                    [-1.1756661, -1.0211056, 1.3271025, 0.72899985, 0.15518472, 0.8523727, -1.9683127, -0.48348847, 0.078370616, 1.0626128, -0.5494013, 0.99333084], 
                    [-0.31816253, -0.36110976, -1.924298, 0.45834693, 0.5747769, 1.6565135, -1.4361217, 0.27335188, 0.59128666, -0.66791457, -0.19240738, 1.3457379], 
                    [0.14595428, 0.11881049, 1.7536689, -0.6420458, 0.31913936, -1.0564035, 0.96772295, 0.89557886, 0.3448268, -0.27100906, -0.28569373, -2.2905495], 
                    [-0.48597512, -1.5939227, 1.3749553, 0.5797474, 0.56244403, -0.39264303, -0.14750348, 1.4252212, -1.6805621, 0.8884908, -0.86461866, 0.33436638], 
                    [0.050558347, 0.0741049, 1.2436341, -1.5460747, 0.5402772, 0.44131476, -0.25474393, 0.69687545, 0.6824534, -1.5245175, 1.2379053, -1.641787], 
                    [0.8219666, -1.3542501, 1.0846912, -0.3981854, 0.3517857, -0.121625386, -0.31963912, 1.6371974, -0.7659111, -0.35910767, 1.1866858, -1.7636077]
                ]
            ],
        );
    }
}
