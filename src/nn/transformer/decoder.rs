use super::mha::MultiHeadAttention;
use crate::gradients::{CanUpdateWithGradients, GradientProvider, UnusedTensors};
use crate::prelude::*;
use crate::tensor_ops::utils::BinaryOpTyping;
use rand::Rng;

/// **Requires Nightly** A transformer decoder.
///
/// Generics
/// - `MODEL_DIM`: The size of query/key/value tensors. Given to [MultiHeadAttention].
/// - `NUM_HEADS`: The number of heads in [MultiHeadAttention].
/// - `FF_DIM`: The size of the hidden layer in
///   the feedforward network in [TransformerDecoderBlock].
/// - `NUM_LAYERS`: The number of [TransformerDecoderBlock] to use.
/// TODO: Doctests
#[derive(Clone, Debug, Default)]
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
    Mem: Tensor<NoTape = Mem> + Clone,
    TransformerDecoderBlock<M, H, F>: Module<(Tgt, Mem), Output = Tgt>,
{
    type Output = Tgt;

    fn forward(&self, (mut x, mem): (Tgt, Mem)) -> Self::Output {
        for block in self.0.modules.iter() {
            x = block.forward((x, mem.clone()));
        }
        x
    }
}

impl<const M: usize, const H: usize, const F: usize, const L: usize, T> ModuleMut<T>
    for TransformerDecoder<M, H, F, L>
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
#[derive(Clone, Default, Debug)]
pub struct TransformerDecoderBlock<
    const MODEL_DIM: usize,
    const NUM_HEADS: usize,
    const FF_DIM: usize,
> {
    pub self_attn: MultiHeadAttention<MODEL_DIM, NUM_HEADS>,
    pub norm1: LayerNorm1D<MODEL_DIM>,
    pub mh_attn: MultiHeadAttention<MODEL_DIM, NUM_HEADS>,
    pub norm2: LayerNorm1D<MODEL_DIM>,
    pub ff: FF<MODEL_DIM, FF_DIM>,
    pub norm3: LayerNorm1D<MODEL_DIM>,
}

type FF<const M: usize, const F: usize> = Residual<(Linear<M, F>, ReLU, Linear<F, M>)>;

impl<const MODEL_DIM: usize, const NUM_HEADS: usize, const FF_DIM: usize> ResetParams
    for TransformerDecoderBlock<MODEL_DIM, NUM_HEADS, FF_DIM>
{
    fn reset_params<R: Rng>(&mut self, rng: &mut R) {
        self.self_attn.reset_params(rng);
        self.norm1.reset_params(rng);
        self.mh_attn.reset_params(rng);
        self.norm2.reset_params(rng);
        self.ff.reset_params(rng);
        self.norm3.reset_params(rng);
    }
}

impl<const MODEL_DIM: usize, const NUM_HEADS: usize, const FF_DIM: usize> CanUpdateWithGradients
    for TransformerDecoderBlock<MODEL_DIM, NUM_HEADS, FF_DIM>
{
    fn update<G: GradientProvider>(&mut self, grads: &mut G, unused: &mut UnusedTensors) {
        self.self_attn.update(grads, unused);
        self.norm1.update(grads, unused);
        self.mh_attn.update(grads, unused);
        self.norm2.update(grads, unused);
        self.ff.update(grads, unused);
        self.norm3.update(grads, unused);
    }
}

impl<const M: usize, const H: usize, const F: usize, Tgt, Mem> Module<(Tgt, Mem)>
    for TransformerDecoderBlock<M, H, F>
where
    Tgt: Tensor<Dtype = f32> + BinaryOpTyping<Tgt::NoTape, Out = Tgt>,
    Mem: Tensor<Dtype = f32, NoTape = Mem> + Clone,
    MultiHeadAttention<M, H>: Module<(Tgt, Tgt::NoTape, Tgt::NoTape), Output = Tgt>
        + Module<(Tgt, Mem, Mem), Output = Tgt>,
    LayerNorm1D<M>: Module<Tgt, Output = Tgt>,
    FF<M, F>: Module<Tgt, Output = Tgt>,
{
    type Output = Tgt;

    fn forward(&self, (tgt, mem): (Tgt, Mem)) -> Self::Output {
        let (tgt, tape) = tgt.split_tape();
        let x = self
            .self_attn
            .forward((tgt.clone().put_tape(tape), tgt.clone(), tgt.clone()));
        let x = add(x, tgt);
        let x = self.norm1.forward(x);

        let (x, tape) = x.split_tape();
        let x_ = x.clone();
        let x = self.mh_attn.forward((x.put_tape(tape), mem.clone(), mem));
        let x = add(x, x_);
        let x = self.norm2.forward(x);
        let x = self.ff.forward(x);
        self.norm3.forward(x)
    }
}

#[cfg(feature = "nightly")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::assert_close;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn test_decoder_block_forward() {
        let mut rng = StdRng::seed_from_u64(2);

        const BATCH: usize = 4;
        const S1: usize = 8;
        const S2: usize = 6;
        const EMBED_DIM: usize = 12;
        const NUM_HEADS: usize = 6;
        const FF_DIM: usize = 2;

        let mut decoder: TransformerDecoderBlock<EMBED_DIM, NUM_HEADS, FF_DIM> = Default::default();
        decoder.reset_params(&mut rng);

        let tgt: Tensor3D<BATCH, S1, EMBED_DIM> = TensorCreator::randn(&mut rng);
        let mem: Tensor3D<BATCH, S2, EMBED_DIM> = TensorCreator::randn(&mut rng);
        let y: Tensor3D<BATCH, S1, EMBED_DIM> = decoder.forward((tgt, mem));

        // This expected y was generated by:
        // 1. saving `decoder` parameters, `tgt`, `mem` and `y` to a npz files
        // 2. Running pytorch with the same values
        // 3. printing out the output
        // See https://github.com/coreylowman/dfdx/wiki/Exporting-MultiHeadAttention-to-pytorch-for-unit-tests
        #[rustfmt::skip]
        assert_close(
            y.data(),
            &[
                [
                    [-1.87558722, 0.45965099, 0.20498508,-1.73645127, 1.19475269,-0.07198015, 1.87802076, 0.18534835, 0.09591459,-0.19824848,-0.35261178, 0.21620668],
                    [-1.65146410, 0.36979428, 2.44077325, 0.06124005,-1.35236311, 0.06834260, 0.15826070,-0.82507777, 0.37757808, 0.65084165,-0.26028851,-0.03763753],
                    [-0.30696073,-0.83636290, 1.20258296, 0.11318116, 2.23617601,-0.58318114, 0.66371393,-0.26198950,-0.46798199,-1.64899850, 0.63527161,-0.74545103],
                    [-0.23854624,-1.12693906, 1.16869855,-0.19282928, 1.83873713,-0.11721543, 1.00944722,-0.97332841,-0.75959450,-0.69980252, 1.23692346,-1.14555120],
                    [1.36781275,-1.00360036,-0.45941362, 1.16563404, 0.24138503, 0.51682448,-0.20305091,-0.68849629, 0.21949562,-2.32909155, 1.11119950, 0.06130134],
                    [-0.70381856, 1.24304760, 1.32746470, 0.43500248,-1.45963287,-0.33785006, 0.95192397,-0.72454590,-0.56011575,-1.33778274, 1.46311414,-0.29680732],
                    [-0.72720474,-1.29362297, 0.24656427, 0.25788289,-1.20061839, 0.20161679,-0.18183309,-0.28182927, 1.85331190,-0.41204709, 2.05122447,-0.51344484],
                    [-0.45356780, 1.31273413, 0.69735909,-1.96937740, 0.33488208,-0.99047261, 0.59060574,-0.65752614, 1.89437556,-0.41522720,-0.09553659,-0.24824893],
                ],
                [
                    [0.92695564,-0.37954834, 0.74523187, 0.91893858, 0.26190025,-1.12540352, 0.87693417,-0.56255865, 0.20910029,-2.21528411, 1.21251309,-0.86877924],
                    [-0.94927889,-1.28225541, 1.38664925,-0.47819123, 1.60083365,-0.25243780, 1.21168947,-0.77403182, 0.60282439,-0.67139530, 0.72949010,-1.12389636],
                    [0.32318670, 0.44635653, 0.69037175,-2.00356507, 0.31796345,-1.09540510, 1.65720248, 0.18892130, 0.52996045,-0.80869401, 0.91539401,-1.16169262],
                    [-0.93624949, 0.90174866,-0.35485053, 0.28630549,-0.67549163,-1.74944031, 0.75101191, 0.73161471, 2.11734390,-0.91214812, 0.20135719,-0.36120197],
                    [-0.12938653,-0.65747797, 2.05397773,-1.01142454,-0.12065405,-2.02726126, 0.42845321, 0.56529117, 1.02239680, 0.41882706, 0.12460811,-0.66735017],
                    [1.61325872, 1.18383896, 0.58100909,-1.39098096,-0.86362296, 0.16341744,-0.44804084,-0.85499638,-0.94598162, 0.20620863, 1.56031752,-0.80442756],
                    [0.15400597, 0.30694833,-0.10923728,-1.54726267, 2.59482384,-0.72448921,-0.47337827, 0.94458705,-0.74652761, 0.43154043,-0.49556813,-0.33544219],
                    [0.06703589,-1.33028281, 1.29519308, 0.01789100, 1.73138475, 0.11349702, 0.98292470,-1.37452459,-0.57708341,-0.04158162, 0.54672015,-1.43117404],
                ],
                [
                    [-1.13928354,-0.41951340, 1.02809525, 1.10831285,-0.37338197, 0.62760144,-0.49609870, 0.89603722, 0.28748062,-2.46635914, 0.32486960, 0.62223953],
                    [0.66343045, 0.17840990,-0.32520610,-0.91180247,-1.24669814, 0.98684084, 1.03520977,-0.66813290, 2.06043386,-1.47457957, 0.05163103,-0.34953672],
                    [0.70942575,-1.41629028, 0.57625329, 1.22837853, 0.26442787,-1.24242258,-0.38967255,-0.10485345, 1.34950197,-1.88799143, 0.64463151, 0.26861122],
                    [-0.90124643, 2.06094766, 0.20568365, 0.06078637, 1.68658400,-0.19301027,-0.56969130,-0.80906254,-1.20984066, 0.12565698, 0.62286967,-1.07967734],
                    [-0.58323914,-0.91550159, 2.76294446,-0.23104562, 1.03537095,-0.79180622,-0.30585235,-0.37028444, 0.06941666,-0.66646379, 0.61295509,-0.61649406],
                    [-0.69953281,-0.53587002, 0.10623999,-1.43030167,-1.28995168,-0.84757996,-0.18267554,-0.03703059, 1.55741370, 1.54363191, 0.52537125, 1.29028559],
                    [-0.70696884,-0.75943643, 1.45195222,-0.89612883,-0.74769866, 0.21710433,-0.64992350,-1.06435382,-0.16617794, 2.16994262, 1.05082333, 0.10086535],
                    [-0.37381354,-0.70111430, 1.83576059, 0.72364914,-1.35405958, 0.72988695, 0.52067578,-0.01720174,-0.46059695, 1.23575497,-0.43288255,-1.70605886],
                ],
                [
                    [-1.20804095, 0.38654494, 1.65309286,-1.20736289, 1.07261550, 0.46114275, 0.83086872,-0.01955486,-1.26059496,-0.11887560, 0.79357809,-1.38341355],
                    [-0.56300515,-0.59784967, 2.81054258,-0.37848800,-0.41372916,-0.90938121, 0.82510620, 0.12329611, 0.14460202, 0.12636989,-1.24349451, 0.07603064],
                    [-1.36658132,-1.11734688, 1.74118745, 0.56276298, 0.35426524, 0.82628661,-1.63426054,-0.80171925, 0.09229738, 0.71951282,-0.27681157, 0.90040714],
                    [-0.47256982,-0.39320827,-1.71228957, 0.24000385, 0.71217608, 1.75911832,-1.24219942,-0.00148612, 0.80727738,-1.04095078, 0.02052352, 1.32360506],
                    [-0.00462395, 0.10117173, 1.83498573,-0.69001645, 0.46190643,-1.00014806, 1.14456511, 0.55384815, 0.36776620,-0.55358148,-0.00812254,-2.20775104],
                    [-0.59229124,-1.63409364, 1.70002937, 0.40580338, 0.76335514,-0.50594056, 0.32149875, 1.17081654,-1.73462892, 0.50679129,-0.56456679, 0.16322602],
                    [-0.28135568, 0.12212670, 1.39109802,-1.15742660, 0.81334966, 0.21747869,-0.01345161, 0.15832950, 0.68586451,-1.60281539, 1.38292646,-1.71612430],
                    [0.52762824,-1.20023167, 1.34064293,-0.40414453, 0.61767668,-0.24842866, 0.06679908, 1.13988364,-0.66101944,-0.71850598, 1.43029106,-1.89059174],
                ],
            ],
        );
    }
}
