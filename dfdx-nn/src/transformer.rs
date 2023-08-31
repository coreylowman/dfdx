use crate::*;

use dfdx::prelude::*;

#[derive(Clone, Debug, Sequential)]
#[built(FeedForward)]
pub struct FeedForwardConfig<Model: Dim, F: Dim> {
    pub l1: LinearConfig<Model, F>,
    pub act1: ReLU,
    pub l2: LinearConfig<F, Model>,
}

/// A single transformer encoder block
///
/// Generics
/// - `Model`: The size of query/key/value tensors. Given to [MultiHeadAttention].
/// - `NumHeads`: The number of heads in [MultiHeadAttention].
/// - `F`: The size of the hidden layer in the feedforward network.
///
/// **Pytorch equivalent**:
/// ```python
/// encoder = torch.nn.TransformerEncoderLayer(
///    Model, NumHeads, dim_feedforward=F, batch_first=True, dropout=0.0
/// )
/// ```
#[derive(Clone, Debug, Sequential)]
#[built(EncoderBlock)]
pub struct EncoderBlockConfig<Model: Dim, NumHeads: Dim, F: Dim> {
    pub self_attn: ResidualAdd<MultiHeadAttentionConfig<Model, NumHeads>>,
    pub norm1: LayerNorm1DConfig<Model>,
    pub ff: ResidualAdd<FeedForwardConfig<Model, F>>,
    pub norm2: LayerNorm1DConfig<Model>,
}

impl<Model: Dim, NumHeads: Dim, F: Dim> EncoderBlockConfig<Model, NumHeads, F> {
    pub fn new(model: Model, num_heads: NumHeads, f: F) -> Self {
        EncoderBlockConfig {
            self_attn: ResidualAdd(MultiHeadAttentionConfig::new(
                model, num_heads, model, model,
            )),
            norm1: LayerNorm1DConfig(model),
            ff: ResidualAdd(FeedForwardConfig {
                l1: LinearConfig::new(model, f),
                act1: ReLU,
                l2: LinearConfig::new(f, model),
            }),
            norm2: LayerNorm1DConfig(model),
        }
    }
}

/// A transformer decoder block. Different than the normal transformer block
/// as this self attention accepts an additional sequence from the encoder.
///
/// Generics
/// - `Model`: The size of query/key/value tensors. Given to [MultiHeadAttention].
/// - `NumHeads`: The number of heads in [MultiHeadAttention].
/// - `F`: The size of the hidden layer in the feedforward network.
///
/// **Pytorch equivalent**:
/// ```python
/// decoder = torch.nn.TransformerDecoderLayer(
///    Model, NumHeads, dim_feedforward=F, batch_first=True, dropout=0.0
/// )
/// ```
#[derive(Clone, Debug, CustomModule)]
#[built(DecoderBlock)]
pub struct DecoderBlockConfig<Model: Dim, NumHeads: Dim, F: Dim> {
    #[module]
    pub self_attn: ResidualAdd<MultiHeadAttentionConfig<Model, NumHeads>>,
    #[module]
    pub norm1: LayerNorm1DConfig<Model>,
    #[module]
    pub mh_attn: MultiHeadAttentionConfig<Model, NumHeads>,
    #[module]
    pub norm2: LayerNorm1DConfig<Model>,
    #[module]
    pub ff: ResidualAdd<FeedForwardConfig<Model, F>>,
    #[module]
    pub norm3: LayerNorm1DConfig<Model>,
}

impl<Model: Dim, NumHeads: Dim, F: Dim> DecoderBlockConfig<Model, NumHeads, F> {
    pub fn new(model: Model, num_heads: NumHeads, f: F) -> Self {
        DecoderBlockConfig {
            self_attn: ResidualAdd(MultiHeadAttentionConfig::new(
                model, num_heads, model, model,
            )),
            norm1: LayerNorm1DConfig(model),
            mh_attn: MultiHeadAttentionConfig::new(model, num_heads, model, model),
            norm2: LayerNorm1DConfig(model),
            ff: ResidualAdd(FeedForwardConfig {
                l1: LinearConfig::new(model, f),
                act1: ReLU,
                l2: LinearConfig::new(f, model),
            }),
            norm3: LayerNorm1DConfig(model),
        }
    }
}

impl<M: Dim, H: Dim, F: Dim, E: Dtype, D: Device<E>, Tgt, Mem> Module<(Tgt, Mem)>
    for DecoderBlock<M, H, F, E, D>
where
    Tgt: WithEmptyTape + SplitTape + TryAdd<Tgt::NoTape, Output = Tgt> + HasErr<Err = D::Err>,
    Mem: Clone,
    ResidualAdd<MultiHeadAttention<M, H, M, M, E, D>>: Module<Tgt, Output = Tgt, Error = D::Err>,
    MultiHeadAttention<M, H, M, M, E, D>: Module<(Tgt, Mem, Mem), Output = Tgt, Error = D::Err>,
    LayerNorm1D<M, E, D>: Module<Tgt, Output = Tgt, Error = D::Err>,
    ResidualAdd<FeedForward<M, F, E, D>>: Module<Tgt, Output = Tgt, Error = D::Err>,
{
    type Output = Tgt;
    type Error = D::Err;

    fn try_forward(&self, (tgt, mem): (Tgt, Mem)) -> Result<Self::Output, D::Err> {
        let x = self.self_attn.try_forward(tgt)?;
        let x = self.norm1.try_forward(x)?;

        let (x, tape) = x.split_tape();
        let x_residual = x.clone();
        let x = self
            .mh_attn
            .try_forward((x.put_tape(tape), mem.clone(), mem))?;
        let x = x.try_add(x_residual)?;
        let x = self.norm2.try_forward(x)?;
        let x = self.ff.try_forward(x)?;
        self.norm3.try_forward(x)
    }
}

/// Transformer architecture as described in
/// [Attention is all you need](https://arxiv.org/abs/1706.03762).
///
/// This is comprised of a [EncoderBlockConfig] and a [DecoderBlockConfig].
///
/// Generics:
/// - `Model`: Size of the input features to the encoder/decoder.
/// - `NumHeads`: Number of heads for [MultiHeadAttention].
/// - `F`: Feedforward hidden dimension for both encoder/decoder
///
/// **Pytorch equivalent**:
/// ```python
/// torch.nn.Transformer(
///     d_model=Model,
///     nhead=NumHeads,
///     num_encoder_layers=cfg.encoder.len(),
///     num_decoder_layers=cfg.decoder.len(),
///     dim_feedforward=F,
///     batch_first=True,
/// )
/// ```
#[derive(Clone, Debug, CustomModule)]
#[built(Transformer)]
pub struct TransformerConfig<Model: Dim, NumHeads: Dim, F: Dim> {
    #[module]
    pub encoder: Vec<EncoderBlockConfig<Model, NumHeads, F>>,
    #[module]
    pub decoder: Vec<DecoderBlockConfig<Model, NumHeads, F>>,
}

impl<Model: Dim, NumHeads: Dim, F: Dim> TransformerConfig<Model, NumHeads, F> {
    pub fn new(
        model: Model,
        num_heads: NumHeads,
        f: F,
        num_encoder_layers: usize,
        num_decoder_layers: usize,
    ) -> Self {
        let mut encoder = Vec::with_capacity(num_encoder_layers);
        for _ in 0..num_encoder_layers {
            encoder.push(EncoderBlockConfig::new(model, num_heads, f));
        }
        let mut decoder = Vec::with_capacity(num_decoder_layers);
        for _ in 0..num_decoder_layers {
            decoder.push(DecoderBlockConfig::new(model, num_heads, f));
        }
        Self { encoder, decoder }
    }
}

impl<M: Dim, H: Dim, F: Dim, E: Dtype, D: Device<E>, Src: SplitTape, Tgt: PutTape<Src::Tape>>
    Module<(Src, Tgt)> for Transformer<M, H, F, E, D>
where
    Vec<EncoderBlock<M, H, F, E, D>>: Module<Src, Output = Src, Error = D::Err>,
    DecoderBlock<M, H, F, E, D>: Module<
        (<Tgt as PutTape<Src::Tape>>::Output, Src::NoTape),
        Output = <Tgt as PutTape<Src::Tape>>::Output,
        Error = D::Err,
    >,
{
    type Output = <Tgt as PutTape<Src::Tape>>::Output;
    type Error = D::Err;

    fn try_forward(&self, (src, tgt): (Src, Tgt)) -> Result<Self::Output, D::Err> {
        let (mem, tape) = self.encoder.try_forward(src)?.split_tape();
        let mut tgt = tgt.put_tape(tape);
        for block in self.decoder.iter() {
            tgt = block.try_forward((tgt, mem.clone()))?;
        }
        Ok(tgt)
    }
}

#[cfg(test)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    #[test]
    fn test_transformer_forward() {
        let dev = TestDevice::seed_from_u64(0);
        let mut t = dev.build_module::<TestDtype>(TransformerConfig::new(
            Const::<16>,
            Const::<4>,
            Const::<8>,
            3,
            3,
        ));

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
    fn test_transformer_backward() {
        let dev = TestDevice::seed_from_u64(0);
        let mut t = dev.build_module::<TestDtype>(TransformerConfig::new(
            Const::<16>,
            Const::<4>,
            Const::<8>,
            3,
            3,
        ));

        let src = dev.sample_normal::<Rank3<4, 12, 16>>();
        let tgt = dev.sample_normal::<Rank3<4, 6, 16>>();
        let out: Tensor<Rank3<4, 6, 16>, _, _, _> = t.forward_mut((src.leaky_trace(), tgt));
        let g = out.mean().backward();

        let mut opt = crate::optim::Sgd::new(&t, Default::default());
        opt.update(&mut t, &g).expect("");
    }

    #[test]
    fn test_encoder_block_forward() {
        let dev = TestDevice::seed_from_u64(2);

        const BATCH: usize = 3;
        const SEQ_LEN: usize = 5;
        const EMBED_DIM: usize = 9;
        const NUM_HEADS: usize = 3;
        const FF_DIM: usize = 16;

        type Dtype = f32;

        let encoder = dev.build_module::<Dtype>(EncoderBlockConfig::new(
            Const::<EMBED_DIM>,
            Const::<NUM_HEADS>,
            Const::<FF_DIM>,
        ));

        let x: Tensor<Rank3<BATCH, SEQ_LEN, EMBED_DIM>, Dtype, _> = dev.sample_normal();
        let y = encoder.forward(x);

        // This expected y was generated by:
        // 1. saving `encoder` parameters, `x` and `y` to a npz files
        // 2. Running pytorch with the same values
        // 3. printing out the output
        // See https://github.com/coreylowman/dfdx/wiki/Exporting-MultiHeadAttention-to-pytorch-for-unit-tests
        #[rustfmt::skip]
        assert_close_to_literal!(
            y,
            [
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
            ]
        );
    }

    #[test]
    fn test_decoder_block_forward() {
        let dev = TestDevice::seed_from_u64(2);

        const BATCH: usize = 4;
        const S1: usize = 8;
        const S2: usize = 6;
        const EMBED_DIM: usize = 12;
        const NUM_HEADS: usize = 6;
        const FF_DIM: usize = 2;

        type Dtype = f32;

        let decoder = dev.build_module::<Dtype>(DecoderBlockConfig::new(
            Const::<EMBED_DIM>,
            Const::<NUM_HEADS>,
            Const::<FF_DIM>,
        ));

        let tgt: Tensor<Rank3<BATCH, S1, EMBED_DIM>, Dtype, _> = dev.sample_normal();
        let mem: Tensor<Rank3<BATCH, S2, EMBED_DIM>, Dtype, _> = dev.sample_normal();
        let y = decoder.forward((tgt, mem));

        // This expected y was generated by:
        // 1. saving `decoder` parameters, `tgt`, `mem` and `y` to a npz files
        // 2. Running pytorch with the same values
        // 3. printing out the output
        // See https://github.com/coreylowman/dfdx/wiki/Exporting-MultiHeadAttention-to-pytorch-for-unit-tests
        #[rustfmt::skip]
        assert_close_to_literal!(
            y,
            [
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
            ]
        );
    }
}
