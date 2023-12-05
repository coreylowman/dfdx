use crate::prelude::*;

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
    Tgt: WithEmptyTape + SplitTape + TryAdd<Tgt::NoTape, Output = Tgt>,
    Mem: Clone,
    ResidualAdd<MultiHeadAttention<M, H, M, M, E, D>>: Module<Tgt, Output = Tgt>,
    MultiHeadAttention<M, H, M, M, E, D>: Module<(Tgt, Mem, Mem), Output = Tgt>,
    LayerNorm1D<M, E, D>: Module<Tgt, Output = Tgt>,
    ResidualAdd<FeedForward<M, F, E, D>>: Module<Tgt, Output = Tgt>,
{
    type Output = Tgt;
    fn try_forward(&self, (tgt, mem): (Tgt, Mem)) -> Result<Self::Output, crate::tensor::Error> {
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
    Vec<EncoderBlock<M, H, F, E, D>>: Module<Src, Output = Src>,
    DecoderBlock<M, H, F, E, D>: Module<
        (<Tgt as PutTape<Src::Tape>>::Output, Src::NoTape),
        Output = <Tgt as PutTape<Src::Tape>>::Output,
    >,
{
    type Output = <Tgt as PutTape<Src::Tape>>::Output;
    fn try_forward(&self, (src, tgt): (Src, Tgt)) -> Result<Self::Output, crate::tensor::Error> {
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
            Const::<2>, Const::<2>, Const::<2>, 2, 2,
        ));

        // unbatched
        let src = dev.sample_normal::<Rank2<3, 2>>();
        let tgt = dev.sample_normal::<Rank2<5, 2>>();
        let _: Tensor<Rank2<5, 2>, _, _, _> = t.forward_mut((src, tgt));

        // batched
        let src = dev.sample_normal::<Rank3<2, 4, 2>>();
        let tgt = dev.sample_normal::<Rank3<2, 2, 2>>();
        let _: Tensor<Rank3<2, 2, 2>, _, _, _> = t.forward_mut((src, tgt));
    }

    #[test]
    fn test_transformer_backward() {
        let dev = TestDevice::seed_from_u64(0);

        let mut t = dev.build_module::<TestDtype>(TransformerConfig::new(
            Const::<2>, Const::<2>, Const::<2>, 2, 2,
        ));

        let src = dev.sample_normal::<Rank3<2, 4, 2>>();
        let tgt = dev.sample_normal::<Rank3<2, 2, 2>>();
        let out: Tensor<Rank3<2, 2, 2>, _, _, _> = t.forward_mut((src.leaky_trace(), tgt));
        let g = out.mean().backward();

        let mut opt = crate::nn::optim::Sgd::new(&t, Default::default());
        opt.update(&mut t, &g).expect("");
    }

    #[test]
    fn test_encoder_block_forward() {
        let dev = TestDevice::seed_from_u64(2);

        const BATCH: usize = 2;
        const SEQ_LEN: usize = 3;
        const EMBED_DIM: usize = 4;
        const NUM_HEADS: usize = 2;
        const FF_DIM: usize = 2;

        type Dtype = f32;

        let encoder = dev.build_module::<Dtype>(EncoderBlockConfig::new(
            Const::<EMBED_DIM>,
            Const::<NUM_HEADS>,
            Const::<FF_DIM>,
        ));

        let x: Tensor<Rank3<BATCH, SEQ_LEN, EMBED_DIM>, Dtype, _> = dev.sample_normal();

        // uncomment to save for this specific test params and inputs
        //
        // encoder.save_safetensors("encoder.safetensor").unwrap();
        // x.save_safetensors("x.safetensor").unwrap();

        let y = encoder.forward(x);

        // uncomment to save for this specific test params and inputs
        //
        // y.save_safetensors("y.safetensor").unwrap();

        // This expected y was generated by:
        // 1. saving `encoder` parameters, `x` and `y` to a npz files
        // 2. Running pytorch with the same values
        // 3. printing out the output
        // See https://github.com/coreylowman/dfdx/wiki/Exporting-MultiHeadAttention-to-pytorch-for-unit-tests
        assert_close_to_literal!(
            y,
            [
                [
                    [-1.7209842, 0.6216407, 0.7037436, 0.39559996],
                    [0.53576326, -1.4666773, 1.2166189, -0.28570476],
                    [-1.3280064, 0.42387456, -0.45566577, 1.3597975]
                ],
                [
                    [0.89139193, -1.2803736, 1.0577338, -0.668752],
                    [-0.41001588, 1.6245831, -1.084222, -0.13034514],
                    [0.9247901, -1.1639801, -0.8187512, 1.0579412]
                ]
            ]
        );
    }

    #[test]
    fn test_decoder_block_forward() {
        let dev = TestDevice::seed_from_u64(2);

        const BATCH: usize = 2;
        const S1: usize = 3;
        const S2: usize = 2;
        const EMBED_DIM: usize = 4;
        const NUM_HEADS: usize = 2;
        const FF_DIM: usize = 2;

        type Dtype = f32;

        let decoder = dev.build_module::<Dtype>(DecoderBlockConfig::new(
            Const::<EMBED_DIM>,
            Const::<NUM_HEADS>,
            Const::<FF_DIM>,
        ));

        let tgt: Tensor<Rank3<BATCH, S1, EMBED_DIM>, Dtype, _> = dev.sample_normal();
        let mem: Tensor<Rank3<BATCH, S2, EMBED_DIM>, Dtype, _> = dev.sample_normal();

        // uncomment to save for this specific test params and inputs
        //
        // decoder.save_safetensors("decoder.safetensor").unwrap();
        // tgt.save_safetensors("tgt.safetensor").unwrap();
        // mem.save_safetensors("mem.safetensor").unwrap();

        let y = decoder.forward((tgt, mem));

        // uncomment to save for this specific test params and inputs
        //
        // y.save_safetensors("y.safetensor").unwrap();

        println!("{:?}", y.array());

        // This expected y was generated by:
        // 1. saving `decoder` parameters, `tgt`, `mem` and `y` to a npz files
        // 2. Running pytorch with the same values
        // 3. printing out the output
        // See https://github.com/coreylowman/dfdx/wiki/Exporting-MultiHeadAttention-to-pytorch-for-unit-tests
        assert_close_to_literal!(
            y,
            [
                [
                    [0.94532686, -0.46526614, 0.93781346, -1.4178741],
                    [1.6348482, -1.0348053, -0.49546495, -0.10457793],
                    [0.8033758, 1.1668185, -0.823479, -1.146715]
                ],
                [
                    [1.2232355, -1.5628394, 0.2116476, 0.12795626],
                    [0.99152863, -0.98818815, 1.0083598, -1.0117002],
                    [-1.4775288, 0.47518563, -0.23662777, 1.2389709]
                ]
            ]
        );
    }
}
