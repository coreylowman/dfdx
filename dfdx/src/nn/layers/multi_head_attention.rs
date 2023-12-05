use crate::prelude::*;

use num_traits::Float;

/// A multi-head attention layer.
///
/// Generics:
/// - `Embed`: The size of query vectors.
/// - `NumHeads` The number of heads to split query/key/value into.
/// - *Optional* `K`: The size of key vectors. Defaults to `Embed`
/// - *Optional* `V` The size of value vectors. Defaults to `Embed`
///
/// **Pytorch equivalent**: `torch.nn.MultiheadAttention(Embed, NumHeads, batch_first=True)`
///
/// Examples
/// - `MultiHeadAttention<8, 2>` is an attention layer with 2 heads and 8 token, key and value dims.
/// - `MultiHeadAttention<8, 2, 6, 4>` is an attention layer with the key and value dimension different
///   than the embed dimension
#[derive(Default, Debug, Copy, Clone, CustomModule)]
#[built(MultiHeadAttention)]
pub struct MultiHeadAttentionConfig<Embed: Dim, NumHeads: Dim, K: Dim = Embed, V: Dim = Embed> {
    #[module]
    pub w_q: LinearConfig<Embed, K>,
    #[module]
    pub w_k: LinearConfig<Embed, K>,
    #[module]
    pub w_v: LinearConfig<Embed, V>,
    #[module]
    pub w_o: LinearConfig<V, Embed>,
    pub num_heads: NumHeads,
    pub k_dim: K,
    pub v_dim: V,
}

impl<Embed: Dim, NumHeads: Dim, K: Dim, V: Dim> MultiHeadAttentionConfig<Embed, NumHeads, K, V> {
    pub fn new(embed: Embed, num_heads: NumHeads, k: K, v: V) -> Self {
        assert!(
            k.size() % num_heads.size() == 0 && v.size() % num_heads.size() == 0,
            "NUM_HEADS must divide K_DIM & V_DIM evenly! If you haven't specified K_DIM & V_DIM, they default to EMBED_DIM, which means NUM_HEADS must divide EMBED_DIM evenly."
        );
        Self {
            w_q: LinearConfig::new(embed, k),
            w_k: LinearConfig::new(embed, k),
            w_v: LinearConfig::new(embed, v),
            w_o: LinearConfig::new(v, embed),
            num_heads,
            k_dim: k,
            v_dim: v,
        }
    }
}

impl<M: Dim, H: Dim, K: Dim, V: Dim, E, D, S1, S2, T>
    Module<(
        Tensor<(S1, M), E, D, T>,
        Tensor<(S2, M), E, D>,
        Tensor<(S2, M), E, D>,
    )> for MultiHeadAttention<M, H, K, V, E, D>
where
    E: Dtype + Float,
    D: Device<E>,
    S1: Dim,
    S2: Dim,
    T: Tape<E, D>,
{
    type Output = Tensor<(S1, M), E, D, T>;

    /// Encoder-Decoder style self attention where one set of tensors is used for values and keys, and another is used for queries
    fn try_forward(
        &self,
        (q, k, v): (
            Tensor<(S1, M), E, D, T>,
            Tensor<(S2, M), E, D>,
            Tensor<(S2, M), E, D>,
        ),
    ) -> Result<Self::Output, crate::tensor::Error> {
        assert_eq!(k.shape().0, v.shape().0);
        let (s1, m) = *q.shape();
        let s2 = k.shape().0;
        let q = q.broadcast_like(&(Const::<1>, s1, m));
        let k = k.broadcast_like(&(Const::<1>, s2, m));
        let v = v.broadcast_like(&(Const::<1>, s2, m));
        let out = self.try_forward((q, k, v))?;
        out.try_reshape_like(&(s1, m))
    }
}

impl<M: Dim, H: Dim, K: Dim, V: Dim, E, D, B, S1, S2, T>
    Module<(
        Tensor<(B, S1, M), E, D, T>,
        Tensor<(B, S2, M), E, D>,
        Tensor<(B, S2, M), E, D>,
    )> for MultiHeadAttention<M, H, K, V, E, D>
where
    E: Dtype + Float,
    D: Device<E>,
    B: Dim,
    S1: Dim,
    S2: Dim,
    T: Tape<E, D>,
{
    type Output = Tensor<(B, S1, M), E, D, T>;

    /// Batched Encoder-Decoder style self attention where one set of tensors is used for values and keys, and another is used for queries
    fn try_forward(
        &self,
        (q, k, v): (
            Tensor<(B, S1, M), E, D, T>,
            Tensor<(B, S2, M), E, D>,
            Tensor<(B, S2, M), E, D>,
        ),
    ) -> Result<Self::Output, crate::tensor::Error> {
        assert_eq!(q.shape().0, k.shape().0);
        assert_eq!(q.shape().0, v.shape().0);
        assert_eq!(k.shape().1, v.shape().1);

        let (b, s1, _) = *q.shape();
        let s2 = v.shape().1;
        let h_dim = self.num_heads.size();
        let k_dim = self.k_dim.size();
        let v_dim = self.v_dim.size();

        let v = self.w_v.try_forward(v.retaped::<T>())?;
        let v = v.try_reshape_like(&(b, s2, h_dim, v_dim / h_dim))?;
        let v = v.try_permute::<_, Axes4<0, 2, 1, 3>>()?;

        let k = self.w_k.try_forward(k.retaped::<T>())?;
        let k = k.try_reshape_like(&(b, s2, h_dim, k_dim / h_dim))?;
        let k = k.try_permute::<_, Axes4<0, 2, 3, 1>>()?;

        let q = self.w_q.try_forward(q)?;
        let q = q.try_reshape_like(&(b, s1, h_dim, k_dim / h_dim))?;
        let q = q.try_permute::<_, Axes4<0, 2, 1, 3>>()?;

        // Get weights
        let scalar = 1.0 / ((k_dim / h_dim) as f64).sqrt();
        let weights = q.try_matmul(k)?.try_mul(scalar)?;
        let weights = weights.try_softmax::<Axis<3>>()?;

        // Get new tokens
        let tokens = weights.try_matmul(v)?;
        let tokens = tokens.try_permute::<_, Axes4<0, 2, 1, 3>>()?;
        let tokens = tokens.try_reshape_like(&(b, s1, self.v_dim))?;

        self.w_o.try_forward(tokens)
    }
}

impl<M: Dim, H: Dim, K: Dim, V: Dim, E, D, Src> Module<Src> for MultiHeadAttention<M, H, K, V, E, D>
where
    E: Dtype,
    D: Device<E>,
    Src: SplitTape,
    Self: Module<(Src, Src::NoTape, Src::NoTape), Output = Src>,
{
    type Output = Src;

    fn try_forward(&self, src: Src) -> Result<Self::Output, crate::tensor::Error> {
        let (src, tape) = src.split_tape();
        self.try_forward((src.clone().put_tape(tape), src.clone(), src))
    }
}

#[cfg(test)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    #[test]
    fn test_mha_unbatched() {
        let dev = TestDevice::seed_from_u64(0);

        const M: usize = 8;
        const NUM_HEADS: usize = 2;
        const S1: usize = 3;
        const S2: usize = 4;

        type Dtype = f32;

        let mha = dev.build_module::<Dtype>(
            <MultiHeadAttentionConfig<Const<M>, Const<NUM_HEADS>>>::default(),
        );

        let q: Tensor<Rank2<S1, M>, Dtype, _> = dev.sample_normal();
        let k: Tensor<Rank2<S2, M>, Dtype, _> = dev.sample_normal();
        let v: Tensor<Rank2<S2, M>, Dtype, _> = dev.sample_normal();

        let y = mha.forward((q, k, v));

        // This expected y was generated by:
        // 1. saving `mha` parameters, `q`, `k`, `v` to a file
        // 2. Running pytorch with the same values
        // 3. printing out the output
        // See https://github.com/coreylowman/dfdx/wiki/Exporting-MultiHeadAttention-to-pytorch-for-unit-tests
        #[rustfmt::skip]
        assert_close_to_literal!(
            y,
            [
                [-0.41689563,-0.46807843,-0.10825230,-0.05752429, 0.18448383,-0.56645262,-0.03250163,-0.17918219],
                [-0.26238847,-0.43888292,-0.09987387,-0.03572154, 0.11067177,-0.48738408, 0.03990822,-0.34435043],
                [-0.42392737,-0.53368354,-0.11196917,-0.18274316, 0.23356661,-0.71953803,-0.11075453,-0.10825039],
            ]
        );
    }

    #[test]
    fn test_mha_batched() {
        let dev = TestDevice::seed_from_u64(1);

        const BATCH: usize = 2;
        const M: usize = 4;
        const NUM_HEADS: usize = 2;
        const S1: usize = 3;
        const S2: usize = 2;

        type Dtype = f32;

        let mha = dev.build_module::<Dtype>(
            <MultiHeadAttentionConfig<Const<M>, Const<NUM_HEADS>>>::default(),
        );

        let q: Tensor<Rank3<BATCH, S1, M>, Dtype, _> = dev.sample_normal();
        let k: Tensor<Rank3<BATCH, S2, M>, Dtype, _> = dev.sample_normal();
        let v: Tensor<Rank3<BATCH, S2, M>, Dtype, _> = dev.sample_normal();

        // uncomment to save for this specific test params and inputs
        //
        // mha.save_safetensors("mha.safetensor").unwrap();
        // q.save_safetensors("q.safetensor").unwrap();
        // k.save_safetensors("k.safetensor").unwrap();
        // v.save_safetensors("v.safetensor").unwrap();

        let y = mha.forward((q.clone(), k.clone(), v.clone()));

        // uncomment to save for this specific test params and inputs
        //
        // y.save_safetensors("y.safetensor").unwrap();

        // This expected y was generated by:
        // 1. saving `mha` parameters, `q`, `k`, `v` to a file
        // 2. Running pytorch with the same values
        // 3. printing out the output
        // See https://github.com/coreylowman/dfdx/wiki/Exporting-MultiHeadAttention-to-pytorch-for-unit-tests
        assert_close_to_literal!(
            y,
            [
                [
                    [-0.16630043, 0.01757687, 0.22978050, 0.50355506],
                    [-0.19439587, 0.02942148, 0.23266082, 0.48612449],
                    [-0.19675586, 0.06542480, 0.18101424, 0.43833256]
                ],
                [
                    [-0.23499183, -0.21414454, 0.32811928, 0.46780989],
                    [-0.25318044, -0.20085460, 0.37180322, 0.52941465],
                    [-0.22117066, -0.23581570, 0.36783585, 0.53560883]
                ]
            ]
        );
    }

    #[test]
    fn test_backward_updates_all() {
        let dev: TestDevice = Default::default();

        let mut mha = dev
            .build_module::<TestDtype>(<MultiHeadAttentionConfig<Const<4>, Const<2>>>::default());

        let q: Tensor<Rank3<2, 3, 4>, TestDtype, _> = dev.sample_normal();
        let k: Tensor<Rank3<2, 2, 4>, TestDtype, _> = dev.sample_normal();
        let v: Tensor<Rank3<2, 2, 4>, TestDtype, _> = dev.sample_normal();
        let y = mha.forward((q.leaky_trace(), k, v));
        let g = y.square().mean().backward();

        let mut opt = crate::nn::optim::Sgd::new(&mha, Default::default());
        opt.update(&mut mha, &g).expect("");
    }
}
