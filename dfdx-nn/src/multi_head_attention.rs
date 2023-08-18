use crate::*;
use dfdx::{shapes::*, tensor::*, tensor_ops::*};
use num_traits::Float;

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
    dfdx_nn_core::Module<(
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
    type Error = D::Err;

    /// Encoder-Decoder style self attention where one set of tensors is used for values and keys, and another is used for queries
    fn try_forward(
        &self,
        (q, k, v): (
            Tensor<(S1, M), E, D, T>,
            Tensor<(S2, M), E, D>,
            Tensor<(S2, M), E, D>,
        ),
    ) -> Result<Self::Output, D::Err> {
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
    dfdx_nn_core::Module<(
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
    type Error = D::Err;

    /// Batched Encoder-Decoder style self attention where one set of tensors is used for values and keys, and another is used for queries
    fn try_forward(
        &self,
        (q, k, v): (
            Tensor<(B, S1, M), E, D, T>,
            Tensor<(B, S2, M), E, D>,
            Tensor<(B, S2, M), E, D>,
        ),
    ) -> Result<Self::Output, D::Err> {
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
        let scalar: E = E::from_f64(1.0 / ((k_dim / h_dim) as f64).sqrt()).unwrap();
        let weights = q.try_matmul(k)?.try_mul(scalar)?;
        let weights = weights.try_softmax::<Axis<3>>()?;

        // Get new tokens
        let tokens = weights.try_matmul(v)?;
        let tokens = tokens.try_permute::<_, Axes4<0, 2, 1, 3>>()?;
        let tokens = tokens.try_reshape_like(&(b, s1, self.v_dim))?;

        self.w_o.try_forward(tokens)
    }
}

impl<M: Dim, H: Dim, K: Dim, V: Dim, E, D, Src> dfdx_nn_core::Module<Src>
    for MultiHeadAttention<M, H, K, V, E, D>
where
    E: Dtype,
    D: Device<E>,
    Src: SplitTape,
    Self: dfdx_nn_core::Module<(Src, Src::NoTape, Src::NoTape), Output = Src, Error = D::Err>,
{
    type Output = Src;
    type Error = D::Err;

    fn try_forward(&self, src: Src) -> Result<Self::Output, D::Err> {
        let (src, tape) = src.split_tape();
        self.try_forward((src.clone().put_tape(tape), src.clone(), src))
    }
}
