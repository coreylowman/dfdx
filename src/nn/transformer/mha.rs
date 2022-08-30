use crate::prelude::*;
use rand::Rng;

/// **Requires Nightly** A multi-head attention layer.
///
/// # Generics
/// - `M` The embedding size of token vectors from decoder.
/// - `N` The embedding size of token vectors from encoder.
/// - `K` The size of the keys in self attention.
/// - `V` The size of the values.
/// - `H` The number of attention heads.
///
/// # Examples
/// `MultiHeadAttention<8, 10, 10, 10, 2>` is an attention layer with 2 heads and 10 token, key and value dims.
/// TODO: Doctests fail for some reason
#[derive(Debug, Clone, Default)]
pub struct MultiHeadAttention<
    const EMBED_DIM: usize,
    const NUM_HEADS: usize,
    const K_DIM: usize = EMBED_DIM,
    const V_DIM: usize = EMBED_DIM,
> {
    pub w_q: Linear<EMBED_DIM, K_DIM>,
    pub w_k: Linear<EMBED_DIM, K_DIM>,
    pub w_v: Linear<EMBED_DIM, V_DIM>,
    pub w_o: Linear<V_DIM, EMBED_DIM>,
}

impl<const M: usize, const H: usize, const K: usize, const V: usize> ResetParams
    for MultiHeadAttention<M, H, K, V>
{
    fn reset_params<R: Rng>(&mut self, rng: &mut R) {
        self.w_q.reset_params(rng);
        self.w_k.reset_params(rng);
        self.w_v.reset_params(rng);
        self.w_o.reset_params(rng);
    }
}

impl<const M: usize, const H: usize, const K: usize, const V: usize> CanUpdateWithGradients
    for MultiHeadAttention<M, H, K, V>
{
    fn update<G: GradientProvider>(&mut self, grads: &mut G, unused: &mut UnusedTensors) {
        self.w_q.update(grads, unused);
        self.w_k.update(grads, unused);
        self.w_v.update(grads, unused);
        self.w_o.update(grads, unused);
    }
}

impl<
        const M: usize,
        const H: usize,
        const K: usize,
        const V: usize,
        const S1: usize,
        const S2: usize,
        TAPE: 'static + Tape,
    > Module<(Tensor2D<S1, M, TAPE>, Tensor2D<S2, M>, Tensor2D<S2, M>)>
    for MultiHeadAttention<M, H, K, V>
where
    Assert<{ S1 * K == S1 * H * (K / H) }>: ConstTrue,
    Assert<{ S2 * K == S2 * H * (K / H) }>: ConstTrue,
    Assert<{ S2 * V == S2 * H * (V / H) }>: ConstTrue,
    Assert<{ S1 * H * (V / H) == S1 * V }>: ConstTrue,
{
    type Output = Tensor2D<S1, M, TAPE>;

    /// Encoder-Decoder style self attention where one set of tensors is used for values and keys, and another is used for queries
    fn forward(
        &self,
        (q, k, v): (Tensor2D<S1, M, TAPE>, Tensor2D<S2, M>, Tensor2D<S2, M>),
    ) -> Self::Output {
        let (q, tape) = q.split_tape();

        let v: Tensor2D<S2, V, _> = self.w_v.forward(v.put_tape(tape));
        let v: Tensor3D<S2, H, { V / H }, _> = v.reshape();
        let v: Tensor3D<H, S2, { V / H }, _> = v.permute_axes::<1, 0, 2>();
        let (v, tape) = v.split_tape();

        let k: Tensor2D<S2, K, _> = self.w_k.forward(k.put_tape(tape));
        let k: Tensor3D<S2, H, { K / H }, _> = k.reshape();
        let k: Tensor3D<H, S2, { K / H }, _> = k.permute_axes::<1, 0, 2>();
        let (k, tape) = k.split_tape();

        let q: Tensor2D<S1, K, _> = self.w_q.forward(q.put_tape(tape));
        let q: Tensor3D<S1, H, { K / H }, _> = q.reshape();
        let q: Tensor3D<H, S1, { K / H }, _> = q.permute_axes::<1, 0, 2>();

        // Get weights
        let weights: Tensor3D<H, S1, S2, _> = matmul_transpose(q, &k) / (M as f32);

        // Softmax on last dimension
        let weights: Tensor3D<H, S1, S2, _> = softmax(weights);

        // Get new tokens
        let tokens: Tensor3D<H, S1, { V / H }, _> = matmul(weights, &v);
        let tokens: Tensor3D<S1, H, { V / H }, _> = tokens.permute_axes::<1, 0, 2>();
        let tokens: Tensor2D<S1, V, _> = tokens.reshape();

        self.w_o.forward(tokens)
    }
}

impl<
        const M: usize,
        const H: usize,
        const K: usize,
        const V: usize,
        const B: usize,
        const S1: usize,
        const S2: usize,
        TAPE: 'static + Tape,
    >
    Module<(
        Tensor3D<B, S1, M, TAPE>,
        Tensor3D<B, S2, M>,
        Tensor3D<B, S2, M>,
    )> for MultiHeadAttention<M, H, K, V>
where
    Assert<{ B * S1 * K == B * S1 * H * (K / H) }>: ConstTrue,
    Assert<{ B * S2 * K == B * S2 * H * (K / H) }>: ConstTrue,
    Assert<{ B * S2 * V == B * S2 * H * (V / H) }>: ConstTrue,
    Assert<{ B * S1 * H * (V / H) == B * S1 * V }>: ConstTrue,
{
    type Output = Tensor3D<B, S1, M, TAPE>;

    /// Batched Encoder-Decoder style self attention where one set of tensors is used for values and keys, and another is used for queries
    fn forward(
        &self,
        (q, k, v): (
            Tensor3D<B, S1, M, TAPE>,
            Tensor3D<B, S2, M>,
            Tensor3D<B, S2, M>,
        ),
    ) -> Self::Output {
        let (q, tape) = q.split_tape();

        let v: Tensor3D<B, S2, V, _> = self.w_v.forward(v.put_tape(tape));
        let v: Tensor4D<B, S2, H, { V / H }, _> = v.reshape();
        let v: Tensor4D<B, H, S2, { V / H }, _> = v.permute_axes::<0, 2, 1, 3>();
        let (v, tape) = v.split_tape();

        let k: Tensor3D<B, S2, K, _> = self.w_k.forward(k.put_tape(tape));
        let k: Tensor4D<B, S2, H, { K / H }, _> = k.reshape();
        let k: Tensor4D<B, H, S2, { K / H }, _> = k.permute_axes::<0, 2, 1, 3>();
        let (k, tape) = k.split_tape();

        let q: Tensor3D<B, S1, K, _> = self.w_q.forward(q.put_tape(tape));
        let q: Tensor4D<B, S1, H, { K / H }, _> = q.reshape();
        let q: Tensor4D<B, H, S1, { K / H }, _> = q.permute_axes::<0, 2, 1, 3>();

        // Get weights
        let weights: Tensor4D<B, H, S1, S2, _> = matmul_transpose(q, &k) / (M as f32);

        // Softmax on last dimension
        let weights: Tensor4D<B, H, S1, S2, _> = softmax(weights);

        // Get new tokens
        let tokens: Tensor4D<B, H, S1, { V / H }, _> = matmul(weights, &v);
        let tokens: Tensor4D<B, S1, H, { V / H }, _> = tokens.permute_axes::<0, 2, 1, 3>();
        let tokens: Tensor3D<B, S1, V, _> = tokens.reshape();

        self.w_o.forward(tokens)
    }
}
