use crate::prelude::*;
use rand::Rng;

#[derive(Debug, Clone)]
pub struct SingleHeadAttention<const M: usize, const N: usize, const K: usize, const V: usize> {
    w_q: Linear<N, K>,
    w_k: Linear<M, K>,
    w_v: Linear<M, V>,
}

impl<const M: usize, const N: usize, const K: usize, const V: usize> ResetParams for SingleHeadAttention<M, N, K, V> {
    fn reset_params<R: Rng>(&mut self, rng: &mut R) {
        self.w_q.reset_params(rng);
        self.w_k.reset_params(rng);
        self.w_v.reset_params(rng);
    }
}

impl<const M: usize, const N: usize, const K: usize, const V: usize> CanUpdateWithGradients
    for SingleHeadAttention<M, N, K, V>
{
    fn update<G: GradientProvider>(&mut self, grads: &mut G) {
        self.w_q.update(grads);
        self.w_k.update(grads);
        self.w_v.update(grads);
    }
}

/// Normal self attention (where same tensors are used for keys, queries and values)
impl<const M: usize, const K: usize, const V: usize, const S: usize> Module<Tensor2D<S, M>>
    for SingleHeadAttention<M, M, K, V>
{
    type Output = Tensor2D<S, V>;

    fn forward(&self, input: Tensor2D<S, M>) -> Self::Output {
        let queries = self.w_q.forward(input.duplicate());
        let keys = self.w_k.forward(input.duplicate());
        let values = self.w_v.forward(input);

        // Get weights
        let token_weights = matmul_transpose(queries, &keys) / (M as f32);

        // Softmax on last dimension
        let token_weights = softmax(token_weights);

        // Get new tokens
        matmul(token_weights, &values)
    }
}

/// Batched normal self attention (where same tensors are used for keys, queries and values)
impl<const M: usize, const K: usize, const V: usize, const S: usize, const B: usize> Module<Tensor3D<B, S, M>>
    for SingleHeadAttention<M, M, K, V>
{
    type Output = Tensor3D<B, S, V>;

    fn forward(&self, input: Tensor3D<B, S, M>) -> Self::Output {
        let queries = self.w_q.forward(input.duplicate());
        let keys = self.w_k.forward(input.duplicate());
        let values = self.w_v.forward(input);

        // Get weights
        let token_weights = batch_matmul_transpose(queries, &keys) / (M as f32);

        // Softmax on last dimension
        let token_weights = softmax(token_weights);

        // Get new tokens
        batch_matmul(token_weights, &values)
    }
}

/// Encoder-Decoder style self attention where one set of tensors is used for values and keys, and another is used for queries
impl<const M: usize, const N: usize, const K: usize, const V: usize, const S1: usize, const S2: usize> Module<(Tensor2D<S2, M>, Tensor2D<S1, N>)>
    for SingleHeadAttention<M, N, K, V>
{
    type Output = Tensor2D<S1, V>;

    fn forward(&self, (from_enc, input): (Tensor2D<S2, M>, Tensor2D<S1, N>)) -> Self::Output {
        let queries = self.w_q.forward(input);
        let keys = self.w_k.forward(from_enc.duplicate());
        let values = self.w_v.forward(from_enc);

        // Get weights
        let token_weights = matmul_transpose(queries, &keys) / (M as f32);

        // Softmax on last dimension
        let token_weights = softmax(token_weights);

        // Get new tokens
        matmul(token_weights, &values)
    }
}

/// Batched Encoder-Decoder style self attention where one set of tensors is used for values and keys, and another is used for queries
impl<const M: usize, const N: usize, const K: usize, const V: usize, const B: usize, const S1: usize, const S2: usize> Module<(Tensor3D<B, S1, M>, Tensor3D<B, S2, N>)>
    for SingleHeadAttention<M, N, K, V>
{
    type Output = Tensor3D<B, S2, V>;

    fn forward(&self, (from_enc, input): (Tensor3D<B, S1, M>, Tensor3D<B, S2, N>)) -> Self::Output {
        let queries = self.w_q.forward(input);
        let keys = self.w_k.forward(from_enc.duplicate());
        let values = self.w_v.forward(from_enc);

        // Get weights
        let token_weights = batch_matmul_transpose(queries, &keys) / (M as f32);

        // Softmax on last dimension
        let token_weights = softmax(token_weights);

        // Get new tokens
        batch_matmul(token_weights, &values)
    }
}

#[derive(Debug, Clone)]
pub struct MultiHeadAttention<const M: usize, const N: usize, const K: usize, const V: usize, const H: usize> {
    w_q: Linear<N, K>,
    w_kv: SplitInto<(Linear<M, K>, Linear<M, V>)>,
}

impl<const M: usize, const N: usize, const K: usize, const V: usize, const H: usize> ResetParams for MultiHeadAttention<M, N, K, V, H> {
    fn reset_params<R: Rng>(&mut self, rng: &mut R) {
        self.w_q.reset_params(rng);
        self.w_kv.reset_params(rng);
    }
}

impl<const M: usize, const N: usize, const K: usize, const V: usize, const H: usize> CanUpdateWithGradients
    for MultiHeadAttention<M, N, K, V, H>
{
    fn update<G: GradientProvider>(&mut self, grads: &mut G) {
        self.w_q.update(grads);
        self.w_kv.update(grads);
    }
}

/// Normal self attention (where same tensors are used for keys, queries and values)
impl<const M: usize, const K: usize, const V: usize, const S: usize, const H: usize> Module<Tensor2D<S, M>>
    for MultiHeadAttention<M, M, K, V, H>
    where 
    Assert<{S * K == H * S * (K / H)}>: ConstTrue,
    Assert<{S * V == H * S * (V / H)}>: ConstTrue,
    Assert<{H * S * {V / H} == S * V}>: ConstTrue,
{
    type Output = Tensor2D<S, V>;

    fn forward(&self, input: Tensor2D<S, M>) -> Self::Output {
        let queries = self.w_q.forward(input.duplicate());
        let (keys, values) = self.w_kv.forward(input.duplicate());

        let keys: Tensor3D<H, S, {K / H}> = keys.reshape();
        let queries: Tensor3D<H, S, {K / H}> = queries.reshape();
        let values: Tensor3D<H, S, {V / H}> = values.reshape();

        // Get weights
        let token_weights = batch_matmul_transpose(queries, &keys) / (M as f32);

        // Softmax on last dimension
        let token_weights = softmax(token_weights);

        // Get new tokens
        let tokens: Tensor3D<H, S, {V / H}> = batch_matmul(token_weights, &values);
        tokens.reshape()
    }
}

/// Encoder-Decoder style self attention where one set of tensors is used for values and keys, and another is used for queries
impl<const M: usize, const N: usize, const K: usize, const V: usize, const S1: usize, const S2: usize, const H: usize> Module<(Tensor2D<S1, M>, Tensor2D<S2, N>)>
    for MultiHeadAttention<M, N, K, V, H>
    where 
    Assert<{S2 * K == H * S2 * (K / H)}>: ConstTrue,
    Assert<{S1 * K == H * S1 * (K / H)}>: ConstTrue,
    Assert<{S1 * V == H * S1 * (V / H)}>: ConstTrue,
    Assert<{H * S2 * {V / H} == S2 * V}>: ConstTrue,
{
    type Output = Tensor2D<S2, V>;

    fn forward(&self, (from_enc, input): (Tensor2D<S1, M>, Tensor2D<S2, N>)) -> Self::Output {
        let queries = self.w_q.forward(input);
        let (keys, values) = self.w_kv.forward(from_enc);

        let keys: Tensor3D<H, S1, {K / H}> = keys.reshape();
        let queries: Tensor3D<H, S2, {K / H}> = queries.reshape();
        let values: Tensor3D<H, S1, {V / H}> = values.reshape();

        // Get weights
        let token_weights = batch_matmul_transpose(queries, &keys) / (M as f32);

        // Softmax on last dimension
        let token_weights = softmax(token_weights);

        // Get new tokens
        let tokens: Tensor3D<H, S2, {V / H}> = batch_matmul(token_weights, &values);
        tokens.reshape()
    }
}

/// Currently only uses single head attention. Reshape and concat are required for multi head attention.
/// Also multi head attention will likely require nightly for N_HEADS * HEAD_DIM
#[derive(Debug, Clone)]
pub struct TransformerBlock<const M: usize, const I: usize, const K: usize> {
    attn: SingleHeadAttention<M, M, K, M>,
    norm1: LayerNorm1D<M>,
    norm2: LayerNorm1D<M>,
    ff: (Linear<M, I>, ReLU, Linear<I, M>),
}

impl<const M: usize, const I: usize, const K: usize> ResetParams
    for TransformerBlock<M, I, K>
{
    fn reset_params<R: Rng>(&mut self, rng: &mut R) {
        self.attn.reset_params(rng);
        self.norm1.reset_params(rng);
        self.norm2.reset_params(rng);
        self.ff.reset_params(rng);
    }
}

impl<const M: usize, const I: usize, const K: usize> CanUpdateWithGradients
    for TransformerBlock<M, I, K>
{
    fn update<G: GradientProvider>(&mut self, grads: &mut G) {
        self.attn.update(grads);
        self.norm1.update(grads);
        self.norm2.update(grads);
        self.ff.update(grads);
    }
}

/// Single sequence impl
impl<const M: usize, const I: usize, const K: usize, const S: usize> Module<Tensor2D<S, M>> for TransformerBlock<M, I, K> {
    type Output = Tensor2D<S, M>;

    fn forward(&self, input: Tensor2D<S, M>) -> Self::Output {
        let x = self.norm1.forward(input.duplicate() + &self.attn.forward(input));
        self.norm2.forward(x.duplicate() + &self.ff.forward(x))
    }
}

/// Batch sequence impl
impl<const M: usize, const I: usize, const K: usize, const S: usize, const B: usize> Module<Tensor3D<B, S, M>> for TransformerBlock<M, I, K> {
    type Output = Tensor3D<B, S, M>;

    fn forward(&self, input: Tensor3D<B, S, M>) -> Self::Output {
        let x = self.norm1.forward(input.duplicate() + &self.attn.forward(input));
        self.norm2.forward(x.duplicate() + &self.ff.forward(x))
    }
}

/// Currently only uses single head attention. Reshape and concat are required for multi head attention.
/// Also multi head attention will likely require nightly for N_HEADS * HEAD_DIM
#[derive(Debug, Clone)]
pub struct TransformerEncoder<const M: usize, const I: usize, const K: usize, const L: usize> {
    blocks: Repeated<
        TransformerBlock<M, I, K>,
        L,
    >,
}

impl<const M: usize, const I: usize, const K: usize, const L: usize> ResetParams
    for TransformerEncoder<M, I, K, L>
{
    fn reset_params<R: Rng>(&mut self, rng: &mut R) {
        self.blocks.reset_params(rng);
    }
}

impl<const M: usize, const I: usize, const K: usize, const L: usize> CanUpdateWithGradients
    for TransformerEncoder<M, I, K, L>
{
    fn update<G: GradientProvider>(&mut self, grads: &mut G) {
        self.blocks.update(grads);
    }
}

impl<const S: usize, const M: usize, const I: usize, const K: usize, const L: usize>
    Module<Tensor2D<S, M>> for TransformerEncoder<M, I, K, L>
{
    type Output = Tensor2D<S, M>;

    fn forward(&self, input: Tensor2D<S, M>) -> Self::Output {
        self.blocks.forward(input)
    }
}

impl<const B: usize, const S: usize, const M: usize, const I: usize, const K: usize, const L: usize>
    Module<Tensor3D<B, S, M>> for TransformerEncoder<M, I, K, L>
{
    type Output = Tensor3D<B, S, M>;

    fn forward(&self, input: Tensor3D<B, S, M>) -> Self::Output {
        self.blocks.forward(input)
    }
}