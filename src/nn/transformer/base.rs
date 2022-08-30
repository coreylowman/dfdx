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
    const M: usize,
    const N: usize,
    const K: usize,
    const V: usize,
    const H: usize,
> {
    pub w_q: Linear<M, K>,
    pub w_k: Linear<N, K>,
    pub w_v: Linear<N, V>,
    pub w_o: Linear<V, M>,
}

impl<const M: usize, const N: usize, const K: usize, const V: usize, const H: usize> ResetParams
    for MultiHeadAttention<M, N, K, V, H>
{
    fn reset_params<R: Rng>(&mut self, rng: &mut R) {
        self.w_q.reset_params(rng);
        self.w_k.reset_params(rng);
        self.w_v.reset_params(rng);
        self.w_o.reset_params(rng);
    }
}

impl<const M: usize, const N: usize, const K: usize, const V: usize, const H: usize>
    CanUpdateWithGradients for MultiHeadAttention<M, N, K, V, H>
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
        const K: usize,
        const V: usize,
        const S: usize,
        const H: usize,
        T: 'static + Tape,
    > Module<Tensor2D<S, M, T>> for MultiHeadAttention<M, M, K, V, H>
where
    Assert<{ S * K == S * H * (K / H) }>: ConstTrue,
    Assert<{ S * V == S * H * (V / H) }>: ConstTrue,
    Assert<{ S * H * (V / H) == S * V }>: ConstTrue,
{
    type Output = Tensor2D<S, M, T>;

    /// Normal self attention (where same tensors are used for keys, queries and values)
    fn forward(&self, input: Tensor2D<S, M, T>) -> Self::Output {
        let (input, tape) = input.split_tape();

        let values: Tensor2D<S, V, _> = self.w_v.forward(input.duplicate().put_tape(tape));
        let values: Tensor3D<S, H, { V / H }, _> = values.reshape();
        let values: Tensor3D<H, S, { V / H }, _> = values.permute_axes::<1, 0, 2>();
        let (values, tape) = values.split_tape();

        let keys: Tensor2D<S, K, _> = self.w_k.forward(input.duplicate().put_tape(tape));
        let keys: Tensor3D<S, H, { K / H }, _> = keys.reshape();
        let keys: Tensor3D<H, S, { K / H }, _> = keys.permute_axes::<1, 0, 2>();
        let (keys, tape) = keys.split_tape();

        let queries: Tensor2D<S, K, _> = self.w_q.forward(input.put_tape(tape));
        let queries: Tensor3D<S, H, { K / H }, _> = queries.reshape();
        let queries: Tensor3D<H, S, { K / H }, _> = queries.permute_axes::<1, 0, 2>();

        // Get weights
        let token_weights: Tensor3D<H, S, S, _> = matmul_transpose(queries, &keys) / (M as f32);

        // Softmax on last dimension
        let token_weights: Tensor3D<H, S, S, _> = softmax(token_weights);

        // Get new tokens
        let tokens: Tensor3D<H, S, { V / H }, _> = matmul(token_weights, &values);
        let tokens: Tensor3D<S, H, { V / H }, _> = tokens.permute_axes::<1, 0, 2>();
        let tokens: Tensor2D<S, V, _> = tokens.reshape();

        self.w_o.forward(tokens)
    }
}

impl<
        const M: usize,
        const N: usize,
        const K: usize,
        const V: usize,
        const S1: usize,
        const S2: usize,
        const H: usize,
        T: 'static + Tape,
    > Module<(Tensor2D<S1, M, T>, Tensor2D<S2, N>)> for MultiHeadAttention<M, N, K, V, H>
where
    Assert<{ S1 * K == S1 * H * (K / H) }>: ConstTrue,
    Assert<{ S2 * K == S2 * H * (K / H) }>: ConstTrue,
    Assert<{ S2 * V == S2 * H * (V / H) }>: ConstTrue,
    Assert<{ S1 * H * (V / H) == S1 * V }>: ConstTrue,
{
    type Output = Tensor2D<S1, M, T>;

    /// Encoder-Decoder style self attention where one set of tensors is used for values and keys, and another is used for queries
    fn forward(&self, (input, from_enc): (Tensor2D<S1, M, T>, Tensor2D<S2, N>)) -> Self::Output {
        let (input, tape) = input.split_tape();

        let values: Tensor2D<S2, V, T> = self.w_v.forward(from_enc.duplicate().put_tape(tape));
        let values: Tensor3D<S2, H, { V / H }, _> = values.reshape();
        let values: Tensor3D<H, S2, { V / H }, _> = values.permute_axes::<1, 0, 2>();
        let (values, tape) = values.split_tape();

        let keys: Tensor2D<S2, K, _> = self.w_k.forward(from_enc.put_tape(tape));
        let keys: Tensor3D<S2, H, { K / H }, _> = keys.reshape();
        let keys: Tensor3D<H, S2, { K / H }, _> = keys.permute_axes::<1, 0, 2>();
        let (keys, tape) = keys.split_tape();

        let queries: Tensor2D<S1, K, _> = self.w_q.forward(input.put_tape(tape));
        let queries: Tensor3D<S1, H, { K / H }, _> = queries.reshape();
        let queries: Tensor3D<H, S1, { K / H }, _> = queries.permute_axes::<1, 0, 2>();

        // Get weights
        let token_weights: Tensor3D<H, S1, S2, _> = matmul_transpose(queries, &keys) / (M as f32);

        // Softmax on last dimension
        let token_weights: Tensor3D<H, S1, S2, _> = softmax(token_weights);

        // Get new tokens
        let tokens: Tensor3D<H, S1, { V / H }, _> = matmul(token_weights, &values);
        let tokens: Tensor3D<S1, H, { V / H }, _> = tokens.permute_axes::<1, 0, 2>();
        let tokens: Tensor2D<S1, V, T> = tokens.reshape();

        self.w_o.forward(tokens)
    }
}

impl<
        const B: usize,
        const M: usize,
        const K: usize,
        const V: usize,
        const S: usize,
        const H: usize,
        T: 'static + Tape,
    > Module<Tensor3D<B, S, M, T>> for MultiHeadAttention<M, M, K, V, H>
where
    Assert<{ B * S * K == B * S * H * (K / H) }>: ConstTrue,
    Assert<{ B * S * V == B * S * H * (V / H) }>: ConstTrue,
    Assert<{ B * S * H * (V / H) == B * S * V }>: ConstTrue,
{
    type Output = Tensor3D<B, S, M, T>;

    /// Batched normal self attention (where same tensors are used for keys, queries and values)
    fn forward(&self, input: Tensor3D<B, S, M, T>) -> Self::Output {
        let (input, tape) = input.split_tape();

        let values: Tensor3D<B, S, V, _> = self.w_v.forward(input.duplicate().put_tape(tape));
        let values: Tensor4D<B, S, H, { V / H }, _> = values.reshape();
        let values: Tensor4D<B, H, S, { V / H }, _> = values.permute_axes::<0, 2, 1, 3>();
        let (values, tape) = values.split_tape();

        let keys: Tensor3D<B, S, K, _> = self.w_k.forward(input.duplicate().put_tape(tape));
        let keys: Tensor4D<B, S, H, { K / H }, _> = keys.reshape();
        let keys: Tensor4D<B, H, S, { K / H }, _> = keys.permute_axes::<0, 2, 1, 3>();
        let (keys, tape) = keys.split_tape();

        let queries: Tensor3D<B, S, K, _> = self.w_q.forward(input.put_tape(tape));
        let queries: Tensor4D<B, S, H, { K / H }, _> = queries.reshape();
        let queries: Tensor4D<B, H, S, { K / H }, _> = queries.permute_axes::<0, 2, 1, 3>();

        // Get weights
        let token_weights: Tensor4D<B, H, S, S, _> = matmul_transpose(queries, &keys) / (M as f32);

        // Softmax on last dimension
        let token_weights: Tensor4D<B, H, S, S, _> = softmax(token_weights);

        // Get new tokens
        let tokens: Tensor4D<B, H, S, { V / H }, _> = matmul(token_weights, &values);
        let tokens: Tensor4D<B, S, H, { V / H }, _> = tokens.permute_axes::<0, 2, 1, 3>();
        let tokens: Tensor3D<B, S, V, _> = tokens.reshape();

        self.w_o.forward(tokens)
    }
}

impl<
        const B: usize,
        const M: usize,
        const N: usize,
        const K: usize,
        const V: usize,
        const S1: usize,
        const S2: usize,
        const H: usize,
        T: 'static + Tape,
    > Module<(Tensor3D<B, S1, M, T>, Tensor3D<B, S2, N>)> for MultiHeadAttention<M, N, K, V, H>
where
    Assert<{ B * S1 * K == B * S1 * H * (K / H) }>: ConstTrue,
    Assert<{ B * S2 * K == B * S2 * H * (K / H) }>: ConstTrue,
    Assert<{ B * S2 * V == B * S2 * H * (V / H) }>: ConstTrue,
    Assert<{ B * S1 * H * (V / H) == B * S1 * V }>: ConstTrue,
{
    type Output = Tensor3D<B, S1, M, T>;

    /// Batched Encoder-Decoder style self attention where one set of tensors is used for values and keys, and another is used for queries
    fn forward(
        &self,
        (input, from_enc): (Tensor3D<B, S1, M, T>, Tensor3D<B, S2, N>),
    ) -> Self::Output {
        let (input, tape) = input.split_tape();

        let values: Tensor3D<B, S2, V, _> = self.w_v.forward(from_enc.duplicate().put_tape(tape));
        let values: Tensor4D<B, S2, H, { V / H }, _> = values.reshape();
        let values: Tensor4D<B, H, S2, { V / H }, _> = values.permute_axes::<0, 2, 1, 3>();
        let (values, tape) = values.split_tape();

        let keys: Tensor3D<B, S2, K, _> = self.w_k.forward(from_enc.put_tape(tape));
        let keys: Tensor4D<B, S2, H, { K / H }, _> = keys.reshape();
        let keys: Tensor4D<B, H, S2, { K / H }, _> = keys.permute_axes::<0, 2, 1, 3>();
        let (keys, tape) = keys.split_tape();

        let queries: Tensor3D<B, S1, K, _> = self.w_q.forward(input.put_tape(tape));
        let queries: Tensor4D<B, S1, H, { K / H }, _> = queries.reshape();
        let queries: Tensor4D<B, H, S1, { K / H }, _> = queries.permute_axes::<0, 2, 1, 3>();

        // Get weights
        let token_weights: Tensor4D<B, H, S1, S2, _> =
            matmul_transpose(queries, &keys) / (M as f32);

        // Softmax on last dimension
        let token_weights: Tensor4D<B, H, S1, S2, _> = softmax(token_weights);

        // Get new tokens
        let tokens: Tensor4D<B, H, S1, { V / H }, _> = matmul(token_weights, &values);
        let tokens: Tensor4D<B, S1, H, { V / H }, _> = tokens.permute_axes::<0, 2, 1, 3>();
        let tokens: Tensor3D<B, S1, V, _> = tokens.reshape();

        self.w_o.forward(tokens)
    }
}
