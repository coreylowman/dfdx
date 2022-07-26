use crate::prelude::*;
use rand::Rng;

/// A multi-head attention layer.
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
    fn update<G: GradientProvider>(&mut self, grads: &mut G) {
        self.w_q.update(grads);
        self.w_k.update(grads);
        self.w_v.update(grads);
        self.w_o.update(grads);
    }
}

/// Normal self attention (where same tensors are used for keys, queries and values)
impl<const M: usize, const K: usize, const V: usize, const S: usize, const H: usize, T: Tape>
    Module<Tensor2D<S, M, T>> for MultiHeadAttention<M, M, K, V, H>
where
    Assert<{ S * K == H * S * (K / H) }>: ConstTrue,
    Assert<{ S * V == H * S * (V / H) }>: ConstTrue,
    Assert<{ H * S * (V / H) == S * V }>: ConstTrue,
{
    type Output = Tensor2D<S, M, T>;

    fn forward(&self, input: Tensor2D<S, M, T>) -> Self::Output {
        let (input, tape) = input.split_tape();
        let (queries, tape) = Reshape::<Tensor3D<H, S, { K / H }, T>>::reshape(
            self.w_q.forward(input.duplicate().put_tape(tape)),
        )
        .split_tape();
        let (keys, tape) = Reshape::<Tensor3D<H, S, { K / H }, T>>::reshape(
            self.w_k.forward(input.duplicate().put_tape(tape)),
        )
        .split_tape();
        let (values, tape) = Reshape::<Tensor3D<H, S, { V / H }, T>>::reshape(
            self.w_v.forward(input.put_tape(tape)),
        )
        .split_tape();

        // Get weights
        let token_weights = batch_3d_matmul_transpose(queries.put_tape(tape), &keys) / (M as f32);

        // Softmax on last dimension
        let token_weights = softmax(token_weights);

        // Get new tokens
        let tokens: Tensor2D<S, V, T> = batch_3d_matmul(token_weights, &values).reshape();
        self.w_o.forward(tokens)
    }
}

/// Encoder-Decoder style self attention where one set of tensors is used for values and keys, and another is used for queries
impl<
        const M: usize,
        const N: usize,
        const K: usize,
        const V: usize,
        const S1: usize,
        const S2: usize,
        const H: usize,
        T: Tape,
    > Module<(Tensor2D<S1, M, T>, Tensor2D<S2, N>)> for MultiHeadAttention<M, N, K, V, H>
where
    Assert<{ S1 * K == H * S1 * (K / H) }>: ConstTrue,
    Assert<{ S2 * K == H * S2 * (K / H) }>: ConstTrue,
    Assert<{ S2 * V == H * S2 * (V / H) }>: ConstTrue,
    Assert<{ H * S1 * (V / H) == S1 * V }>: ConstTrue,
{
    type Output = Tensor2D<S1, M, T>;

    fn forward(&self, (input, from_enc): (Tensor2D<S1, M, T>, Tensor2D<S2, N>)) -> Self::Output {
        let queries: Tensor3D<H, S1, { K / H }, T> = self.w_q.forward(input).reshape();
        let (queries, tape) = queries.split_tape();
        let (keys, tape) = Reshape::<Tensor3D<H, S2, { K / H }, T>>::reshape(
            self.w_k.forward(from_enc.duplicate().put_tape(tape)),
        )
        .split_tape();
        let (values, tape) = Reshape::<Tensor3D<H, S2, { V / H }, T>>::reshape(
            self.w_v.forward(from_enc.put_tape(tape)),
        )
        .split_tape();

        // Get weights
        let token_weights = batch_3d_matmul_transpose(queries.put_tape(tape), &keys) / (M as f32);

        // Softmax on last dimension
        let token_weights = softmax(token_weights);

        // Get new tokens
        let tokens: Tensor2D<S1, V, T> = batch_3d_matmul(token_weights, &values).reshape();
        self.w_o.forward(tokens)
    }
}

/// Batched normal self attention (where same tensors are used for keys, queries and values)
impl<
        const B: usize,
        const M: usize,
        const K: usize,
        const V: usize,
        const S: usize,
        const H: usize,
        T: Tape,
    > Module<Tensor3D<B, S, M, T>> for MultiHeadAttention<M, M, K, V, H>
where
    Assert<{ B * S * K == B * H * S * (K / H) }>: ConstTrue,
    Assert<{ B * S * V == B * H * S * (V / H) }>: ConstTrue,
    Assert<{ B * H * S * (V / H) == B * S * V }>: ConstTrue,
{
    type Output = Tensor3D<B, S, M, T>;

    fn forward(&self, input: Tensor3D<B, S, M, T>) -> Self::Output {
        let (input, tape) = input.split_tape();
        let (queries, tape) = Reshape::<Tensor4D<B, H, S, { K / H }, T>>::reshape(
            self.w_q.forward(input.duplicate().put_tape(tape)),
        )
        .split_tape();
        let (keys, tape) = Reshape::<Tensor4D<B, H, S, { K / H }, T>>::reshape(
            self.w_k.forward(input.duplicate().put_tape(tape)),
        )
        .split_tape();
        let (values, tape) = Reshape::<Tensor4D<B, H, S, { V / H }, T>>::reshape(
            self.w_v.forward(input.put_tape(tape)),
        )
        .split_tape();

        // Get weights
        let token_weights = batch_4d_matmul_transpose(queries.put_tape(tape), &keys) / (M as f32);

        // Softmax on last dimension
        let token_weights = softmax(token_weights);

        // Get new tokens
        let tokens: Tensor3D<B, S, V, T> = batch_4d_matmul(token_weights, &values).reshape();
        self.w_o.forward(tokens)
    }
}

/// Batched Encoder-Decoder style self attention where one set of tensors is used for values and keys, and another is used for queries
impl<
        const B: usize,
        const M: usize,
        const N: usize,
        const K: usize,
        const V: usize,
        const S1: usize,
        const S2: usize,
        const H: usize,
        T: Tape,
    > Module<(Tensor3D<B, S1, M, T>, Tensor3D<B, S2, N>)> for MultiHeadAttention<M, N, K, V, H>
where
    Assert<{ B * S1 * K == B * H * S1 * (K / H) }>: ConstTrue,
    Assert<{ B * S2 * K == B * H * S2 * (K / H) }>: ConstTrue,
    Assert<{ B * S2 * V == B * H * S2 * (V / H) }>: ConstTrue,
    Assert<{ B * H * S1 * (V / H) == B * S1 * V }>: ConstTrue,
{
    type Output = Tensor3D<B, S1, M, T>;

    fn forward(
        &self,
        (input, from_enc): (Tensor3D<B, S1, M, T>, Tensor3D<B, S2, N>),
    ) -> Self::Output {
        let queries: Tensor4D<B, H, S1, { K / H }, T> = self.w_q.forward(input).reshape();
        let (queries, tape) = queries.split_tape();
        let (keys, tape) = Reshape::<Tensor4D<B, H, S2, { K / H }, T>>::reshape(
            self.w_k.forward(from_enc.duplicate().put_tape(tape)),
        )
        .split_tape();
        let (values, tape) = Reshape::<Tensor4D<B, H, S2, { V / H }, T>>::reshape(
            self.w_v.forward(from_enc.put_tape(tape)),
        )
        .split_tape();

        // Get weights
        let token_weights = batch_4d_matmul_transpose(queries.put_tape(tape), &keys) / (M as f32);

        // Softmax on last dimension
        let token_weights = softmax(token_weights);

        // Get new tokens
        let tokens: Tensor3D<B, S1, V, T> = batch_4d_matmul(token_weights, &values).reshape();
        self.w_o.forward(tokens)
    }
}
