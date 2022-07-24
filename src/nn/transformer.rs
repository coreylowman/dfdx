use crate::prelude::*;
use rand::Rng;

#[derive(Debug, Clone, Default)]
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
/// TODO: Doctests
#[derive(Debug, Clone, Default)]
pub struct MultiHeadAttention<const M: usize, const N: usize, const K: usize, const V: usize, const H: usize> {
    w_q: Tensor2D<N, K>,
    w_k: Tensor2D<M, K>,
    w_v: Tensor2D<M, V>,
    w_o: Tensor2D<V, M>,
}

impl<const M: usize, const N: usize, const K: usize, const V: usize, const H: usize> ResetParams for MultiHeadAttention<M, N, K, V, H> {
    fn reset_params<R: Rng>(&mut self, rng: &mut R) {
        let bound: f32 = 1.0 / (N as f32).sqrt();
        let dist = rand_distr::Uniform::new(-bound, bound);
        self.w_q.randomize(rng, &dist);
        self.w_k.randomize(rng, &dist);
        self.w_v.randomize(rng, &dist);
        self.w_o.randomize(rng, &dist);
    }
}

impl<const M: usize, const N: usize, const K: usize, const V: usize, const H: usize> CanUpdateWithGradients
    for MultiHeadAttention<M, N, K, V, H>
{
    fn update<G: GradientProvider>(&mut self, grads: &mut G) {
        self.w_q.update(grads);
        self.w_k.update(grads);
        self.w_v.update(grads);
        self.w_o.update(grads);
    }
}

/// Normal self attention (where same tensors are used for keys, queries and values)
impl<const M: usize, const K: usize, const V: usize, const S: usize, const H: usize> Module<Tensor2D<S, M>>
    for MultiHeadAttention<M, M, K, V, H>
    where 
    Assert<{S * K == H * S * (K / H)}>: ConstTrue,
    Assert<{S * V == H * S * (V / H)}>: ConstTrue,
    Assert<{H * S * (V / H) == S * V}>: ConstTrue,
{
    type Output = Tensor2D<S, M>;

    fn forward(&self, input: Tensor2D<S, M>) -> Self::Output {
        let queries = matmul(input.duplicate(), &self.w_q);
        let keys = matmul(input.duplicate(), &self.w_k);
        let values = matmul(input, &self.w_v);

        let keys: Tensor3D<H, S, {K / H}> = keys.reshape();
        let queries: Tensor3D<H, S, {K / H}> = queries.reshape();
        let values: Tensor3D<H, S, {V / H}> = values.reshape();

        // Get weights
        let token_weights = batch_matmul_transpose(queries, &keys) / (M as f32);

        // Softmax on last dimension
        let token_weights = softmax(token_weights);

        // Get new tokens
        let tokens: Tensor2D<S, V> = batch_matmul(token_weights, &values).reshape();
        matmul(tokens, &self.w_o)
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
    type Output = Tensor2D<S2, M>;

    fn forward(&self, (from_enc, input): (Tensor2D<S1, M>, Tensor2D<S2, N>)) -> Self::Output {
        let queries = matmul(input, &self.w_q);
        let keys = matmul(from_enc.duplicate(), &self.w_k);
        let values = matmul(from_enc, &self.w_v);

        let keys: Tensor3D<H, S1, {K / H}> = keys.reshape();
        let queries: Tensor3D<H, S2, {K / H}> = queries.reshape();
        let values: Tensor3D<H, S1, {V / H}> = values.reshape();

        // Get weights
        let token_weights = batch_matmul_transpose(queries, &keys) / (M as f32);

        // Softmax on last dimension
        let token_weights = softmax(token_weights);

        // Get new tokens
        let tokens: Tensor2D<S2, V> = batch_matmul(token_weights, &values).reshape();
        matmul(tokens, &self.w_o)
    }
}

#[derive(Debug, Clone, Default)]
pub struct TransformerBlock<const M: usize, const I: usize, const K: usize, const H: usize> {
    attn: MultiHeadAttention<M, M, K, M, H>,
    norm1: LayerNorm1D<M>,
    norm2: LayerNorm1D<M>,
    ff: (Linear<M, I>, ReLU, Linear<I, M>),
}

impl<const M: usize, const I: usize, const K: usize, const H: usize> ResetParams
    for TransformerBlock<M, I, K, H>
{
    fn reset_params<R: Rng>(&mut self, rng: &mut R) {
        self.attn.reset_params(rng);
        self.norm1.reset_params(rng);
        self.norm2.reset_params(rng);
        self.ff.reset_params(rng);
    }
}

impl<const M: usize, const I: usize, const K: usize, const H: usize> CanUpdateWithGradients
    for TransformerBlock<M, I, K, H>
{
    fn update<G: GradientProvider>(&mut self, grads: &mut G) {
        self.attn.update(grads);
        self.norm1.update(grads);
        self.norm2.update(grads);
        self.ff.update(grads);
    }
}

/// Single sequence impl
impl<const M: usize, const I: usize, const K: usize, const S: usize, const H: usize> Module<Tensor2D<S, M>> for TransformerBlock<M, I, K, H> 
where Assert<{S * K == H * S * (K / H)}>: ConstTrue,
Assert<{S * M == H * S * (M / H)}>: ConstTrue,
Assert<{H * S * (M / H) == S * M}>: ConstTrue,
 {
    type Output = Tensor2D<S, M>;

    fn forward(&self, input: Tensor2D<S, M>) -> Self::Output {
        let x = self.norm1.forward(input.duplicate() + &self.attn.forward(input));
        self.norm2.forward(x.duplicate() + &self.ff.forward(x))
    }
}

// /// Batch sequence impl
// impl<const M: usize, const I: usize, const K: usize, const S: usize, const B: usize, const H: usize> Module<Tensor3D<B, S, M>> for TransformerBlock<M, I, K, H> {
//     type Output = Tensor3D<B, S, M>;

//     fn forward(&self, input: Tensor3D<B, S, M>) -> Self::Output {
//         let x = self.norm1.forward(input.duplicate() + &self.attn.forward(input));
//         self.norm2.forward(x.duplicate() + &self.ff.forward(x))
//     }
// }

/// A transformer encoder.
///
/// # Generics
/// - `M` The embedding size of token vectors.
/// - `I` The inner size of the feedforward layers.
/// - `L` The number of layers.
/// - `H` The number of heads for self attention.
/// TODO: Doctests
#[derive(Debug, Clone)]
pub struct TransformerEncoder<const M: usize, const I: usize, const L: usize, const H: usize>
where Assert<{M % H == 0}>: ConstTrue {
    blocks: Repeated<
        TransformerBlock<M, I, M, H>,
        L,
    >,
}

impl <const M: usize, const I: usize, const L: usize, const H: usize>Default for TransformerEncoder<M, I, L, H>
where Assert<{M % H == 0}>: ConstTrue, 
[TransformerBlock<M, I, M, H>; L]: Default {
    fn default() -> Self {
        Self { blocks: Default::default() }
    }
}

impl<const M: usize, const I: usize, const L: usize, const H: usize> ResetParams
    for TransformerEncoder<M, I, L, H>
    where Assert<{M % H == 0}>: ConstTrue
{
    fn reset_params<R: Rng>(&mut self, rng: &mut R) {
        self.blocks.reset_params(rng);
    }
}

impl<const M: usize, const I: usize, const L: usize, const H: usize> CanUpdateWithGradients
    for TransformerEncoder<M, I, L, H>
    where Assert<{M % H == 0}>: ConstTrue
{
    fn update<G: GradientProvider>(&mut self, grads: &mut G) {
        self.blocks.update(grads);
    }
}

impl<const S: usize, const M: usize, const I: usize, const L: usize, const H: usize>
    Module<Tensor2D<S, M>> for TransformerEncoder<M, I, L, H>
where
Assert<{M % H == 0}>: ConstTrue,
Assert<{S * M == H * S * (M / H)}>: ConstTrue,
Assert<{H * S * (M / H) == S * M}>: ConstTrue,
{
    type Output = Tensor2D<S, M>;

    fn forward(&self, input: Tensor2D<S, M>) -> Self::Output {
        self.blocks.forward(input)
    }
}

// impl<const B: usize, const S: usize, const M: usize, const I: usize, const K: usize, const L: usize>
//     Module<Tensor3D<B, S, M>> for TransformerEncoder<M, I, K, L>
// {
//     type Output = Tensor3D<B, S, M>;

//     fn forward(&self, input: Tensor3D<B, S, M>) -> Self::Output {
//         self.blocks.forward(input)
//     }
// }


#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_self_attention() {
        let model: MultiHeadAttention<8, 8, 8, 8, 1> = MultiHeadAttention { 
            w_q: Tensor2D::new([[ 0.1574, -0.2003,  0.0850,  0.2589, -0.0813,  0.0932, -0.0137,  0.2020],
                [ 0.2021, -0.1780, -0.2722, -0.1615,  0.4079, -0.3185, -0.3676, -0.2339],
                [-0.4066,  0.4068, -0.0236,  0.2187,  0.0192, -0.2541, -0.3628, -0.3462],
                [-0.3576,  0.1455, -0.2628, -0.3512,  0.2617,  0.4011, -0.1893,  0.0074],
                [ 0.3362, -0.1857, -0.1462,  0.2258,  0.2525, -0.1959,  0.4204,  0.0527],
                [-0.2779,  0.2277,  0.0287, -0.3090, -0.2154, -0.3343, -0.4102,  0.1247],
                [ 0.1978, -0.0637,  0.3727, -0.1929, -0.2977,  0.0057,  0.2015, -0.3023],
                [-0.0626, -0.3986, -0.0338,  0.0366, -0.3096,  0.1367, -0.0734, -0.3320]]),
            w_k: Tensor2D::new([[ 0.2069,  0.0154, -0.2676, -0.3061, -0.2987, -0.3143, -0.3604,  0.1183],
                [-0.4073, -0.4290,  0.1581, -0.0480, -0.0837,  0.2044, -0.0503, -0.3374],
                [-0.3744,  0.1356,  0.3755, -0.4040, -0.3553,  0.2100, -0.2551,  0.0068],
                [ 0.0492, -0.0793,  0.3224,  0.0782,  0.4010, -0.2416,  0.2916, -0.1970],
                [ 0.2141, -0.4310,  0.1829,  0.1139, -0.1791, -0.0331, -0.2026, -0.0118],
                [ 0.0752,  0.2486,  0.3596,  0.2715,  0.1719,  0.3920,  0.1833, -0.0486],
                [-0.3989, -0.4021, -0.1274, -0.1533, -0.2212,  0.2649, -0.0964,  0.0363],
                [-0.2067,  0.1342,  0.4172, -0.1923,  0.3606,  0.1490, -0.1655, -0.2564]]),
            w_v: Tensor2D::new([[ 0.2284, -0.1289,  0.0660,  0.3557,  0.0571, -0.1956,  0.3716, -0.3293],
                [ 0.0483,  0.1731,  0.2582,  0.1026, -0.1180, -0.0721, -0.1970, -0.3602],
                [-0.1556,  0.0342,  0.2193, -0.2418,  0.2231, -0.0216,  0.0725, -0.2824],
                [-0.1965, -0.0953, -0.2434,  0.1300, -0.3424,  0.2907, -0.1313,  0.3331],
                [ 0.0551,  0.1247,  0.2200,  0.0062, -0.4232, -0.1389,  0.1476, -0.0718],
                [-0.0776, -0.3066, -0.0368, -0.1757, -0.0697, -0.2670,  0.1791,  0.2097],
                [-0.0299,  0.3960,  0.1764,  0.0571,  0.2683, -0.3625,  0.2716, -0.1853],
                [-0.3581, -0.1497, -0.2204, -0.1340, -0.0511,  0.2451, -0.1244,  0.1805]]),
            w_o: Tensor2D::new([[ 0.1190, -0.2099,  0.1869, -0.3508,  0.0826, -0.3263,  0.2366, -0.2100],
                [ 0.2002, -0.2365, -0.1015,  0.2539, -0.1125,  0.2926, -0.0981, -0.1495],
                [-0.1831,  0.0348,  0.2623,  0.1650,  0.2114, -0.0376,  0.1850, -0.3326],
                [-0.0636,  0.1737, -0.1024, -0.0246,  0.2178, -0.3127, -0.0506,  0.0568],
                [-0.3384, -0.1202, -0.2316,  0.0117,  0.2929, -0.2060,  0.1966, -0.3274],
                [ 0.2589,  0.3003, -0.2277,  0.2488, -0.0594, -0.0645,  0.0931,  0.2376],
                [ 0.3371,  0.0463, -0.1292,  0.1341,  0.2008, -0.0325,  0.0914,  0.0517],
                [-0.2241,  0.0426,  0.2326, -0.3048, -0.2760, -0.0868, -0.2429,  0.1446]])
        };
        let x: Tensor2D<2, 8> = Tensor2D::new([
            [0.7207, 0.3572, 0.2341, 0.4865, 0.2949, 0.5450, 0.8236, 0.4674],
            [0.4800, 0.6774, 0.9052, 0.4714, 0.5683, 0.7339, 0.1975, 0.3909]
        ]); // Sequence of 2 token vectors with 8 dims each
        let _y: Tensor2D<2, 8> = model.forward(x);
    }

    #[test]
    fn test_transformer_encoder() {
        let model: TransformerEncoder<8, 16, 1, 2> = TransformerEncoder {
            blocks: Repeated {
                modules: [
                    TransformerBlock {
                        attn: MultiHeadAttention { 
                            w_q: Tensor2D::default(),
                            w_k: Tensor2D::default(),
                            w_v: Tensor2D::default(),
                            w_o: Tensor2D::default()
                        },
                        norm1: LayerNorm1D { gamma: Tensor1D::default(), beta: Tensor1D::default(), epsilon: 1e-5 },
                        norm2: LayerNorm1D::default(),
                        ff: (Linear {
                            weight: Tensor2D::default(),
                            bias: Tensor1D::default()
                        }, ReLU::default(), Linear {
                            weight: Tensor2D::default(),
                            bias: Tensor1D::default()
                        })
                    },
                ]
            }
         };
        let x: Tensor2D<2, 8> = Tensor2D::new([
            [0.2965, 0.7154, 0.9717, 0.5441, 0.7356, 0.2681, 0.4032, 0.4670],
            [0.7770, 0.1897, 0.0112, 0.6603, 0.6334, 0.4040, 0.1425, 0.1704]
        ]);
        let _y: Tensor2D<2, 8> = model.forward(x);
        //println!("Y: {:?}", y);
        //panic!("");
    }
}