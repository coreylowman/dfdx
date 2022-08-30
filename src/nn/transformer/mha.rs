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
        let scalar: f32 = 1.0 / ((K / H) as f32).sqrt();
        let weights: Tensor3D<H, S1, S2, _> = matmul_transpose(q, &k) * scalar;

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
        let scalar: f32 = 1.0 / ((K / H) as f32).sqrt();
        let weights: Tensor4D<B, H, S1, S2, _> = matmul_transpose(q, &k) * scalar;

        // Softmax on last dimension
        let weights: Tensor4D<B, H, S1, S2, _> = softmax(weights);

        // Get new tokens
        let tokens: Tensor4D<B, H, S1, { V / H }, _> = matmul(weights, &v);
        let tokens: Tensor4D<B, S1, H, { V / H }, _> = tokens.permute_axes::<0, 2, 1, 3>();
        let tokens: Tensor3D<B, S1, V, _> = tokens.reshape();

        let o = self.w_o.forward(tokens);
        println!("{:?}", self.w_o.weight.data());
        println!("{:?}", self.w_o.bias.data());
        // println!("{:?}", o.data());
        o
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::assert_close;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn test_mha_unbatched() {
        let mut rng = StdRng::seed_from_u64(0);

        const EMBED_DIM: usize = 8;
        const NUM_HEADS: usize = 2;
        const SEQ_LEN: usize = 3;

        let mut mha: MultiHeadAttention<EMBED_DIM, NUM_HEADS> = Default::default();
        mha.reset_params(&mut rng);

        let q: Tensor2D<SEQ_LEN, EMBED_DIM> = TensorCreator::randn(&mut rng);
        let k: Tensor2D<SEQ_LEN, EMBED_DIM> = TensorCreator::randn(&mut rng);
        let v: Tensor2D<SEQ_LEN, EMBED_DIM> = TensorCreator::randn(&mut rng);

        let y: Tensor2D<SEQ_LEN, EMBED_DIM> = mha.forward((q, k, v));

        // This expected y was generated by:
        // 1. saving `mha` parameters, `q`, `k`, `v` to a file
        // 2. Running pytorch with the same values
        // 3. printing out the output
        // See https://github.com/coreylowman/dfdx/wiki/Exporting-MultiHeadAttention-to-pytorch-for-unit-tests
        #[rustfmt::skip]
        assert_close(
            y.data(),
            &[
                [-0.58498609,-0.33808267,-0.14341073, 0.27620849, 0.37429583,-0.21466538,-0.13287979,-0.04122865],
                [-0.61057514,-0.36417598,-0.15251875, 0.18421736, 0.32708097,-0.35524112,-0.12564486, 0.00822827],
                [-0.64026815,-0.38109598,-0.13273168, 0.24654171, 0.40275624,-0.26089215,-0.11465970,-0.04107106],
            ],
        );
    }

    #[test]
    fn test_mha_batched() {
        let mut rng = StdRng::seed_from_u64(1);

        const BATCH: usize = 5;
        const EMBED_DIM: usize = 8;
        const NUM_HEADS: usize = 2;
        const SEQ_LEN: usize = 3;

        let mut mha: MultiHeadAttention<EMBED_DIM, NUM_HEADS> = Default::default();
        mha.reset_params(&mut rng);

        let q: Tensor3D<BATCH, SEQ_LEN, EMBED_DIM> = TensorCreator::randn(&mut rng);
        let k: Tensor3D<BATCH, SEQ_LEN, EMBED_DIM> = TensorCreator::randn(&mut rng);
        let v: Tensor3D<BATCH, SEQ_LEN, EMBED_DIM> = TensorCreator::randn(&mut rng);

        let y: Tensor3D<BATCH, SEQ_LEN, EMBED_DIM> = mha.forward((q, k, v));

        // This expected y was generated by:
        // 1. saving `mha` parameters, `q`, `k`, `v` to a file
        // 2. Running pytorch with the same values
        // 3. printing out the output
        // See https://github.com/coreylowman/dfdx/wiki/Exporting-MultiHeadAttention-to-pytorch-for-unit-tests
        #[rustfmt::skip]
        assert_close(
            y.data(),
            &[
                [
                    [-0.54876709, 0.16964084, 0.49195337,-0.33731836, 0.33880740,-0.06813130, 0.27716312, 0.16173348],
                    [-0.53962874, 0.18113990, 0.48955891,-0.32123145, 0.34569934,-0.05345859, 0.27961808, 0.18161729],
                    [-0.59251058, 0.14441319, 0.50615567,-0.39405537, 0.32771301,-0.10393527, 0.27619639, 0.13343287],
                ],
                [
                    [-0.14347501, 0.33244792, 0.19656339,-0.25272137, 0.15943763,-0.20900822, 0.42001662, 0.26504010],
                    [-0.29173288, 0.39812529, 0.30828360,-0.12839445, 0.33921015, 0.08642961, 0.39518371, 0.43284720],
                    [-0.18362300, 0.36481178, 0.25245655,-0.16965692, 0.22731936,-0.05351603, 0.42168203, 0.39724022],
                ],
                [
                    [-0.30136436, 0.14744721, 0.28378472,-0.48647106, 0.14806966,-0.44610590, 0.45483866,-0.03253588],
                    [-0.42411470, 0.07871948, 0.32910275,-0.63309073, 0.12007453,-0.53931695, 0.46482331,-0.15187648],
                    [-0.34929872, 0.09329192, 0.30408362,-0.56438422, 0.13897458,-0.50592148, 0.45566535,-0.07735601],
                ],
                [
                    [-0.07390994, 0.14987674, 0.30987290,-0.19632836, 0.08741264,-0.22045007, 0.46401465, 0.25927910],
                    [-0.07530174, 0.15076782, 0.30932930,-0.19645353, 0.08872256,-0.21871342, 0.46238160, 0.26088321],
                    [-0.06678163, 0.12455779, 0.31057101,-0.22205283, 0.06943876,-0.26624525, 0.46386302, 0.23858917],
                ],
                [
                    [-0.12307644,-0.03581740, 0.26402885,-0.51472521,-0.10128927,-0.58983749, 0.49667436,-0.12181333],
                    [-0.05412547, 0.00873812, 0.20380302,-0.51770800,-0.10924360,-0.63563013, 0.52258074,-0.05686778],
                    [-0.12557350, 0.09155779, 0.27587649,-0.38954079, 0.04291603,-0.45441478, 0.52418828, 0.02906075],
                ],
            ],
        );
    }
}
