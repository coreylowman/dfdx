use crate::prelude::*;
use rand::Rng;

/// **Requires Nightly** A multi-head attention layer.
///
/// Generics:
/// - `EMBED_DIM`: The size of query vectors.
/// - `NUM_HEADS` The number of heads to split query/key/value into.
/// - *Optional* `K_DIM`: The size of key vectors. Defaults to `EMBED_DIM`
/// - *Optional* `V_DIM` The size of value vectors. Defaults to `EMBED_DIM`
///
/// **Pytorch equivalent**: `torch.nn.MultiheadAttention(EMBED_DIM, NUM_HEADS, batch_first=True)`
///
/// Examples
/// - `MultiHeadAttention<8, 2>` is an attention layer with 2 heads and 8 token, key and value dims.
/// - `MultiHeadAttention<8, 2, 6, 4>` is an attention layer with the key and value dimension different
///   than the embed dimension
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

impl<const M: usize, const H: usize, const K: usize, const V: usize> SaveToNpz
    for MultiHeadAttention<M, H, K, V>
{
    fn write<W>(&self, pre: &str, w: &mut zip::ZipWriter<W>) -> zip::result::ZipResult<()>
    where
        W: std::io::Write + std::io::Seek,
    {
        self.w_q.write(&format!("{pre}w_q."), w)?;
        self.w_k.write(&format!("{pre}w_k."), w)?;
        self.w_v.write(&format!("{pre}w_v."), w)?;
        self.w_o.write(&format!("{pre}w_o."), w)?;
        Ok(())
    }
}

impl<const M: usize, const H: usize, const K: usize, const V: usize> LoadFromNpz
    for MultiHeadAttention<M, H, K, V>
{
    fn read<R>(&mut self, pre: &str, r: &mut zip::ZipArchive<R>) -> Result<(), NpzError>
    where
        R: std::io::Read + std::io::Seek,
    {
        self.w_q.read(&format!("{pre}w_q."), r)?;
        self.w_k.read(&format!("{pre}w_k."), r)?;
        self.w_v.read(&format!("{pre}w_v."), r)?;
        self.w_o.read(&format!("{pre}w_o."), r)?;
        Ok(())
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

        self.w_o.forward(tokens)
    }
}

impl<T, const M: usize, const H: usize, const K: usize, const V: usize> ModuleMut<T>
    for MultiHeadAttention<M, H, K, V>
where
    Self: Module<T>,
{
    type Output = <Self as Module<T>>::Output;

    fn forward_mut(&mut self, input: T) -> Self::Output {
        self.forward(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{nn::tests::SimpleGradients, tests::assert_close};
    use rand::{rngs::StdRng, thread_rng, SeedableRng};
    use tempfile::NamedTempFile;

    #[test]
    fn test_mha_unbatched() {
        let mut rng = StdRng::seed_from_u64(0);

        const EMBED_DIM: usize = 8;
        const NUM_HEADS: usize = 2;
        const S1: usize = 3;
        const S2: usize = 4;

        let mut mha: MultiHeadAttention<EMBED_DIM, NUM_HEADS> = Default::default();
        mha.reset_params(&mut rng);

        let q: Tensor2D<S1, EMBED_DIM> = TensorCreator::randn(&mut rng);
        let k: Tensor2D<S2, EMBED_DIM> = TensorCreator::randn(&mut rng);
        let v: Tensor2D<S2, EMBED_DIM> = TensorCreator::randn(&mut rng);

        let y: Tensor2D<S1, EMBED_DIM> = mha.forward((q, k, v));

        // This expected y was generated by:
        // 1. saving `mha` parameters, `q`, `k`, `v` to a file
        // 2. Running pytorch with the same values
        // 3. printing out the output
        // See https://github.com/coreylowman/dfdx/wiki/Exporting-MultiHeadAttention-to-pytorch-for-unit-tests
        #[rustfmt::skip]
        assert_close(
            y.data(),
            &[
                [-0.41689563,-0.46807843,-0.10825230,-0.05752429, 0.18448383,-0.56645262,-0.03250163,-0.17918219],
                [-0.26238847,-0.43888292,-0.09987387,-0.03572154, 0.11067177,-0.48738408, 0.03990822,-0.34435043],
                [-0.42392737,-0.53368354,-0.11196917,-0.18274316, 0.23356661,-0.71953803,-0.11075453,-0.10825039],
            ],
        );
    }

    #[test]
    fn test_mha_batched() {
        let mut rng = StdRng::seed_from_u64(1);

        const BATCH: usize = 5;
        const EMBED_DIM: usize = 8;
        const NUM_HEADS: usize = 2;
        const S1: usize = 3;
        const S2: usize = 4;

        let mut mha: MultiHeadAttention<EMBED_DIM, NUM_HEADS> = Default::default();
        mha.reset_params(&mut rng);

        let q: Tensor3D<BATCH, S1, EMBED_DIM> = TensorCreator::randn(&mut rng);
        let k: Tensor3D<BATCH, S2, EMBED_DIM> = TensorCreator::randn(&mut rng);
        let v: Tensor3D<BATCH, S2, EMBED_DIM> = TensorCreator::randn(&mut rng);

        let y: Tensor3D<BATCH, S1, EMBED_DIM> = mha.forward((q, k, v));

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
                    [-0.32666653, 0.23977730, 0.25563523,-0.46537930, 0.19651681,-0.37467819, 0.44978297, 0.04501118],
                    [-0.32847843, 0.22905068, 0.24268147,-0.49660331, 0.17547092,-0.41919118, 0.45197228,-0.01052883],
                    [-0.28976738, 0.26420441, 0.24134403,-0.41927847, 0.21895495,-0.35072452, 0.44843924, 0.07374063],
                ],
                [
                    [-0.10029950, 0.15455982, 0.23578438,-0.36703593, 0.03778699,-0.41743413, 0.50207543, 0.11432818],
                    [-0.04076880, 0.24567264, 0.23325926,-0.19454414, 0.11575195,-0.22209120, 0.49752438, 0.30388331],
                    [-0.06600001, 0.20277922, 0.24651963,-0.24732135, 0.08645092,-0.28015324, 0.49499762, 0.23243824],
                ],
                [
                    [-0.18352799, 0.15783942, 0.36657059,-0.24797240, 0.11065251,-0.22565264, 0.46300891, 0.18687661],
                    [-0.15986431, 0.26687002, 0.30500177,-0.22695602, 0.18453379,-0.21377291, 0.46498343, 0.30064404],
                    [-0.09165541, 0.31019136, 0.20057595,-0.29627919, 0.15811513,-0.33667034, 0.48559439, 0.32546705],
                ],
                [
                    [-0.45827997, 0.08988418, 0.44279462,-0.45245945, 0.16884868,-0.26618001, 0.40024126, 0.01272556],
                    [-0.43258160, 0.11801003, 0.42784777,-0.41539627, 0.19628736,-0.23836099, 0.39999473, 0.05304383],
                    [-0.44729146, 0.09233949, 0.45179683,-0.41795415, 0.16631508,-0.22713992, 0.39473629, 0.04260518],
                ],
                [
                    [-0.51776350, 0.05404706, 0.39951840,-0.61738086, 0.21067555,-0.51225299, 0.41040331,-0.25894681],
                    [-0.47914022, 0.09410305, 0.36355501,-0.59280866, 0.24956036,-0.50058168, 0.40235144,-0.16756263],
                    [-0.55189615,-0.06088167, 0.41224611,-0.76746291, 0.09680001,-0.70136547, 0.40278757,-0.45541200],
                ],
            ],
        );
    }

    #[test]
    fn test_backward_updates_all() {
        let mut rng = thread_rng();

        let mut mha: MultiHeadAttention<12, 4> = Default::default();
        mha.reset_params(&mut rng);

        let q: Tensor3D<2, 3, 12> = TensorCreator::randn(&mut rng);
        let k: Tensor3D<2, 4, 12> = TensorCreator::randn(&mut rng);
        let v: Tensor3D<2, 4, 12> = TensorCreator::randn(&mut rng);
        let y: Tensor3D<2, 3, 12, _> = mha.forward((q.trace(), k, v));

        let mut g = SimpleGradients(y.mean().backward());
        let mut unused = Default::default();
        mha.update(&mut g, &mut unused);
        assert!(unused.is_empty());
    }

    #[test]
    fn test_save_and_load() {
        let mut rng = thread_rng();

        let mut saved: MultiHeadAttention<12, 4> = Default::default();
        saved.reset_params(&mut rng);

        let file = NamedTempFile::new().expect("failed to create tempfile");
        saved.save(file.path()).expect("");

        let mut loaded: MultiHeadAttention<12, 4> = Default::default();
        loaded.load(file.path()).expect("");

        let q: Tensor3D<2, 3, 12> = TensorCreator::randn(&mut rng);
        let k: Tensor3D<2, 4, 12> = TensorCreator::randn(&mut rng);
        let v: Tensor3D<2, 4, 12> = TensorCreator::randn(&mut rng);
        let y1: Tensor3D<2, 3, 12, _> = saved.forward((q.clone(), k.clone(), v.clone()));
        let y2: Tensor3D<2, 3, 12, _> = loaded.forward((q.clone(), k.clone(), v.clone()));

        assert_eq!(y1.data(), y2.data());
    }
}
