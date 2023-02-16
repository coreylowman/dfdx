use num_traits::Float;
use rand_distr::uniform::SampleUniform;

use crate::{
    nn::{modules::*, *},
    shapes::Dtype,
    tensor::*,
    tensor_ops::*,
};

#[cfg(feature = "nightly")]
use crate::{gradients::Tape, shapes::*, Assert, ConstTrue};

pub mod builder {
    #[derive(Debug, Clone)]
    pub struct MultiHeadAttention<
        const EMBED_DIM: usize,
        const NUM_HEADS: usize,
        const K_DIM: usize = EMBED_DIM,
        const V_DIM: usize = EMBED_DIM,
    >;
}

impl<const M: usize, const H: usize, const K: usize, const V: usize, E: Dtype, D: Device<E>>
    BuildOnDevice<D, E> for builder::MultiHeadAttention<M, H, K, V>
where
    MultiHeadAttention<M, H, K, V, E, D>: BuildModule<D, E>,
{
    type Built = MultiHeadAttention<M, H, K, V, E, D>;
    fn try_build_on_device(device: &D) -> Result<Self::Built, <D>::Err> {
        Self::Built::try_build(device)
    }
}

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
#[derive(Debug, Clone)]
pub struct MultiHeadAttention<
    const EMBED_DIM: usize,
    const NUM_HEADS: usize,
    const K_DIM: usize,
    const V_DIM: usize,
    E: Dtype,
    D: DeviceStorage,
> {
    pub w_q: Linear<EMBED_DIM, K_DIM, E, D>,
    pub w_k: Linear<EMBED_DIM, K_DIM, E, D>,
    pub w_v: Linear<EMBED_DIM, V_DIM, E, D>,
    pub w_o: Linear<V_DIM, EMBED_DIM, E, D>,
}

impl<
        const N: usize,
        const M: usize,
        const EMBED_DIM: usize,
        const NUM_HEADS: usize,
        const K_DIM: usize,
        const V_DIM: usize,
        E: Dtype,
        D: DeviceStorage,
    > VisitTensorGroups<N, M, E, D>
    for MultiHeadAttention<EMBED_DIM, NUM_HEADS, K_DIM, V_DIM, E, D>
{
    #[rustfmt::skip]
    fn visit_groups<F: TensorVisitor<N, M, E, D>>(
        mut self_refs: ModuleGroup<N, M, Self>,
        func: &mut F,
    ) -> Result<(), F::Err> {
        self_refs.map(|s| &s.w_q, |s| &mut s.w_q, "w_q.").visit(func)?;
        self_refs.map(|s| &s.w_k, |s| &mut s.w_k, "w_k.").visit(func)?;
        self_refs.map(|s| &s.w_v, |s| &mut s.w_v, "w_v.").visit(func)?;
        self_refs.map(|s| &s.w_o, |s| &mut s.w_o, "w_o.").visit(func)
    }
}

impl<const M: usize, const H: usize, const K: usize, const V: usize, E, D: Device<E>>
    BuildModule<D, E> for MultiHeadAttention<M, H, K, V, E, D>
where
    E: Dtype + Float + SampleUniform,
{
    fn try_build(device: &D) -> Result<Self, <D>::Err> {
        Ok(Self {
            w_q: BuildModule::try_build(device)?,
            w_k: BuildModule::try_build(device)?,
            w_v: BuildModule::try_build(device)?,
            w_o: BuildModule::try_build(device)?,
        })
    }
}

impl<const M: usize, const H: usize, const K: usize, const V: usize, E, D: Device<E>>
    ResetParams<D, E> for MultiHeadAttention<M, H, K, V, E, D>
where
    E: Dtype + Float + SampleUniform,
{
    fn try_reset_params(&mut self) -> Result<(), <D>::Err> {
        self.w_q.try_reset_params()?;
        self.w_k.try_reset_params()?;
        self.w_v.try_reset_params()?;
        self.w_o.try_reset_params()?;
        Ok(())
    }
}

impl<const M: usize, const H: usize, const K: usize, const V: usize, E, D1, D2> ToDevice<D2>
    for MultiHeadAttention<M, H, K, V, E, D1>
where
    E: Dtype,
    D1: Device<E>,
    D2: Device<E>,
{
    type Output = MultiHeadAttention<M, H, K, V, E, D2>;

    fn to_device(&self, device: &D2) -> Self::Output {
        MultiHeadAttention {
            w_q: self.w_q.to_device(device),
            w_k: self.w_k.to_device(device),
            w_v: self.w_v.to_device(device),
            w_o: self.w_o.to_device(device),
        }
    }
}

#[cfg(feature = "nightly")]
impl<
        const M: usize,
        const H: usize,
        const K: usize,
        const V: usize,
        E: Dtype + Float,
        D: Device<E>,
        const S1: usize,
        const S2: usize,
        T: Tape<D>,
    >
    Module<(
        Tensor<Rank2<S1, M>, E, D, T>,
        Tensor<Rank2<S2, M>, E, D>,
        Tensor<Rank2<S2, M>, E, D>,
    )> for MultiHeadAttention<M, H, K, V, E, D>
where
    Assert<{ S1 * K == S1 * H * (K / H) }>: ConstTrue,
    Assert<{ S2 * K == S2 * H * (K / H) }>: ConstTrue,
    Assert<{ S2 * V == S2 * H * (V / H) }>: ConstTrue,
    Assert<{ S1 * H * (V / H) == S1 * V }>: ConstTrue,
{
    type Output = Tensor<Rank2<S1, M>, E, D, T>;

    /// Encoder-Decoder style self attention where one set of tensors is used for values and keys, and another is used for queries
    fn forward(
        &self,
        (q, k, v): (
            Tensor<Rank2<S1, M>, E, D, T>,
            Tensor<Rank2<S2, M>, E, D>,
            Tensor<Rank2<S2, M>, E, D>,
        ),
    ) -> Self::Output {
        let v: Tensor<Rank2<S2, V>, _, _, _> = self.w_v.forward(v.retaped::<T>());
        let v = v.reshape::<Rank3<S2, H, { V / H }>>();
        let v = v.permute::<Rank3<H, S2, { V / H }>, _>();

        let k: Tensor<Rank2<S2, K>, _, _, _> = self.w_k.forward(k.retaped::<T>());
        let k = k.reshape::<Rank3<S2, H, { K / H }>>();
        let k = k.permute::<Rank3<H, { K / H }, S2>, _>();

        let q: Tensor<Rank2<S1, K>, _, _, _> = self.w_q.forward(q);
        let q = q.reshape::<Rank3<S1, H, { K / H }>>();
        let q = q.permute::<Rank3<H, S1, { K / H }>, _>();

        // Get weights
        let scalar: E = E::ONE / E::from_usize(K / H).unwrap().sqrt();
        let weights: Tensor<Rank3<H, S1, S2>, _, _, _> = q.matmul(k) * scalar;
        let weights = weights.softmax::<Axis<2>>();

        // Get new tokens
        let tokens: Tensor<Rank3<H, S1, { V / H }>, _, _, _> = weights.matmul(v);
        let tokens = tokens.permute::<Rank3<S1, H, { V / H }>, _>();
        let tokens = tokens.reshape::<Rank2<S1, V>>();

        self.w_o.forward(tokens)
    }
}

#[cfg(feature = "nightly")]
impl<
        const M: usize,
        const H: usize,
        const K: usize,
        const V: usize,
        E: Dtype + Float,
        D: Device<E>,
        const B: usize,
        const S1: usize,
        const S2: usize,
        T: Tape<D>,
    >
    Module<(
        Tensor<Rank3<B, S1, M>, E, D, T>,
        Tensor<Rank3<B, S2, M>, E, D>,
        Tensor<Rank3<B, S2, M>, E, D>,
    )> for MultiHeadAttention<M, H, K, V, E, D>
where
    Assert<{ B * S1 * K == B * S1 * H * (K / H) }>: ConstTrue,
    Assert<{ B * S2 * K == B * S2 * H * (K / H) }>: ConstTrue,
    Assert<{ B * S2 * V == B * S2 * H * (V / H) }>: ConstTrue,
    Assert<{ B * S1 * H * (V / H) == B * S1 * V }>: ConstTrue,
{
    type Output = Tensor<Rank3<B, S1, M>, E, D, T>;

    /// Batched Encoder-Decoder style self attention where one set of tensors is used for values and keys, and another is used for queries
    fn forward(
        &self,
        (q, k, v): (
            Tensor<Rank3<B, S1, M>, E, D, T>,
            Tensor<Rank3<B, S2, M>, E, D>,
            Tensor<Rank3<B, S2, M>, E, D>,
        ),
    ) -> Self::Output {
        let v: Tensor<Rank3<B, S2, V>, _, _, _> = self.w_v.forward(v.retaped::<T>());
        let v = v.reshape::<Rank4<B, S2, H, { V / H }>>();
        let v = v.permute::<Rank4<B, H, S2, { V / H }>, _>();

        let k: Tensor<Rank3<B, S2, K>, _, _, _> = self.w_k.forward(k.retaped::<T>());
        let k = k.reshape::<Rank4<B, S2, H, { K / H }>>();
        let k = k.permute::<Rank4<B, H, { K / H }, S2>, _>();

        let q: Tensor<Rank3<B, S1, K>, _, _, _> = self.w_q.forward(q);
        let q = q.reshape::<Rank4<B, S1, H, { K / H }>>();
        let q = q.permute::<Rank4<B, H, S1, { K / H }>, _>();

        // Get weights
        let scalar: E = E::ONE / E::from_usize(K / H).unwrap().sqrt();
        let weights: Tensor<Rank4<B, H, S1, S2>, _, _, _> = q.matmul(k) * scalar;
        let weights = weights.softmax::<Axis<3>>();

        // Get new tokens
        let tokens: Tensor<Rank4<B, H, S1, { V / H }>, _, _, _> = weights.matmul(v);
        let tokens = tokens.permute::<Rank4<B, S1, H, { V / H }>, _>();
        let tokens = tokens.reshape::<Rank3<B, S1, V>>();

        self.w_o.forward(tokens)
    }
}

impl<const M: usize, const H: usize, const K: usize, const V: usize, E, D, Src> Module<Src>
    for MultiHeadAttention<M, H, K, V, E, D>
where
    E: Dtype,
    D: Device<E>,
    Src: SplitTape,
    Self: Module<(Src, Src::NoTape, Src::NoTape), Output = Src>,
{
    type Output = Src;
    fn forward(&self, src: Src) -> Self::Output {
        let (src, tape) = src.split_tape();
        self.forward((src.clone().put_tape(tape), src.clone(), src))
    }
}

impl<const M: usize, const H: usize, const K: usize, const V: usize, E: Dtype, D: Device<E>, T>
    ModuleMut<T> for MultiHeadAttention<M, H, K, V, E, D>
where
    Self: Module<T>,
{
    type Output = <Self as Module<T>>::Output;

    fn forward_mut(&mut self, t: T) -> Self::Output {
        self.forward(t)
    }
}

#[cfg(feature = "nightly")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{nn::tests::SimpleUpdater, optim::GradientUpdate, tests::*};

    #[test]
    fn test_mha_unbatched() {
        let dev = TestDevice::seed_from_u64(0);

        const M: usize = 8;
        const NUM_HEADS: usize = 2;
        const S1: usize = 3;
        const S2: usize = 4;

        type Dtype = f32;

        let mha = dev.build_module::<builder::MultiHeadAttention<M, NUM_HEADS>, Dtype>();

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
        assert_close(
            &y.array(),
            &[
                [-0.41689563,-0.46807843,-0.10825230,-0.05752429, 0.18448383,-0.56645262,-0.03250163,-0.17918219],
                [-0.26238847,-0.43888292,-0.09987387,-0.03572154, 0.11067177,-0.48738408, 0.03990822,-0.34435043],
                [-0.42392737,-0.53368354,-0.11196917,-0.18274316, 0.23356661,-0.71953803,-0.11075453,-0.10825039],
            ],
        );
    }

    #[test]
    fn test_mha_batched() {
        let dev = TestDevice::seed_from_u64(1);

        const BATCH: usize = 5;
        const M: usize = 8;
        const NUM_HEADS: usize = 2;
        const S1: usize = 3;
        const S2: usize = 4;

        type Dtype = f32;

        let mha = dev.build_module::<builder::MultiHeadAttention<M, NUM_HEADS>, Dtype>();

        let q: Tensor<Rank3<BATCH, S1, M>, Dtype, _> = dev.sample_normal();
        let k: Tensor<Rank3<BATCH, S2, M>, Dtype, _> = dev.sample_normal();
        let v: Tensor<Rank3<BATCH, S2, M>, Dtype, _> = dev.sample_normal();

        let y = mha.forward((q, k, v));

        // This expected y was generated by:
        // 1. saving `mha` parameters, `q`, `k`, `v` to a file
        // 2. Running pytorch with the same values
        // 3. printing out the output
        // See https://github.com/coreylowman/dfdx/wiki/Exporting-MultiHeadAttention-to-pytorch-for-unit-tests
        #[rustfmt::skip]
        assert_close(
            &y.array(),
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
        let dev: TestDevice = Default::default();

        let mut mha = dev.build_module::<builder::MultiHeadAttention<12, 4>, TestDtype>();

        let q: Tensor<Rank3<2, 3, 12>, TestDtype, _> = dev.sample_normal();
        let k: Tensor<Rank3<2, 4, 12>, TestDtype, _> = dev.sample_normal();
        let v: Tensor<Rank3<2, 4, 12>, TestDtype, _> = dev.sample_normal();
        let y = mha.forward((q.trace(), k, v));

        let mut g = SimpleUpdater(y.mean().backward());
        let mut unused = Default::default();
        mha.update(&mut g, &mut unused).unwrap();
        assert!(unused.is_empty());
    }
}
