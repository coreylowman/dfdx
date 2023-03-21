use num_traits::Float;
use rand_distr::uniform::SampleUniform;

use crate::{nn::modules::*, shapes::*, tensor::*, tensor_ops::*};

pub mod builder {
    #[derive(Debug, Clone)]
    pub struct MultiHeadAttention<
        const EMBED_DIM: usize,
        const NUM_HEADS: usize,
        const K_DIM: usize = EMBED_DIM,
        const V_DIM: usize = EMBED_DIM,
    >;
    impl<const M: usize, const H: usize, const K: usize, const V: usize>
        MultiHeadAttention<M, H, K, V>
    {
        pub const TYPE_CHECK: () = assert!(
            K % H == 0 && V % H == 0,
            "NUM_HEADS must divide K_DIM & V_DIM evenly! If you haven't specified K_DIM & V_DIM, they default to EMBED_DIM, which means NUM_HEADS must divide EMBED_DIM evenly."
        );
    }
}

impl<const M: usize, const H: usize, const K: usize, const V: usize, E: Dtype, D: Device<E>>
    BuildOnDevice<D, E> for builder::MultiHeadAttention<M, H, K, V>
where
    MultiHeadAttention<M, H, K, V, E, D>: BuildModule<D, E>,
{
    type Built = MultiHeadAttention<M, H, K, V, E, D>;
    fn try_build_on_device(device: &D) -> Result<Self::Built, <D>::Err> {
        #[allow(clippy::let_unit_value)]
        let _ = Self::TYPE_CHECK;
        Self::Built::try_build(device)
    }
}

/// A multi-head attention layer.
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

impl<const M: usize, const H: usize, const K: usize, const V: usize, E, D: Device<E>>
    TensorCollection<E, D> for MultiHeadAttention<M, H, K, V, E, D>
where
    E: Dtype + Float + SampleUniform,
{
    type To<E2: Dtype, D2: Device<E2>> = MultiHeadAttention<M, H, K, V, E2, D2>;

    fn iter_tensors<Vi: ModuleVisitor<Self, E, D>>(
        visitor: &mut Vi,
    ) -> Result<Option<Self::To<Vi::E2, Vi::D2>>, Vi::Err> {
        visitor.visit_fields(
            (
                Self::module("w_q", |s| &s.w_q, |s| &mut s.w_q),
                Self::module("w_k", |s| &s.w_k, |s| &mut s.w_k),
                Self::module("w_v", |s| &s.w_v, |s| &mut s.w_v),
                Self::module("w_o", |s| &s.w_o, |s| &mut s.w_o),
            ),
            |(w_q, w_k, w_v, w_o)| MultiHeadAttention { w_q, w_k, w_v, w_o },
        )
    }
}

impl<const M: usize, const H: usize, const K: usize, const V: usize, E, D, S1, S2, T>
    Module<(
        Tensor<(S1, Const<M>), E, D, T>,
        Tensor<(S2, Const<M>), E, D>,
        Tensor<(S2, Const<M>), E, D>,
    )> for MultiHeadAttention<M, H, K, V, E, D>
where
    E: Dtype + Float,
    D: Device<E>,
    S1: Dim,
    S2: Dim,
    T: Tape<E, D>,
{
    type Output = Tensor<(S1, Const<M>), E, D, T>;
    type Error = D::Err;

    /// Encoder-Decoder style self attention where one set of tensors is used for values and keys, and another is used for queries
    fn try_forward(
        &self,
        (q, k, v): (
            Tensor<(S1, Const<M>), E, D, T>,
            Tensor<(S2, Const<M>), E, D>,
            Tensor<(S2, Const<M>), E, D>,
        ),
    ) -> Result<Self::Output, D::Err> {
        assert_eq!(k.shape.0, v.shape.0);
        let s1 = q.shape.0;
        let s2 = k.shape.0;
        let v = self.w_v.try_forward(v.retaped::<T>())?;
        let v = v.try_reshape_like(&(s2, H, V / H)).unwrap()?;
        let v = v.try_permute::<_, Axes3<1, 0, 2>>()?;

        let k = self.w_k.try_forward(k.retaped::<T>())?;
        let k = k.try_reshape_like(&(s2, H, K / H)).unwrap()?;
        let k = k.try_permute::<_, Axes3<1, 2, 0>>()?;

        let q = self.w_q.try_forward(q)?;
        let q = q.try_reshape_like(&(s1, H, K / H)).unwrap()?;
        let q = q.try_permute::<_, Axes3<1, 0, 2>>()?;

        // Get weights
        let scalar: E = E::ONE / E::from_usize(K / H).unwrap().sqrt();
        let weights = q.try_matmul(k)?.try_mul(scalar)?;
        let weights = weights.try_softmax::<Axis<2>>()?;

        // Get new tokens
        let tokens = weights.try_matmul(v)?;
        let tokens = tokens.try_permute::<_, Axes3<1, 0, 2>>()?;
        let tokens = tokens.try_reshape_like(&(s1, Const::<V>)).unwrap()?;

        self.w_o.try_forward(tokens)
    }
}

impl<const M: usize, const H: usize, const K: usize, const V: usize, E, D, B, S1, S2, T>
    Module<(
        Tensor<(B, S1, Const<M>), E, D, T>,
        Tensor<(B, S2, Const<M>), E, D>,
        Tensor<(B, S2, Const<M>), E, D>,
    )> for MultiHeadAttention<M, H, K, V, E, D>
where
    E: Dtype + Float,
    D: Device<E>,
    B: Dim,
    S1: Dim,
    S2: Dim,
    T: Tape<E, D>,
{
    type Output = Tensor<(B, S1, Const<M>), E, D, T>;
    type Error = D::Err;

    /// Batched Encoder-Decoder style self attention where one set of tensors is used for values and keys, and another is used for queries
    fn try_forward(
        &self,
        (q, k, v): (
            Tensor<(B, S1, Const<M>), E, D, T>,
            Tensor<(B, S2, Const<M>), E, D>,
            Tensor<(B, S2, Const<M>), E, D>,
        ),
    ) -> Result<Self::Output, D::Err> {
        assert_eq!(q.shape.0, k.shape.0);
        assert_eq!(q.shape.0, v.shape.0);
        assert_eq!(k.shape.1, v.shape.1);

        let b = q.shape.0;
        let s1 = q.shape.1;
        let s2 = v.shape.1;

        let v = self.w_v.try_forward(v.retaped::<T>())?;
        let v = v.try_reshape_like(&(b, s2, H, V / H)).unwrap()?;
        let v = v.try_permute::<_, Axes4<0, 2, 1, 3>>()?;

        let k = self.w_k.try_forward(k.retaped::<T>())?;
        let k = k.try_reshape_like(&(b, s2, H, K / H)).unwrap()?;
        let k = k.try_permute::<_, Axes4<0, 2, 3, 1>>()?;

        let q = self.w_q.try_forward(q)?;
        let q = q.try_reshape_like(&(b, s1, H, K / H)).unwrap()?;
        let q = q.try_permute::<_, Axes4<0, 2, 1, 3>>()?;

        // Get weights
        let scalar: E = E::ONE / E::from_usize(K / H).unwrap().sqrt();
        let weights = q.try_matmul(k)?.try_mul(scalar)?;
        let weights = weights.try_softmax::<Axis<3>>()?;

        // Get new tokens
        let tokens = weights.try_matmul(v)?;
        let tokens = tokens.try_permute::<_, Axes4<0, 2, 1, 3>>()?;
        let tokens = tokens.try_reshape_like(&(b, s1, Const::<V>)).unwrap()?;

        self.w_o.try_forward(tokens)
    }
}

impl<const M: usize, const H: usize, const K: usize, const V: usize, E, D, Src> Module<Src>
    for MultiHeadAttention<M, H, K, V, E, D>
where
    E: Dtype,
    D: Device<E>,
    Src: SplitTape,
    Self: Module<(Src, Src::NoTape, Src::NoTape), Output = Src, Error = D::Err>,
{
    type Output = Src;
    type Error = D::Err;

    fn try_forward(&self, src: Src) -> Result<Self::Output, D::Err> {
        let (src, tape) = src.split_tape();
        self.try_forward((src.clone().put_tape(tape), src.clone(), src))
    }
}

impl<const M: usize, const H: usize, const K: usize, const V: usize, E: Dtype, D: Device<E>>
    NonMutableModule for MultiHeadAttention<M, H, K, V, E, D>
{
}

#[cfg(test)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::{optim::*, tests::*};

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
        let y = mha.forward((q.leaky_trace(), k, v));
        let g = y.square().mean().backward();

        let mut opt = Sgd::new(&mha, Default::default());
        opt.update(&mut mha, &g).expect("");
    }
}
