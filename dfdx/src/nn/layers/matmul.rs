use crate::prelude::*;

use rand_distr::Uniform;

/// Performs matrix multiplication of the form `x * W^T`, where `x` is the input, and `W` is the weight matrix.
/// `x` can be 1d, 2d, or 3d.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// # use dfdx::*;
/// # let dev: Cpu = Default::default();
/// type Model = MatMulConstConfig<5, 2>;
/// let model = dev.build_module::<f32>(Model::default());
/// // single item forward
/// let _: Tensor<Rank1<2>, f32, _> = model.forward(dev.zeros::<Rank1<5>>());
/// // batched forward
/// let _: Tensor<Rank2<10, 2>, f32, _> = model.forward(dev.zeros::<Rank2<10, 5>>());
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct MatMulConfig<I: Dim, O: Dim> {
    pub inp: I,
    pub out: O,
}

/// Compile time sugar alias around [MatMulConfig].
pub type MatMulConstConfig<const I: usize, const O: usize> = MatMulConfig<Const<I>, Const<O>>;

impl<I: Dim, O: Dim, E: Dtype, D: Device<E>> BuildOnDevice<E, D> for MatMulConfig<I, O> {
    type Built = MatMul<I, O, E, D>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, crate::tensor::Error> {
        Ok(MatMul {
            weight: device.try_zeros_like(&(self.out, self.inp))?,
        })
    }
}

/// See [MatMulConfig].
#[derive(Clone, Debug, UpdateParams, ZeroGrads, WithGrads)]
#[cfg_attr(feature = "safetensors", derive(SaveSafeTensors, LoadSafeTensors))]
pub struct MatMul<I: Dim, O: Dim, Elem: Dtype, Dev: Device<Elem>> {
    #[param]
    #[cfg_attr(feature = "safetensors", serialize)]
    pub weight: Tensor<(O, I), Elem, Dev>,
}

impl<I: Dim, O: Dim, E, D: Device<E>> ResetParams<E, D> for MatMul<I, O, E, D>
where
    E: Dtype + num_traits::Float + rand_distr::uniform::SampleUniform,
{
    fn try_reset_params(&mut self) -> Result<(), Error> {
        let (_o, i) = self.weight.shape();
        let scale = E::from_f64(1.0 / (i.size() as f64).sqrt()).unwrap();
        self.weight.try_fill_with_distr(Uniform::new(-scale, scale))
    }
}

impl<S: Shape, I: Dim, O: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<S, E, D, T>>
    for MatMul<I, O, E, D>
where
    Tensor<S, E, D, T>: TryMatMul<Tensor<(I, O), E, D, T>>,
{
    type Output = <Tensor<S, E, D, T> as TryMatMul<Tensor<(I, O), E, D, T>>>::Output;
    fn try_forward(&self, x: Tensor<S, E, D, T>) -> Result<Self::Output, Error> {
        x.try_matmul(self.weight.retaped::<T>().try_permute()?)
    }
}
