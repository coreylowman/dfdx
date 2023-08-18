use crate::*;
use dfdx::{
    prelude::{Device, Dim, Dtype, HasErr, HasShape, Shape, Tape, Tensor},
    shapes::Const,
    tensor_ops::TryMatMul,
};
use rand_distr::Uniform;

#[derive(Clone, Copy, Debug, Default)]
pub struct MatMulConfig<I: Dim, O: Dim> {
    pub inp: I,
    pub out: O,
}

pub type MatMulConstConfig<const I: usize, const O: usize> = MatMulConfig<Const<I>, Const<O>>;

impl<I: Dim, O: Dim, E: Dtype, D: Device<E>> BuildOnDevice<E, D> for MatMulConfig<I, O> {
    type Built = MatMul<I, O, E, D>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, D::Err> {
        Ok(MatMul {
            weight: device.try_zeros_like(&(self.inp, self.out))?,
        })
    }
}

#[derive(Clone, Debug, UpdateParams, ZeroGrads, SaveSafeTensors, LoadSafeTensors)]
pub struct MatMul<I: Dim, O: Dim, Elem: Dtype, Dev: Device<Elem>> {
    #[param]
    #[serialize]
    pub weight: Tensor<(I, O), Elem, Dev>,
}

// NOTE: others can simply #[derive(ResetParams)]
impl<I: Dim, O: Dim, E, D: Device<E>> ResetParams<E, D> for MatMul<I, O, E, D>
where
    E: Dtype + num_traits::Float + rand_distr::uniform::SampleUniform,
{
    fn try_reset_params(&mut self) -> Result<(), D::Err> {
        let (i, _) = self.weight.shape();
        let scale = E::from_f64(1.0 / (i.size() as f64).sqrt()).unwrap();
        self.weight.try_fill_with_distr(Uniform::new(-scale, scale))
    }
}

impl<S: Shape, I: Dim, O: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<S, E, D, T>>
    for MatMul<I, O, E, D>
where
    Tensor<S, E, D, T>: TryMatMul<Tensor<(I, O), E, D>>,
{
    type Output = <Tensor<S, E, D, T> as TryMatMul<Tensor<(I, O), E, D>>>::Output;
    type Error = <Tensor<S, E, D, T> as HasErr>::Err;
    fn try_forward(&self, x: Tensor<S, E, D, T>) -> Result<Self::Output, Self::Error> {
        x.try_matmul(self.weight.clone())
    }
}
