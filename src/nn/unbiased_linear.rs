use crate::{gradients::Tape, shapes::*, tensor::*, tensor_ops::*};

use super::{tensor_collection::*, BuildModule, BuildOnDevice, Module, NonMutableModule, ToDevice};

use num_traits::Float;
use rand_distr::{uniform::SampleUniform, Uniform};

pub mod builder {
    #[derive(Debug, Copy, Clone, Eq, PartialEq)]
    pub struct UnbiasedLinear<const I: usize, const O: usize>;
}

impl<const I: usize, const O: usize, E: Dtype, D: Device<E>> BuildOnDevice<D, E>
    for builder::UnbiasedLinear<I, O>
where
    UnbiasedLinear<I, O, E, D>: BuildModule<D, E>,
{
    type Built = UnbiasedLinear<I, O, E, D>;
    fn try_build_on_device(device: &D) -> Result<Self::Built, <D>::Err> {
        Self::Built::try_build(device)
    }
}

/// A linear transformation of the form `weight * x`, where `weight` is a matrix, `x` is a vector or matrix.
///
/// Initializes [Self::weight] from a Uniform distribution
/// between [-1 / sqrt(I), 1 / sqrt(I)].
///
/// # Generics
/// - `I` The "input" size of vectors & matrices.
/// - `O` The "output" size of vectors & matrices.
///
/// # Examples
/// `UnbiasedLinear<5, 2>` can act on vectors with 5 elements, and results in vectors with 2 elements.
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// type Model = UnbiasedLinear<5, 2>;
/// let model = dev.build_module::<Model, f32>();
/// // single item forward
/// let _: Tensor<Rank1<2>, f32, _> = model.forward(dev.zeros::<Rank1<5>>());
/// // batched forward
/// let _: Tensor<Rank2<10, 2>, f32, _> = model.forward(dev.zeros::<Rank2<10, 5>>());
/// ```
#[derive(Debug, Clone)]
pub struct UnbiasedLinear<const I: usize, const O: usize, E: Dtype, D: DeviceStorage> {
    /// Transposed weight matrix, shape (I, O)
    pub weight: Tensor<Rank2<O, I>, E, D>,
}

impl<const I: usize, const O: usize, E: Dtype, D: DeviceStorage> NonMutableModule
    for UnbiasedLinear<I, O, E, D>
{
}

impl<const I: usize, const O: usize, E: Dtype + Float + SampleUniform, D: Device<E>>
    BuildModule<D, E> for UnbiasedLinear<I, O, E, D>
{
    fn try_build(device: &D) -> Result<Self, D::Err> {
        let b: E = E::ONE / E::from_usize(I).unwrap().sqrt();
        let weight = device.try_sample(Uniform::new(-b, b))?;
        Ok(Self { weight })
    }
}

impl<const I: usize, const O: usize, E: Dtype + Float + SampleUniform, D: SampleTensor<E>>
    TensorCollection<E, D> for UnbiasedLinear<I, O, E, D>
{
    fn iter_tensors<V: ModuleVisitor<Self, E, D>>(visitor: &mut V) -> Result<(), V::Err> {
        visitor.visit_tensor(
            "weight",
            |s| &s.weight,
            |s| &mut s.weight,
            TensorOptions::reset_with(|t| {
                let b: E = E::ONE / E::from_usize(I).unwrap().sqrt();
                t.try_fill_with_distr(Uniform::new(-b, b))
            }),
        )
    }
}

impl<const I: usize, const O: usize, E: Dtype, D1: Device<E>, D2: Device<E>> ToDevice<D2>
    for UnbiasedLinear<I, O, E, D1>
{
    type Output = UnbiasedLinear<I, O, E, D2>;
    fn to_device(&self, device: &D2) -> Self::Output {
        UnbiasedLinear {
            weight: self.weight.to_device(device),
        }
    }
}

impl<const I: usize, const O: usize, E: Dtype, D: Device<E>, T> Module<T>
    for UnbiasedLinear<I, O, E, D>
where
    T: SplitTape + TryMatMul<Tensor<Rank2<I, O>, E, D, T::Tape>> + HasErr<Err = D::Err>,
    T::Tape: Tape<D>,
    for<'a> Bias1D<'a, O, E, D>: Module<T::Output, Output = T::Output, Error = D::Err>,
{
    type Output = T::Output;
    type Error = D::Err;

    /// 1d forward using [matmul()] and [add()].
    fn try_forward(&self, x: T) -> Result<Self::Output, D::Err> {
        x.try_matmul(self.weight.retaped::<T::Tape>().try_permute()?)
    }
}

#[derive(Clone, Debug)]
struct Bias1D<'a, const M: usize, E: Dtype, D: DeviceStorage> {
    beta: &'a Tensor<Rank1<M>, E, D>,
}

impl<'a, const M: usize, E: Dtype, D: Device<E>, T: Tape<D>> Module<Tensor<Rank1<M>, E, D, T>>
    for Bias1D<'a, M, E, D>
{
    type Output = Tensor<Rank1<M>, E, D, T>;
    type Error = D::Err;

    fn try_forward(&self, input: Tensor<Rank1<M>, E, D, T>) -> Result<Self::Output, D::Err> {
        input.try_add(self.beta.clone())
    }
}

impl<'a, B: Dim, const M: usize, E: Dtype, D: Device<E>, T: Tape<D>>
    Module<Tensor<(B, Const<M>), E, D, T>> for Bias1D<'a, M, E, D>
{
    type Output = Tensor<(B, Const<M>), E, D, T>;
    type Error = D::Err;

    fn try_forward(&self, input: Tensor<(B, Const<M>), E, D, T>) -> Result<Self::Output, D::Err> {
        self.beta
            .retaped::<T>()
            .try_broadcast_like(input.shape())?
            .try_add(input)
    }
}

impl<'a, B: Dim, S: Dim, const M: usize, E: Dtype, D: Device<E>, T: Tape<D>>
    Module<Tensor<(B, S, Const<M>), E, D, T>> for Bias1D<'a, M, E, D>
{
    type Output = Tensor<(B, S, Const<M>), E, D, T>;
    type Error = D::Err;

    fn try_forward(
        &self,
        input: Tensor<(B, S, Const<M>), E, D, T>,
    ) -> Result<Self::Output, D::Err> {
        self.beta
            .retaped::<T>()
            .try_broadcast_like(input.shape())?
            .try_add(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{nn::DeviceBuildExt, tests::*};

    const W: [[TestDtype; 5]; 2] = [
        [-0.3458893, -0.30371523, -0.3712057, 0.14303583, -0.0268966],
        [0.11733949, 0.14059687, -0.10670426, -0.09373143, 0.18974298],
    ];

    #[test]
    fn test_unbiased_linear_ondevice() {
        let dev: TestDevice = Default::default();
        let _: UnbiasedLinear<1, 1, TestDtype, _> = BuildModule::build(&dev);
        let _: UnbiasedLinear<1, 1, TestDtype, TestDevice> =
            builder::UnbiasedLinear::<1, 1>::build_on_device(&dev);
        let _: UnbiasedLinear<1, 1, TestDtype, _> =
            builder::UnbiasedLinear::<1, 1>::build_on_device(&dev);
        let _ = dev.build_module::<builder::UnbiasedLinear<1, 1>, TestDtype>();
    }

    #[test]
    fn test_unbiased_linear_initialize() {
        let dev: TestDevice = Default::default();
        let m = dev.build_module::<builder::UnbiasedLinear<2000, 1>, TestDtype>();
        let bound: TestDtype = 1.0 / 2000.0;
        let bound = bound.sqrt();
        for v in m.weight.as_vec() {
            assert!(-bound <= v && v <= bound && v != 0.0);
        }
    }

    #[test]
    fn test_forward_1d() {
        let dev: TestDevice = Default::default();

        let model = UnbiasedLinear {
            weight: dev.tensor(W),
        };

        let x = dev.tensor([-0.8808001, 2.4185333, 2.2478335, 0.0565211, 2.031299]);
        let y = model.forward(x.trace());
        assert_close(&y.array(), &[-1.3108451, 0.37695912]);

        let g = y.square().mean().backward();
        assert_close(
            &g.get(&model.weight).array(),
            &[
                [1.1545925, -3.1703227, -2.9465616, -0.07409041, -2.6627185],
                [-0.33202565, 0.9116882, 0.8473413, 0.021306144, 0.76571673],
            ],
        );
    }

    #[test]
    fn test_forward_2d() {
        let dev: TestDevice = Default::default();

        let model = UnbiasedLinear {
            weight: dev.tensor(W),
        };

        let x = dev.tensor([
            [-1.9468665, 1.4611785, -1.6698982, 1.408863, 1.3425643],
            [-1.3399831, 3.0510678, -0.17936817, -0.04943254, -0.8052705],
            [-0.8291412, 0.07691376, -0.26538327, 0.90017676, -1.8790455],
        ]);
        let y = model.forward(x.trace());
        assert_close(
            &y.array(),
            &[
                [1.0149013, 0.27786547],
                [-0.38199908, 0.14271596],
                [0.5412404, -0.4990702],
            ],
        );

        let g = y.square().mean().backward();
        assert_close(
            &g.get(&model.weight).array(),
            &[
                [-0.63758993, 0.11969192, -0.58996654, 0.6453174, 0.21772254],
                [
                    -0.106134765,
                    0.26768726,
                    -0.11905362,
                    -0.021610606,
                    0.39863434,
                ],
            ],
        );
    }

    #[test]
    fn test_forward_3d() {
        let dev: TestDevice = Default::default();

        let model = UnbiasedLinear {
            weight: dev.tensor(W),
        };

        #[rustfmt::skip]
        let x = dev.tensor([
            [[-1.9468665, 1.4611785, -1.6698982, 1.408863, 1.3425643], [-1.3399831, 3.0510678, -0.17936817, -0.04943254, -0.8052705], [-0.8291412, 0.07691376, -0.26538327, 0.90017676, -1.8790455]],
            [[1.2879219, 0.70150787, -1.6746868, 1.7261779, -0.94021803], [-2.6883178, 2.9369607, 2.9256766, 0.27559614, -0.17530347], [0.17499207, -0.11440835, 0.16627812, -0.91773695, 1.1128315]],
        ]);
        let y = model.forward(x.trace());
        assert_close(
            &y.array(),
            &[
                [
                    [1.0149013, 0.27786547],
                    [-0.38199908, 0.14271596],
                    [0.5412404, -0.4990702],
                ],
                [
                    [0.2353102, 0.0882532],
                    [-1.0040319, -0.27379516],
                    [-0.24870436, 0.2838782],
                ],
            ],
        );

        let g = y.square().mean().backward();
        #[rustfmt::skip]
        assert_close(
            &g.get(&model.weight).array(),
            &[[0.17432114, -0.3993668, -0.85713285, 0.38227955, 0.05519483], [0.096830636, 0.004728079, -0.20979846, -0.04141225, 0.2461386]]
        );
    }
}
