use crate::{shapes::*, tensor::SplitTape, tensor_ops::Device};

use super::{
    BuildModule, BuildOnDevice, DeviceStorage, Module, ModuleMut, TensorFunction, TensorVisitor,
    ToDevice, VisitTensors,
};

use std::ops::Add;

/// A residual connection around `F`: `F(x) + x`,
/// as introduced in [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).
///
/// # Generics
/// - `F`: The underlying module to do a skip connection around.
///
/// # Examples
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// type Model = Residual<ReLU>;
/// let model = dev.build_module::<Model, f32>();
/// let x = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
/// let y = model.forward(x);
/// assert_eq!(y.array(), [-2.0, -1.0, 0.0, 2.0, 4.0]);
/// ```
#[derive(Debug, Clone, Default)]
pub struct Residual<F>(pub F);

impl<
        F: VisitTensors<E, D> + std::fmt::Debug,
        E: Dtype,
        D: DeviceStorage,
    > VisitTensors<E, D> for Residual<F>
{
    fn visit_groups<const N: usize, const M: usize, Func: TensorFunction<N, M, E, D>>(
        mut visitor: TensorVisitor<N, M, Self, Func>,
    ) -> Result<(), Func::Err> {
        visitor.visit_field(|s| &s.0, |s| &mut s.0, "0.")
    }
}

impl<D: Device<E>, E: Dtype, F: BuildOnDevice<D, E>> BuildOnDevice<D, E> for Residual<F> {
    type Built = Residual<F::Built>;
}

impl<D: Device<E>, E: Dtype, F: BuildModule<D, E>> BuildModule<D, E> for Residual<F> {
    fn try_build(device: &D) -> Result<Self, <D>::Err> {
        Ok(Self(BuildModule::try_build(device)?))
    }
}

impl<F: ToDevice<D>, D> ToDevice<D> for Residual<F> {
    type Output = Residual<F::Output>;
    fn to_device(&self, device: &D) -> Self::Output {
        Residual(self.0.to_device(device))
    }
}

impl<T: SplitTape + Add<T, Output = T>, F: Module<T, Output = T>> Module<T> for Residual<F> {
    type Output = T;
    fn forward(&self, x: T) -> Self::Output {
        self.0.forward(x.with_empty_tape()) + x
    }
}

impl<T: SplitTape + Add<T, Output = T>, F: ModuleMut<T, Output = T>> ModuleMut<T> for Residual<F> {
    type Output = T;
    fn forward_mut(&mut self, x: T) -> Self::Output {
        self.0.forward_mut(x.with_empty_tape()) + x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::DeviceBuildExt;
    use crate::tests::*;
    use crate::{nn::builders::Linear, tensor::*, tensor_ops::*};

    #[test]
    fn test_residual_reset() {
        let dev: TestDevice = Default::default();
        let model = dev.build_module::<Residual<Linear<2, 5>>, TestDtype>();
        assert_ne!(model.0.weight.array(), [[0.0; 2]; 5]);
        assert_ne!(model.0.bias.array(), [0.0; 5]);
    }

    #[test]
    fn test_residual_gradients() {
        let dev: TestDevice = Default::default();

        let model = <Residual<Linear<2, 2>>>::build_on_device(&dev);

        let x: Tensor<Rank2<4, 2>, f32, TestDevice> = dev.sample_normal();
        let y = model.forward(x.trace());

        #[rustfmt::skip]
        assert_close(&y.array(), &[[0.25372928, -2.4258814],[1.7892148, -2.6242268],[1.5131638, 0.23407778],[3.4201493, 1.597525]]);

        let g = y.mean().backward();
        assert_close(&g.get(&model.0.weight).array(), &[[0.475242, -0.075136]; 2]);
        assert_close(&g.get(&model.0.bias).array(), &[0.5; 2]);
        assert_close(&g.get(&x).array(), &[[0.18806472, 0.21419683]; 4]);
    }
}
