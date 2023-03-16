use crate::{prelude::Device, shapes::*, tensor::*, tensor_ops::TryAdd};

use super::*;

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

impl<D: Device<E>, E: Dtype, F: BuildOnDevice<D, E>> BuildOnDevice<D, E> for Residual<F> {
    type Built = Residual<F::Built>;
}

impl<E: Dtype, D: Device<E>, F: TensorCollection<E, D>> TensorCollection<E, D> for Residual<F> {
    type To<E2: Dtype, D2: Device<E2>> = Residual<F::To<E2, D2>>;

    fn iter_tensors<V: ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(Self::module("0", |s| &s.0, |s| &mut s.0), Residual)
    }
}

impl<T: WithEmptyTape + TryAdd<T>, F: Module<T, Output = T, Error = T::Err>> Module<T>
    for Residual<F>
{
    type Output = T;
    type Error = F::Error;

    fn try_forward(&self, x: T) -> Result<Self::Output, F::Error> {
        self.0.try_forward(x.with_empty_tape())?.try_add(x)
    }
}

impl<T: WithEmptyTape + TryAdd<T>, F: ModuleMut<T, Output = T, Error = T::Err>> ModuleMut<T>
    for Residual<F>
{
    type Output = T;
    type Error = F::Error;

    fn try_forward_mut(&mut self, x: T) -> Result<Self::Output, F::Error> {
        self.0.try_forward_mut(x.with_empty_tape())?.try_add(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::*;
    use crate::{nn::builders::Linear, tensor_ops::*};

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
        let y = model.forward(x.leaky_trace());

        #[rustfmt::skip]
        assert_close(&y.array(), &[[0.25372928, -2.4258814],[1.7892148, -2.6242268],[1.5131638, 0.23407778],[3.4201493, 1.597525]]);

        let g = y.mean().backward();
        assert_close(&g.get(&model.0.weight).array(), &[[0.475242, -0.075136]; 2]);
        assert_close(&g.get(&model.0.bias).array(), &[0.5; 2]);
        assert_close(&g.get(&x).array(), &[[0.18806472, 0.21419683]; 4]);
    }
}
