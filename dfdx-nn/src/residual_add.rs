use crate::*;
use dfdx::prelude::*;

/// A residual connection around `T`: `T(x) + x`,
/// as introduced in [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).
///
/// # Generics
/// - `T`: The underlying module to do a skip connection around.
///
/// # Examples
/// ```rust
/// # use dfdx::prelude::*;
/// # use dfdx_nn::*;
/// # let dev: Cpu = Default::default();
/// type Model = ResidualAdd<ReLU>;
/// let model = dev.build_module::<f32>(Model::default());
/// let x = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
/// let y = model.forward(x);
/// assert_eq!(y.array(), [-2.0, -1.0, 0.0, 2.0, 4.0]);
/// ```
#[derive(
    Default, Clone, Debug, ResetParams, ZeroGrads, UpdateParams, SaveSafeTensors, LoadSafeTensors,
)]
#[repr(transparent)]
pub struct ResidualAdd<T>(
    #[module]
    #[serialize]
    pub T,
);

// TODO derive this
impl<E: Dtype, D: Device<E>, T: BuildOnDevice<E, D>> BuildOnDevice<E, D> for ResidualAdd<T> {
    type Built = ResidualAdd<T::Built>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, <D>::Err> {
        let t = self.0.try_build_on_device(device)?;
        Ok(ResidualAdd(t))
    }
}

impl<X: WithEmptyTape, T: Module<X>> Module<X> for ResidualAdd<T>
where
    X: TryAdd<T::Output, Err = T::Error>,
{
    type Output = X::Output;
    type Error = T::Error;
    fn try_forward(&self, x: X) -> Result<Self::Output, Self::Error> {
        let t = self.0.try_forward(x.with_empty_tape())?;
        x.try_add(t)
    }
    fn try_forward_mut(&mut self, x: X) -> Result<Self::Output, Self::Error> {
        let t = self.0.try_forward_mut(x.with_empty_tape())?;
        x.try_add(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::*;

    #[test]
    fn test_residual_gradients() {
        let dev: TestDevice = Default::default();

        let model = dev.build_module::<f32>(<ResidualAdd<LinearConstConfig<2, 2>>>::default());
        let model = ResidualAdd(Linear {
            matmul: MatMul {
                weight: model.0.matmul.weight.to_dtype::<TestDtype>(),
            },
            add: Bias1D {
                bias: model.0.add.bias.to_dtype::<TestDtype>(),
            },
        });

        let x: Tensor<Rank2<4, 2>, f32, _> = dev.sample_normal();
        let x = x.to_dtype::<TestDtype>();
        let y = model.forward(x.leaky_trace());

        #[rustfmt::skip]
        assert_close_to_literal!(y, [[0.25372928, -2.4258814],[1.7892148, -2.6242268],[1.5131638, 0.23407778],[3.4201493, 1.597525]]);

        let g = y.mean().backward();
        assert_close_to_literal!(g.get(&model.0.matmul.weight), [[0.475242, -0.075136]; 2]);
        assert_close_to_literal!(g.get(&model.0.add.bias), [0.5; 2]);
        assert_close_to_literal!(g.get(&x), [[0.18806472, 0.21419683]; 4]);
    }
}
