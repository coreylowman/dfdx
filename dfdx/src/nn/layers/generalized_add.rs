use crate::prelude::*;

/// A residual connection around two modules: `T(x) + U(x)`,
/// as introduced in [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).
///
/// # Generics
/// - `T`: The underlying module to do a skip connection around.
/// - `U`: The underlying residual module
///
/// # Examples
/// ```rust
/// # use dfdx::prelude::*;
/// # use dfdx::*;
/// # let dev: Cpu = Default::default();
/// type Model = GeneralizedAdd<ReLU, Square>;
/// let model = dev.build_module::<f32>(Model::default());
/// let x = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
/// let y = model.forward(x);
/// assert_eq!(y.array(), [4.0, 1.0, 0.0, 2.0, 6.0]);
/// ```
#[derive(Default, Clone, Debug, ResetParams, ZeroGrads, WithGrads, UpdateParams)]
#[cfg_attr(feature = "safetensors", derive(SaveSafeTensors, LoadSafeTensors))]
pub struct GeneralizedAdd<T, U> {
    #[module]
    #[cfg_attr(feature = "safetensors", serialize)]
    pub t: T,
    #[module]
    #[cfg_attr(feature = "safetensors", serialize)]
    pub u: U,
}

impl<E: Dtype, D: Device<E>, T: BuildOnDevice<E, D>, U: BuildOnDevice<E, D>> BuildOnDevice<E, D>
    for GeneralizedAdd<T, U>
{
    type Built = GeneralizedAdd<T::Built, U::Built>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, crate::tensor::Error> {
        let t = self.t.try_build_on_device(device)?;
        let u = self.u.try_build_on_device(device)?;
        Ok(GeneralizedAdd { t, u })
    }
}

impl<X: WithEmptyTape, T: Module<X>, U: Module<X>> Module<X> for GeneralizedAdd<T, U>
where
    T::Output: TryAdd<U::Output>,
{
    type Output = <T::Output as TryAdd<U::Output>>::Output;
    fn try_forward(&self, x: X) -> Result<Self::Output, Error> {
        let t = self.t.try_forward(x.with_empty_tape())?;
        let u = self.u.try_forward(x)?;
        t.try_add(u)
    }

    fn try_forward_mut(&mut self, x: X) -> Result<Self::Output, Error> {
        let t = self.t.try_forward_mut(x.with_empty_tape())?;
        let u = self.u.try_forward_mut(x)?;
        t.try_add(u)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::*;

    #[test]
    fn test_generalized_add_gradients() {
        let dev: TestDevice = Default::default();

        type Model = GeneralizedAdd<LinearConstConfig<2, 2>, LinearConstConfig<2, 2>>;
        let model = dev.build_module::<f32>(Model::default());
        let model = GeneralizedAdd {
            t: Linear {
                weight: model.t.weight.to_dtype::<TestDtype>(),
                bias: model.t.bias.to_dtype::<TestDtype>(),
            },
            u: Linear {
                weight: model.u.weight.to_dtype::<TestDtype>(),
                bias: model.u.bias.to_dtype::<TestDtype>(),
            },
        };

        let x: Tensor<Rank2<4, 2>, f32, _> = dev.sample_normal();
        let x = x.to_dtype::<TestDtype>();
        let y = model.forward(x.leaky_trace());

        #[rustfmt::skip]
        assert_close_to_literal!(y, [[-0.81360567, -1.1473482], [1.0925694, 0.17383915], [-0.32519114, 0.49806428], [0.08259219, -0.7277866]]);

        let g = y.mean().backward();
        assert_close_to_literal!(g.get(&x), [[0.15889636, 0.062031522]; 4]);
        assert_close_to_literal!(g.get(&model.t.weight), [[-0.025407, 0.155879]; 2]);
        assert_close_to_literal!(g.get(&model.t.bias), [0.5; 2]);
        assert_close_to_literal!(g.get(&model.u.weight), [[-0.025407, 0.155879]; 2]);
        assert_close_to_literal!(g.get(&model.u.bias), [0.5; 2]);
    }
}
