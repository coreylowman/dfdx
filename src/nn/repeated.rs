use crate::{optim::*, shapes::Dtype, tensor_ops::Device};

use super::{Module, ModuleMut, OnDevice, ResetParams, ToDevice};

/// Repeats `T` `N` times. This requires that `T`'s input is the same as it's output.
///
/// # Generics
/// - `T` the [Module] to repeat
/// - `N` the number of times to repeat `T`.
///
/// # Examples
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let model: Repeated<(Linear<10, 10>, ReLU), 5> = dev.build_module();
/// let out: Tensor<Rank1<10>> = model.forward(dev.zeros());
/// ```
#[derive(Debug, Clone)]
pub struct Repeated<T, const N: usize> {
    pub modules: std::vec::Vec<T>,
}

impl<D: Device<E>, E: Dtype, T: ResetParams<D, E>, const N: usize> ResetParams<D, E>
    for Repeated<T, N>
{
    fn try_build(device: &D) -> Result<Self, <D>::Err> {
        let mut modules = std::vec::Vec::with_capacity(N);
        for _ in 0..N {
            modules.push(ResetParams::try_build(device)?);
        }
        Ok(Self { modules })
    }

    fn try_reset_params(&mut self) -> Result<(), <D>::Err> {
        for m in self.modules.iter_mut() {
            m.try_reset_params()?;
        }
        Ok(())
    }
}

impl<T, const N: usize> std::ops::Index<usize> for Repeated<T, N> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.modules[index]
    }
}

impl<D: Device<E>, E: Dtype, T: GradientUpdate<D, E>, const N: usize> GradientUpdate<D, E>
    for Repeated<T, N>
{
    fn update<U>(&mut self, updater: &mut U, unused: &mut UnusedTensors) -> Result<(), <D>::Err>
    where
        U: ParamUpdater<D, E>,
    {
        for m in self.modules.iter_mut() {
            m.update(updater, unused)?;
        }
        Ok(())
    }
}

impl<T: ToDevice<D>, const N: usize, D> ToDevice<D> for Repeated<T, N> {
    type Output = Repeated<OnDevice<T, D>, N>;

    fn to_device(&self, device: &D) -> Self::Output {
        Repeated {
            modules: self
                .modules
                .iter()
                .map(|module| module.to_device(device))
                .collect(),
        }
    }
}

impl<Input, T: Module<Input, Output = Input>, const N: usize> Module<Input> for Repeated<T, N> {
    type Output = T::Output;
    fn forward(&self, mut x: Input) -> Self::Output {
        for i in 0..N {
            x = self.modules[i].forward(x);
        }
        x
    }
}

impl<Input, T: ModuleMut<Input, Output = Input>, const N: usize> ModuleMut<Input>
    for Repeated<T, N>
{
    type Output = T::Output;
    fn forward_mut(&mut self, mut x: Input) -> Self::Output {
        for i in 0..N {
            x = self.modules[i].forward_mut(x);
        }
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{nn::tests::SimpleUpdater, tests::TestDevice};
    use crate::{nn::*, shapes::*, tensor::*, unique_id::HasUniqueId};

    #[test]
    fn test_default_and_reset() {
        let dev: TestDevice = Default::default();

        let m: Repeated<(Linear<3, 3, _>, ReLU), 5> = dev.build_module();

        for i in 0..5 {
            assert_ne!(m.modules[i].0.weight.array(), [[0.0; 3]; 3]);
            assert_ne!(m.modules[i].0.bias.array(), [0.0; 3]);
        }
    }

    #[test]
    fn test_forward() {
        let dev: TestDevice = Default::default();

        let mut m: Repeated<(Linear<3, 3, _>, ReLU), 5> = dev.build_module();

        let x = dev.zeros::<Rank1<3>>();
        let x = m.modules[0].forward(x);
        let x = m.modules[1].forward(x);
        let x = m.modules[2].forward(x);
        let x = m.modules[3].forward(x);
        let x = m.modules[4].forward(x);

        assert_eq!(x.array(), m.forward_mut(dev.zeros::<Rank1<3>>()).array());
    }

    #[test]
    fn test_repeated_missing_gradients() {
        let dev: TestDevice = Default::default();

        let mut model: Repeated<Linear<5, 5, _>, 3> = dev.build_module();
        let mut g: SimpleUpdater<_> = Default::default();

        // no gradients present
        let mut unused = Default::default();
        model.update(&mut g, &mut unused).unwrap();
        assert_eq!(
            &unused.ids,
            &[
                *model[0].weight.id(),
                *model[0].bias.id(),
                *model[1].weight.id(),
                *model[1].bias.id(),
                *model[2].weight.id(),
                *model[2].bias.id(),
            ]
        );

        // weight gradient is present
        for i in 0..3 {
            g.0.try_alloc_for(&model[i].weight).unwrap();
        }

        let mut unused = Default::default();
        model.update(&mut g, &mut unused).unwrap();
        assert_eq!(
            &unused.ids,
            &[
                *model[0].bias.id(),
                *model[1].bias.id(),
                *model[2].bias.id()
            ]
        );

        // all gradients present
        for i in 0..3 {
            g.0.try_alloc_for(&model[i].weight).unwrap();
            g.0.try_alloc_for(&model[i].bias).unwrap();
        }

        let mut unused = Default::default();
        model.update(&mut g, &mut unused).unwrap();
        assert!(unused.is_empty());
    }
}
