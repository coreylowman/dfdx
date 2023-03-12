use crate::{prelude::Device, shapes::Dtype};

use super::*;

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
/// type Model = Repeated<(Linear<10, 10>, ReLU), 5>;
/// let model = dev.build_module::<Model, f32>();
/// let out: Tensor<Rank1<10>, f32, _> = model.forward(dev.zeros());
/// ```
#[derive(Debug, Clone)]
pub struct Repeated<T, const N: usize> {
    pub modules: std::vec::Vec<T>,
}

impl<D: Device<E>, E: Dtype, T: BuildOnDevice<D, E>, const N: usize> BuildOnDevice<D, E>
    for Repeated<T, N>
{
    type Built = Repeated<T::Built, N>;
}

impl<E: Dtype, D: Device<E>, T: TensorCollection<E, D>, const N: usize> TensorCollection<E, D>
    for Repeated<T, N>
{
    type Output<E2: Dtype, D2: Device<E2>> = Repeated<T::Output<E2, D2>, N>;

    fn iter_tensors<E2: Dtype, D2: Device<E2>, V: ModuleVisitor<Self, E, D, E2, D2>>(
        visitor: &mut V,
    ) -> ModuleVisitorOutput<V::Func, Self, E, D, E2, D2> {
        let mut tmp = Vec::new();

        for i in 0..N {
            tmp.push(visitor.visit_module(
                &std::format!("{i}"),
                |s| &s.modules[i],
                |s| &mut s.modules[i],
            )?);
        }

        let mut modules = Vec::new();

        for x in tmp {
            if let Some(x) = x {
                modules.push(x);
            } else {
                return Ok(None);
            }
        }

        Ok(Some(Repeated { modules }))
    }
}

impl<T, const N: usize> std::ops::Index<usize> for Repeated<T, N> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.modules[index]
    }
}

impl<Input, T: Module<Input, Output = Input>, const N: usize> Module<Input> for Repeated<T, N> {
    type Output = T::Output;
    type Error = T::Error;

    fn try_forward(&self, mut x: Input) -> Result<Self::Output, T::Error> {
        for i in 0..N {
            x = self.modules[i].try_forward(x)?;
        }
        Ok(x)
    }
}

impl<Input, T: ModuleMut<Input, Output = Input>, const N: usize> ModuleMut<Input>
    for Repeated<T, N>
{
    type Output = T::Output;
    type Error = T::Error;

    fn try_forward_mut(&mut self, mut x: Input) -> Result<Self::Output, T::Error> {
        for i in 0..N {
            x = self.modules[i].try_forward_mut(x)?;
        }
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::*;
    use crate::{nn::builders::*, shapes::*};

    #[test]
    fn test_default_and_reset() {
        let dev: TestDevice = Default::default();

        type Model = Repeated<(Linear<3, 3>, ReLU), 5>;
        let m = dev.build_module::<Model, TestDtype>();

        for i in 0..5 {
            assert_ne!(m.modules[i].0.weight.array(), [[0.0; 3]; 3]);
            assert_ne!(m.modules[i].0.bias.array(), [0.0; 3]);
        }
    }

    #[test]
    fn test_forward() {
        let dev: TestDevice = Default::default();

        type Model = Repeated<(Linear<3, 3>, ReLU), 5>;
        let mut m = dev.build_module::<Model, TestDtype>();

        let x = dev.zeros::<Rank1<3>>();
        let x = m.modules[0].forward(x);
        let x = m.modules[1].forward(x);
        let x = m.modules[2].forward(x);
        let x = m.modules[3].forward(x);
        let x = m.modules[4].forward(x);

        assert_eq!(x.array(), m.forward_mut(dev.zeros::<Rank1<3>>()).array());
    }
}
