use crate::{optim::*, shapes::Dtype, tensor::*, tensor_ops::Device};

use super::{BuildModule, BuildOnDevice, Module, ModuleMut, ResetParams, ToDevice};

/// Splits input into multiple heads. `T` should be a tuple,
/// where every element of the tuple accepts the same input type.
///
/// This provides a utility for multi headed structures where
/// the tape needs to be moved around a number of times.
///
/// # Generics
/// - `T` the module to split the input into.
///
/// # Examples
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// type Model = SplitInto<(Linear<5, 3>, Linear<5, 7>)>;
/// let model = dev.build_module::<Model, f32>();
/// let _: (Tensor<Rank1<3>, f32, _>, Tensor<Rank1<7>, f32, _>) = model.forward(dev.zeros::<Rank1<5>>());
/// ```
#[derive(Debug, Default, Clone)]
pub struct SplitInto<T>(pub T);

impl<T: GradientUpdate<D, E>, D: Device<E>, E: Dtype> GradientUpdate<D, E> for SplitInto<T> {
    fn update<U>(&mut self, updater: &mut U, unused: &mut UnusedTensors) -> Result<(), <D>::Err>
    where
        U: ParamUpdater<D, E>,
    {
        self.0.update(updater, unused)
    }
}

impl<T: BuildOnDevice<D, E>, D: Device<E>, E: Dtype> BuildOnDevice<D, E> for SplitInto<T> {
    type Built = SplitInto<T::Built>;
}

impl<T: BuildModule<D, E>, D: Device<E>, E: Dtype> BuildModule<D, E> for SplitInto<T> {
    fn try_build(device: &D) -> Result<Self, <D>::Err> {
        Ok(Self(BuildModule::try_build(device)?))
    }
}

impl<T: ResetParams<D, E>, D: Device<E>, E: Dtype> ResetParams<D, E> for SplitInto<T> {
    fn try_reset_params(&mut self) -> Result<(), <D>::Err> {
        self.0.try_reset_params()
    }
}

impl<T: ToDevice<D>, D> ToDevice<D> for SplitInto<T> {
    type Output = SplitInto<T::Output>;

    fn to_device(&self, device: &D) -> Self::Output {
        SplitInto(self.0.to_device(device))
    }
}

macro_rules! tuple_impls {
    ([$($heads:ident),+] $tail:ident) => {
impl<
    Input: SplitTape,
    $($heads : Module<Input>,)+
    $tail: Module<Input>
> Module<Input> for SplitInto<($($heads,)+ $tail)>
where
    $($heads::Output: SplitTape<Tape = Input::Tape>,)+
{
    type Output = (
        $(<$heads::Output as SplitTape>::NoTape, )+
        $tail::Output
    );

    #[allow(non_snake_case)]
    fn forward(&self, x: Input) -> Self::Output {
        let (x, tape) = x.split_tape();
        let ($($heads, )+ $tail) = &self.0;
        $(let ($heads, tape) = $heads.forward(x.clone().put_tape(tape)).split_tape();)+
        let $tail = $tail.forward(x.put_tape(tape));
        ($($heads,)+ $tail)
    }
}

impl<
    Input: SplitTape,
    $($heads : ModuleMut<Input>,)+
    $tail: ModuleMut<Input>
> ModuleMut<Input> for SplitInto<($($heads,)+ $tail)>
where
    $($heads::Output: SplitTape<Tape = Input::Tape>,)+
{
    type Output = (
        $(<$heads::Output as SplitTape>::NoTape, )+
        $tail::Output
    );

    #[allow(non_snake_case)]
    fn forward_mut(&mut self, x: Input) -> Self::Output {
        let (x, tape) = x.split_tape();
        let ($($heads, )+ $tail) = &mut self.0;
        $(let ($heads, tape) = $heads.forward_mut(x.clone().put_tape(tape)).split_tape();)+
        let $tail = $tail.forward_mut(x.put_tape(tape));
        ($($heads,)+ $tail)
    }
}
}
}

tuple_impls!([A] B);
tuple_impls!([A, B] C);
tuple_impls!([A, B, C] D);
tuple_impls!([A, B, C, D] E);
tuple_impls!([A, B, C, D, E] F);

#[cfg(test)]
mod tests {
    #![allow(clippy::type_complexity)]

    use super::*;
    use crate::nn::DeviceBuildExt;
    use crate::{gradients::*, shapes::*, tensor_ops::*};
    use crate::{
        nn::{builders::Linear, tests::SimpleUpdater},
        tests::*,
        unique_id::HasUniqueId,
    };

    #[test]
    fn test_unused() {
        let dev: TestDevice = Default::default();
        type Model = SplitInto<(Linear<1, 1>, Linear<1, 1>)>;
        let m = dev.build_module::<Model, TestDtype>();
        let (left, right) = m.forward(dev.sample_normal::<Rank1<1>>().trace());
        let r = right.retaped::<NoneTape>();
        let g = right.mean().backward();
        assert_eq!(g.get(&left).array(), [0.0; 1]);
        assert_ne!(g.get(&r).array(), [0.0; 1]);
    }

    #[test]
    fn test_split_into_2() {
        let dev: TestDevice = Default::default();
        type Model = SplitInto<(Linear<5, 1>, Linear<5, 2>)>;
        let m = dev.build_module::<Model, TestDtype>();
        let _: (Tensor<Rank1<1>, _, _>, Tensor<Rank1<2>, _, _, OwnedTape<_>>) =
            m.forward(dev.zeros::<Rank1<5>>().traced());
        let _: (
            Tensor<Rank2<3, 1>, _, _>,
            Tensor<Rank2<3, 2>, _, _, OwnedTape<_>>,
        ) = m.forward(dev.zeros::<Rank2<3, 5>>().traced());
    }

    #[test]
    fn test_split_into_3() {
        let dev: TestDevice = Default::default();
        type Model = SplitInto<(Linear<5, 1>, Linear<5, 2>, Linear<5, 3>)>;
        let m = dev.build_module::<Model, TestDtype>();
        let _: (
            Tensor<Rank1<1>, _, _>,
            Tensor<Rank1<2>, _, _>,
            Tensor<Rank1<3>, _, _, OwnedTape<_>>,
        ) = m.forward(dev.zeros::<Rank1<5>>().traced());
        let _: (
            Tensor<Rank2<3, 1>, _, _>,
            Tensor<Rank2<3, 2>, _, _>,
            Tensor<Rank2<3, 3>, _, _, OwnedTape<_>>,
        ) = m.forward(dev.zeros::<Rank2<3, 5>>().traced());
    }

    #[test]
    fn test_split_into_4() {
        let dev: TestDevice = Default::default();
        type Model = SplitInto<(Linear<5, 1>, Linear<5, 2>, Linear<5, 3>, Linear<5, 4>)>;
        let m = dev.build_module::<Model, TestDtype>();
        let _: (
            Tensor<Rank1<1>, _, _>,
            Tensor<Rank1<2>, _, _>,
            Tensor<Rank1<3>, _, _>,
            Tensor<Rank1<4>, _, _, OwnedTape<_>>,
        ) = m.forward(dev.zeros::<Rank1<5>>().traced());
        let _: (
            Tensor<Rank2<3, 1>, _, _>,
            Tensor<Rank2<3, 2>, _, _>,
            Tensor<Rank2<3, 3>, _, _>,
            Tensor<Rank2<3, 4>, _, _, OwnedTape<_>>,
        ) = m.forward(dev.zeros::<Rank2<3, 5>>().traced());
    }

    #[test]
    fn test_split_into_5() {
        let dev: TestDevice = Default::default();
        type Model = SplitInto<(
            Linear<5, 1>,
            Linear<5, 2>,
            Linear<5, 3>,
            Linear<5, 4>,
            Linear<5, 5>,
        )>;
        let m = dev.build_module::<Model, TestDtype>();
        let _: (
            Tensor<Rank1<1>, _, _>,
            Tensor<Rank1<2>, _, _>,
            Tensor<Rank1<3>, _, _>,
            Tensor<Rank1<4>, _, _>,
            Tensor<Rank1<5>, _, _, OwnedTape<_>>,
        ) = m.forward(dev.zeros::<Rank1<5>>().traced());
        let _: (
            Tensor<Rank2<3, 1>, _, _>,
            Tensor<Rank2<3, 2>, _, _>,
            Tensor<Rank2<3, 3>, _, _>,
            Tensor<Rank2<3, 4>, _, _>,
            Tensor<Rank2<3, 5>, _, _, OwnedTape<_>>,
        ) = m.forward(dev.zeros::<Rank2<3, 5>>().traced());
    }

    #[test]
    fn test_split_into_6() {
        let dev: TestDevice = Default::default();
        type Model = SplitInto<(
            Linear<5, 1>,
            Linear<5, 2>,
            Linear<5, 3>,
            Linear<5, 4>,
            Linear<5, 5>,
            Linear<5, 6>,
        )>;
        let m = dev.build_module::<Model, TestDtype>();
        let _: (
            Tensor<Rank1<1>, _, _>,
            Tensor<Rank1<2>, _, _>,
            Tensor<Rank1<3>, _, _>,
            Tensor<Rank1<4>, _, _>,
            Tensor<Rank1<5>, _, _>,
            Tensor<Rank1<6>, _, _, OwnedTape<_>>,
        ) = m.forward(dev.zeros::<Rank1<5>>().traced());
        let _: (
            Tensor<Rank2<3, 1>, _, _>,
            Tensor<Rank2<3, 2>, _, _>,
            Tensor<Rank2<3, 3>, _, _>,
            Tensor<Rank2<3, 4>, _, _>,
            Tensor<Rank2<3, 5>, _, _>,
            Tensor<Rank2<3, 6>, _, _, OwnedTape<_>>,
        ) = m.forward(dev.zeros::<Rank2<3, 5>>().traced());
    }

    #[test]
    fn test_missing_gradients() {
        let dev: TestDevice = Default::default();
        type Model = SplitInto<(Linear<5, 3>, Linear<5, 3>)>;
        let mut model = dev.build_module::<Model, TestDtype>();
        let mut g: SimpleUpdater = Default::default();

        // no gradients present
        let mut unused = Default::default();
        model.update(&mut g, &mut unused).unwrap();
        assert_eq!(
            &unused.ids,
            &[
                *model.0 .0.weight.id(),
                *model.0 .0.bias.id(),
                *model.0 .1.weight.id(),
                *model.0 .1.bias.id()
            ]
        );

        // weight gradient is present
        g.0.try_alloc_for(&model.0 .0.weight).unwrap();
        g.0.try_alloc_for(&model.0 .0.bias).unwrap();
        g.0.try_alloc_for(&model.0 .1.weight).unwrap();
        g.0.try_alloc_for(&model.0 .1.bias).unwrap();

        let mut unused = Default::default();
        model.update(&mut g, &mut unused).unwrap();
        assert!(unused.is_empty());
    }
}
