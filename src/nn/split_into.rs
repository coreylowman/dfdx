use crate::{
    arrays::Dtype,
    optim::{CanUpdateWithGradients, UnusedTensors, UpdateParams},
    tensor::{PutTape, SplitTape},
    tensor_ops::Device,
};

use super::{BuildModule, Module, ModuleMut};

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
/// type Model = SplitInto<(Linear<5, 3>, Linear<5, 7>)>;
/// let model: Model = Default::default();
/// let _: (Tensor<Rank1<3>, _, _>, Tensor<Rank1<7>, _, _>) = model.forward(Tensor:Rank1<:>, _, _<5>::zeros());
/// ```
#[derive(Debug, Default, Clone)]
pub struct SplitInto<T>(pub T);

impl<T: CanUpdateWithGradients<D, E>, D: Device<E>, E: Dtype> CanUpdateWithGradients<D, E>
    for SplitInto<T>
{
    fn update<U>(&mut self, updater: &mut U, unused: &mut UnusedTensors) -> Result<(), <D>::Err>
    where
        U: UpdateParams<D, E>,
    {
        self.0.update(updater, unused)
    }
}

impl<T: BuildModule<D, E>, D: Device<E>, E: Dtype> BuildModule<D, E> for SplitInto<T> {
    fn try_build(device: &D) -> Result<Self, <D>::Err> {
        Ok(Self(BuildModule::try_build(device)?))
    }
    fn try_reset_params(&mut self) -> Result<(), <D>::Err> {
        self.0.try_reset_params()
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
    use crate::{arrays::*, gradients::*, tensor::*, tensor_ops::*};
    use crate::{
        nn::{tests::SimpleUpdater, Linear},
        tests::build_test_device,
        unique_id::HasUniqueId,
    };

    #[test]
    fn test_unused() {
        let dev = build_test_device!();
        let m: SplitInto<(Linear<1, 1, _>, Linear<1, 1, _>)> = BuildModule::build(&dev);
        let (left, right) = m.forward(dev.randn::<Rank1<1>>().trace());
        let r = right.retaped::<NoneTape>();
        let g = right.mean().backward();
        assert_eq!(g.get(&left).array(), [0.0; 1]);
        assert_ne!(g.get(&r).array(), [0.0; 1]);
    }

    #[test]
    fn test_split_into_2() {
        let dev = build_test_device!();
        type Model = SplitInto<(Linear<5, 1>, Linear<5, 2>)>;
        let m: Model = BuildModule::build(&dev);
        let _: (Tensor<Rank1<1>, _, _>, Tensor<Rank1<2>, _, _, OwnedTape<_>>) =
            m.forward(dev.zeros::<Rank1<5>>().traced());
        let _: (
            Tensor<Rank2<3, 1>, _, _>,
            Tensor<Rank2<3, 2>, _, _, OwnedTape<_>>,
        ) = m.forward(dev.zeros::<Rank2<3, 5>>().traced());
    }

    #[test]
    fn test_split_into_3() {
        let dev = build_test_device!();
        type Model = SplitInto<(Linear<5, 1>, Linear<5, 2>, Linear<5, 3>)>;
        let m: Model = BuildModule::build(&dev);
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
        type Model = SplitInto<(Linear<5, 1>, Linear<5, 2>, Linear<5, 3>, Linear<5, 4>)>;
        let dev = build_test_device!();
        let m: Model = BuildModule::build(&dev);
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
        type Model = SplitInto<(
            Linear<5, 1>,
            Linear<5, 2>,
            Linear<5, 3>,
            Linear<5, 4>,
            Linear<5, 5>,
        )>;
        let dev = build_test_device!();
        let m: Model = BuildModule::build(&dev);
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
        type Model = SplitInto<(
            Linear<5, 1>,
            Linear<5, 2>,
            Linear<5, 3>,
            Linear<5, 4>,
            Linear<5, 5>,
            Linear<5, 6>,
        )>;
        let dev = build_test_device!();
        let m: Model = BuildModule::build(&dev);
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
        let dev = build_test_device!();
        let mut model: SplitInto<(Linear<5, 3, _>, Linear<5, 3, _>)> = BuildModule::build(&dev);
        let mut g: SimpleUpdater<_> = Default::default();

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
