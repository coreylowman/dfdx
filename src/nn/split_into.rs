use crate::{shapes::Dtype, tensor::*, tensor_ops::Device};

use super::*;

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

impl<T: BuildOnDevice<D, E>, D: Device<E>, E: Dtype> BuildOnDevice<D, E> for SplitInto<T> {
    type Built = SplitInto<T::Built>;
}

impl<E: Dtype, D: Device<E>, T: TensorCollection<E, D>> TensorCollection<E, D> for SplitInto<T> {
    type To<E2: Dtype, D2: Device<E2>> = SplitInto<T::To<E2, D2>>;

    fn iter_tensors<V: ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(Self::module("0", |s| &s.0, |s| &mut s.0), SplitInto)
    }
}

macro_rules! tuple_impls {
    ([$($heads:ident),+] $tail:ident) => {
impl<
    Input: SplitTape,
    $($heads : Module<Input, Error = $tail::Error>,)+
    $tail: Module<Input>
> Module<Input> for SplitInto<($($heads,)+ $tail)>
where
    $($heads::Output: SplitTape<Tape = Input::Tape>,)+
{
    type Output = (
        $(<$heads::Output as SplitTape>::NoTape, )+
        $tail::Output
    );
    type Error = $tail::Error;

    #[allow(non_snake_case)]
    fn try_forward(&self, x: Input) -> Result<Self::Output, $tail::Error> {
        let (x, tape) = x.split_tape();
        let ($($heads, )+ $tail) = &self.0;
        $(let ($heads, tape) = $heads.try_forward(x.clone().put_tape(tape))?.split_tape();)+
        let $tail = $tail.try_forward(x.put_tape(tape))?;
        Ok(($($heads,)+ $tail))
    }
}

impl<
    Input: SplitTape,
    $($heads : ModuleMut<Input, Error = $tail::Error>,)+
    $tail: ModuleMut<Input>
> ModuleMut<Input> for SplitInto<($($heads,)+ $tail)>
where
    $($heads::Output: SplitTape<Tape = Input::Tape>,)+
{
    type Output = (
        $(<$heads::Output as SplitTape>::NoTape, )+
        $tail::Output
    );
    type Error = $tail::Error;

    #[allow(non_snake_case)]
    fn try_forward_mut(&mut self, x: Input) -> Result<Self::Output, $tail::Error> {
        let (x, tape) = x.split_tape();
        let ($($heads, )+ $tail) = &mut self.0;
        $(let ($heads, tape) = $heads.try_forward_mut(x.clone().put_tape(tape))?.split_tape();)+
        let $tail = $tail.try_forward_mut(x.put_tape(tape))?;
        Ok(($($heads,)+ $tail))
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
    use crate::{nn::builders::Linear, tests::*};
    use crate::{shapes::*, tensor_ops::*};

    #[test]
    fn test_unused() {
        let dev: TestDevice = Default::default();
        type Model = SplitInto<(Linear<1, 1>, Linear<1, 1>)>;
        let m = dev.build_module::<Model, TestDtype>();
        let (left, right) = m.forward(dev.sample_normal::<Rank1<1>>().leaky_trace());
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
        let _: (
            Tensor<Rank1<1>, _, _>,
            Tensor<Rank1<2>, _, _, OwnedTape<_, _>>,
        ) = m.forward(dev.zeros::<Rank1<5>>().leaky_traced());
        let _: (
            Tensor<Rank2<3, 1>, _, _>,
            Tensor<Rank2<3, 2>, _, _, OwnedTape<_, _>>,
        ) = m.forward(dev.zeros::<Rank2<3, 5>>().leaky_traced());
    }

    #[test]
    fn test_split_into_3() {
        let dev: TestDevice = Default::default();
        type Model = SplitInto<(Linear<5, 1>, Linear<5, 2>, Linear<5, 3>)>;
        let m = dev.build_module::<Model, TestDtype>();
        let _: (
            Tensor<Rank1<1>, _, _>,
            Tensor<Rank1<2>, _, _>,
            Tensor<Rank1<3>, _, _, OwnedTape<_, _>>,
        ) = m.forward(dev.zeros::<Rank1<5>>().leaky_traced());
        let _: (
            Tensor<Rank2<3, 1>, _, _>,
            Tensor<Rank2<3, 2>, _, _>,
            Tensor<Rank2<3, 3>, _, _, OwnedTape<_, _>>,
        ) = m.forward(dev.zeros::<Rank2<3, 5>>().leaky_traced());
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
            Tensor<Rank1<4>, _, _, OwnedTape<_, _>>,
        ) = m.forward(dev.zeros::<Rank1<5>>().leaky_traced());
        let _: (
            Tensor<Rank2<3, 1>, _, _>,
            Tensor<Rank2<3, 2>, _, _>,
            Tensor<Rank2<3, 3>, _, _>,
            Tensor<Rank2<3, 4>, _, _, OwnedTape<_, _>>,
        ) = m.forward(dev.zeros::<Rank2<3, 5>>().leaky_traced());
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
            Tensor<Rank1<5>, _, _, OwnedTape<_, _>>,
        ) = m.forward(dev.zeros::<Rank1<5>>().leaky_traced());
        let _: (
            Tensor<Rank2<3, 1>, _, _>,
            Tensor<Rank2<3, 2>, _, _>,
            Tensor<Rank2<3, 3>, _, _>,
            Tensor<Rank2<3, 4>, _, _>,
            Tensor<Rank2<3, 5>, _, _, OwnedTape<_, _>>,
        ) = m.forward(dev.zeros::<Rank2<3, 5>>().leaky_traced());
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
            Tensor<Rank1<6>, _, _, OwnedTape<_, _>>,
        ) = m.forward(dev.zeros::<Rank1<5>>().leaky_traced());
        let _: (
            Tensor<Rank2<3, 1>, _, _>,
            Tensor<Rank2<3, 2>, _, _>,
            Tensor<Rank2<3, 3>, _, _>,
            Tensor<Rank2<3, 4>, _, _>,
            Tensor<Rank2<3, 5>, _, _>,
            Tensor<Rank2<3, 6>, _, _, OwnedTape<_, _>>,
        ) = m.forward(dev.zeros::<Rank2<3, 5>>().leaky_traced());
    }
}
