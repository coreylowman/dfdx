use crate::prelude::*;

/// Splits input into multiple heads. `T` should be a tuple,
/// where every element of the tuple accepts the same input type.
///
/// This provides a utility for multi headed structures where
/// the tape needs to be moved around a number of times.
///
/// Each head's operations will be stored in its output's tape, while the operations stored in the
/// input tape will be saved in the first output's tape.
///
/// # Generics
/// - `T` the module to split the input into.
///
/// # Examples
/// ```rust
/// # use dfdx::prelude::*;
/// # use dfdx::*;
/// # let dev: Cpu = Default::default();
/// type Model = SplitInto<(LinearConstConfig<5, 3>, LinearConstConfig<5, 7>)>;
/// let model = dev.build_module::<f32>(Model::default());
/// let _: (Tensor<Rank1<3>, f32, _>, Tensor<Rank1<7>, f32, _>) = model.forward(dev.zeros::<Rank1<5>>());
/// ```
#[derive(Debug, Default, Clone, ResetParams, ZeroGrads, WithGrads, UpdateParams)]
#[cfg_attr(feature = "safetensors", derive(SaveSafeTensors, LoadSafeTensors))]
#[repr(transparent)]
pub struct SplitInto<T>(
    #[module]
    #[cfg_attr(feature = "safetensors", serialize)]
    pub T,
);

impl<E: Dtype, D: Device<E>, T: BuildOnDevice<E, D>> BuildOnDevice<E, D> for SplitInto<T> {
    type Built = SplitInto<T::Built>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, crate::tensor::Error> {
        let t = self.0.try_build_on_device(device)?;
        Ok(SplitInto(t))
    }
}

macro_rules! tuple_impls {
    ($head:ident [$($tails:ident),+]) => {
        impl<
            Input: WithEmptyTape,
            $head: Module<Input>,
            $($tails : Module<Input>,)+
        > Module<Input> for SplitInto<($head, $($tails,)+)> {
            type Output = (
                $head::Output,
                $($tails::Output),+
            );

            #[allow(non_snake_case)]
            fn try_forward(&self, x: Input) -> Result<Self::Output, Error> {
                let ($head, $($tails,)+) = &self.0;
                let ($($tails,)+) = ($($tails.try_forward(x.with_empty_tape())?,)+);
                let $head = $head.try_forward(x)?;
                Ok(($head, $($tails,)+))
            }

            #[allow(non_snake_case)]
            fn try_forward_mut(&mut self, x: Input) -> Result<Self::Output, Error> {
                let ($head, $($tails,)+) = &mut self.0;
                let ($($tails,)+) = ($($tails.try_forward_mut(x.with_empty_tape())?,)+);
                let $head = $head.try_forward_mut(x)?;
                Ok(($head, $($tails,)+))
            }
        }
    }
}

tuple_impls!(A[B]);
tuple_impls!(A [B, C]);
tuple_impls!(A [B, C, D]);
tuple_impls!(A [B, C, D, E]);
tuple_impls!(A [B, C, D, E, F]);

#[cfg(test)]
mod tests {
    #![allow(clippy::type_complexity)]

    use super::*;
    use crate::tests::*;

    #[test]
    fn test_split_into_2() {
        let dev: TestDevice = Default::default();
        type Model = SplitInto<(LinearConstConfig<5, 1>, LinearConstConfig<5, 2>)>;
        let m = dev.build_module::<TestDtype>(Model::default());
        let _: (
            Tensor<Rank1<1>, _, _, OwnedTape<_, _>>,
            Tensor<Rank1<2>, _, _, OwnedTape<_, _>>,
        ) = m.forward(dev.zeros::<Rank1<5>>().leaky_traced());
        let _: (
            Tensor<Rank2<3, 1>, _, _, OwnedTape<_, _>>,
            Tensor<Rank2<3, 2>, _, _, OwnedTape<_, _>>,
        ) = m.forward(dev.zeros::<Rank2<3, 5>>().leaky_traced());
    }

    #[test]
    fn test_split_into_3() {
        let dev: TestDevice = Default::default();
        type Model = SplitInto<(
            LinearConstConfig<5, 1>,
            LinearConstConfig<5, 2>,
            LinearConstConfig<5, 3>,
        )>;
        let m = dev.build_module::<TestDtype>(Model::default());
        let _: (
            Tensor<Rank1<1>, _, _, OwnedTape<_, _>>,
            Tensor<Rank1<2>, _, _, OwnedTape<_, _>>,
            Tensor<Rank1<3>, _, _, OwnedTape<_, _>>,
        ) = m.forward(dev.zeros::<Rank1<5>>().leaky_traced());
        let _: (
            Tensor<Rank2<3, 1>, _, _, OwnedTape<_, _>>,
            Tensor<Rank2<3, 2>, _, _, OwnedTape<_, _>>,
            Tensor<Rank2<3, 3>, _, _, OwnedTape<_, _>>,
        ) = m.forward(dev.zeros::<Rank2<3, 5>>().leaky_traced());
    }
}
