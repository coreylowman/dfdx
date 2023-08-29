use crate::*;

use dfdx::{shapes::Dtype, tensor::WithEmptyTape, tensor_ops::Device};

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
/// # let dev: Cpu = Default::default();
/// type Model = SplitInto<(Linear<5, 3>, Linear<5, 7>)>;
/// let model = dev.build_module::<Model, f32>();
/// let _: (Tensor<Rank1<3>, f32, _>, Tensor<Rank1<7>, f32, _>) = model.forward(dev.zeros::<Rank1<5>>());
/// ```
#[derive(
    Debug, Default, Clone, ResetParams, ZeroGrads, UpdateParams, LoadSafeTensors, SaveSafeTensors,
)]
#[repr(transparent)]
pub struct SplitInto<T>(
    #[module]
    #[serialize]
    pub T,
);

impl<E: Dtype, D: Device<E>, T: BuildOnDevice<E, D>> BuildOnDevice<E, D> for SplitInto<T> {
    type Built = SplitInto<T::Built>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, <D>::Err> {
        let t = self.0.try_build_on_device(device)?;
        Ok(SplitInto(t))
    }
}

macro_rules! tuple_impls {
    ($head:ident [$($tails:ident),+]) => {
        impl<
            Input: WithEmptyTape,
            $head: Module<Input>,
            $($tails : Module<Input, Error = $head::Error>,)+
        > Module<Input> for SplitInto<($head, $($tails,)+)> {
            type Output = (
                $head::Output,
                $($tails::Output),+
            );
            type Error = $head::Error;

            #[allow(non_snake_case)]
            fn try_forward(&self, x: Input) -> Result<Self::Output, $head::Error> {
                let ($head, $($tails,)+) = &self.0;
                let ($($tails,)+) = ($($tails.try_forward(x.with_empty_tape())?,)+);
                let $head = $head.try_forward(x)?;

                Ok(($head, $($tails,)+))
            }

            #[allow(non_snake_case)]
            fn try_forward_mut(&mut self, x: Input) -> Result<Self::Output, $head::Error> {
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
