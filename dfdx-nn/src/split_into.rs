use crate::*;

use dfdx::{shapes::Dtype, tensor::WithEmptyTape, tensor_ops::Device};

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
