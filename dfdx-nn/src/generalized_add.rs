use crate::*;
use dfdx::{
    shapes::Dtype,
    tensor::WithEmptyTape,
    tensor_ops::{Device, TryAdd},
};

#[derive(
    Default, Clone, Debug, ResetParams, ZeroGrads, UpdateParams, LoadSafeTensors, SaveSafeTensors,
)]
pub struct GeneralizedAdd<T, U>(
    #[module]
    #[serialize]
    pub T,
    #[module]
    #[serialize]
    pub U,
);

impl<E: Dtype, D: Device<E>, T: BuildOnDevice<E, D>, U: BuildOnDevice<E, D>> BuildOnDevice<E, D>
    for GeneralizedAdd<T, U>
{
    type Built = GeneralizedAdd<T::Built, U::Built>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, <D>::Err> {
        let t = self.0.try_build_on_device(device)?;
        let u = self.1.try_build_on_device(device)?;
        Ok(GeneralizedAdd(t, u))
    }
}

impl<X: WithEmptyTape, T: Module<X>, U: Module<X, Error = T::Error>> Module<X>
    for GeneralizedAdd<T, U>
where
    T::Output: TryAdd<U::Output, Err = T::Error>,
{
    type Output = <T::Output as TryAdd<U::Output>>::Output;
    type Error = T::Error;
    fn try_forward(&self, x: X) -> Result<Self::Output, Self::Error> {
        let t = self.0.try_forward(x.with_empty_tape())?;
        let u = self.1.try_forward(x)?;
        t.try_add(u)
    }

    fn try_forward_mut(&mut self, x: X) -> Result<Self::Output, Self::Error> {
        let t = self.0.try_forward_mut(x.with_empty_tape())?;
        let u = self.1.try_forward_mut(x)?;
        t.try_add(u)
    }
}
