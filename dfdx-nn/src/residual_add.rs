use crate::*;
use dfdx::{
    shapes::Dtype,
    tensor::WithEmptyTape,
    tensor_ops::{Device, TryAdd},
};

use crate::Module;

#[derive(
    Default, Clone, Debug, ResetParams, ZeroGrads, UpdateParams, SaveSafeTensors, LoadSafeTensors,
)]
#[repr(transparent)]
pub struct ResidualAdd<T>(
    #[module]
    #[serialize]
    pub T,
);

// TODO derive this
impl<E: Dtype, D: Device<E>, T: BuildOnDevice<E, D>> BuildOnDevice<E, D> for ResidualAdd<T> {
    type Built = ResidualAdd<T::Built>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, <D>::Err> {
        let t = self.0.try_build_on_device(device)?;
        Ok(ResidualAdd(t))
    }
}

impl<X: WithEmptyTape, T: Module<X>> Module<X> for ResidualAdd<T>
where
    X: TryAdd<T::Output, Err = T::Error>,
{
    type Output = X::Output;
    type Error = T::Error;
    fn try_forward(&self, x: X) -> Result<Self::Output, Self::Error> {
        let y = self.0.try_forward(x.with_empty_tape())?;
        x.try_add(y)
    }
    fn try_forward_mut(&mut self, x: X) -> Result<Self::Output, Self::Error> {
        let y = self.0.try_forward_mut(x.with_empty_tape())?;
        x.try_add(y)
    }
}
