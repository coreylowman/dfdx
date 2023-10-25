use crate::prelude::*;

/// A residual connection around `T`: `T(x) * x`.
///
/// # Generics
/// - `T`: The underlying module to do a skip connection around.
///
/// # Examples
/// ```rust
/// # use dfdx::prelude::*;
/// # use dfdx_nn::*;
/// # let dev: Cpu = Default::default();
/// type Model = ResidualMul<ReLU>;
/// let model = dev.build_module::<f32>(Model::default());
/// let x = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
/// let y = model.forward(x);
/// assert_eq!(y.array(), [0.0, 0.0, 0.0, 1.0, 4.0]);
/// ```
#[derive(
    Default, Clone, Debug, ResetParams, ZeroGrads, UpdateParams, SaveSafeTensors, LoadSafeTensors,
)]
#[repr(transparent)]
pub struct ResidualMul<T>(
    #[module]
    #[serialize]
    pub T,
);

// TODO derive this
impl<E: Dtype, D: Device<E>, T: BuildOnDevice<E, D>> BuildOnDevice<E, D> for ResidualMul<T> {
    type Built = ResidualMul<T::Built>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, <D>::Err> {
        let t = self.0.try_build_on_device(device)?;
        Ok(ResidualMul(t))
    }
}

impl<X: WithEmptyTape, T: Module<X>> Module<X> for ResidualMul<T>
where
    T::Output: TryMul<X, Err = T::Error>,
{
    type Output = <T::Output as TryMul<X>>::Output;
    type Error = T::Error;
    fn try_forward(&self, x: X) -> Result<Self::Output, Self::Error> {
        let t = self.0.try_forward(x.with_empty_tape())?;
        t.try_mul(x)
    }
    fn try_forward_mut(&mut self, x: X) -> Result<Self::Output, Self::Error> {
        let t = self.0.try_forward_mut(x.with_empty_tape())?;
        t.try_mul(x)
    }
}
