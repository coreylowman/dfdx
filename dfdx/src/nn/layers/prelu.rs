use crate::prelude::*;

/// Calls [crate::tensor_ops::prelu()] with learnable value.
#[derive(Debug, Clone, Copy)]
pub struct PReLUConfig(pub f64);

impl Default for PReLUConfig {
    fn default() -> Self {
        Self(0.25)
    }
}

impl<E: Dtype, D: Device<E>> BuildOnDevice<E, D> for PReLUConfig {
    type Built = PReLU<E, D>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, crate::tensor::Error> {
        let a = device.try_tensor(E::from_f64(self.0).unwrap())?;
        Ok(PReLU { a })
    }
}

/// See [PReLUConfig].
#[derive(Clone, Debug, UpdateParams, ZeroGrads, WithGrads)]
#[cfg_attr(feature = "safetensors", derive(SaveSafeTensors, LoadSafeTensors))]
pub struct PReLU<Elem: Dtype, Dev: Device<Elem>> {
    #[param]
    #[cfg_attr(feature = "safetensors", serialize)]
    pub a: Tensor<(), Elem, Dev>,
}

impl<E: Dtype, D: Device<E>> ResetParams<E, D> for PReLU<E, D> {
    /// Does nothing.
    fn try_reset_params(&mut self) -> Result<(), crate::tensor::Error> {
        Ok(())
    }
}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<S, E, D, T>> for PReLU<E, D> {
    type Output = Tensor<S, E, D, T>;
    fn try_forward(&self, x: Tensor<S, E, D, T>) -> Result<Self::Output, Error> {
        let a = self.a.retaped::<T>().broadcast_like(&x);
        x.try_prelu(a)
    }
}
