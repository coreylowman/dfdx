use dfdx::{dtypes::Dtype, tensor::UniqueId, tensor_ops::Device};

impl<E: Dtype, D: Device<E>, T: crate::BuildOnDevice<E, D>> crate::BuildOnDevice<E, D> for Vec<T> {
    type Built = Vec<T::Built>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, <D>::Err> {
        self.iter()
            .map(|m_i| m_i.try_build_on_device(device))
            .collect()
    }
}

impl<E: Dtype, D: Device<E>, T: crate::ResetParams<E, D>> crate::ResetParams<E, D> for Vec<T> {
    fn try_reset_params(&mut self) -> Result<(), <D>::Err> {
        for m_i in self.iter_mut() {
            m_i.try_reset_params()?;
        }
        Ok(())
    }
}

impl<E: Dtype, D: Device<E>, T: crate::UpdateParams<E, D>> crate::UpdateParams<E, D> for Vec<T> {
    fn try_update_params<M, Optim: crate::Optimizer<M, E, D>>(
        &mut self,
        optimizer: &mut Optim,
        gradients: &dfdx::tensor::Gradients<E, D>,
        missing_tensors: &mut Vec<UniqueId>,
    ) -> Result<(), D::Err> {
        for m_i in self.iter_mut() {
            m_i.try_update_params(optimizer, gradients, missing_tensors)?;
        }
        Ok(())
    }
}

impl<E: Dtype, D: Device<E>, T: crate::ZeroGrads<E, D>> crate::ZeroGrads<E, D> for Vec<T> {
    fn try_zero_grads(&self, grads: &mut dfdx::tensor::Gradients<E, D>) -> Result<(), <D>::Err> {
        for m_i in self.iter() {
            m_i.try_zero_grads(grads)?;
        }
        Ok(())
    }
}

impl<T: crate::SaveSafeTensors> crate::SaveSafeTensors for Vec<T> {
    fn write_safetensors(
        &self,
        location: &str,
        tensors: &mut Vec<(String, safetensors::Dtype, Vec<usize>, Vec<u8>)>,
    ) {
        for (i, t) in self.iter().enumerate() {
            t.write_safetensors(&format!("{location}{i}."), tensors);
        }
    }
}

impl<T: crate::LoadSafeTensors> crate::LoadSafeTensors for Vec<T> {
    fn read_safetensors<'a>(
        &mut self,
        location: &str,
        tensors: &safetensors::SafeTensors<'a>,
    ) -> Result<(), safetensors::SafeTensorError> {
        for (i, t) in self.iter_mut().enumerate() {
            t.read_safetensors(&format!("{location}{i}."), tensors)?;
        }
        Ok(())
    }
}

impl<Input, T: crate::Module<Input, Output = Input>> crate::Module<Input> for Vec<T> {
    type Output = T::Output;
    type Error = T::Error;

    fn try_forward(&self, mut x: Input) -> Result<Self::Output, T::Error> {
        for m_i in self.iter() {
            x = m_i.try_forward(x)?;
        }
        Ok(x)
    }
    fn try_forward_mut(&mut self, mut x: Input) -> Result<Self::Output, Self::Error> {
        for m_i in self.iter_mut() {
            x = m_i.try_forward_mut(x)?;
        }
        Ok(x)
    }
}
