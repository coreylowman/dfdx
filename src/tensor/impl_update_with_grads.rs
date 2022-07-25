use crate::prelude::*;

/// Subtracts the gradient for the tensor from [HasArrayData::mut_data].
impl<T: Tensor<Dtype = f32>> CanUpdateWithGradients for T {
    fn update<G: GradientProvider>(&mut self, grads: &mut G) -> MissingGradients {
        let mut missing: MissingGradients = Default::default();
        match grads.gradient(self) {
            Some(gradient) => <Self as HasDevice>::Device::sub(self.mut_data(), gradient.as_ref()),
            None => missing.add_unnamed(),
        }
        missing
    }
}
