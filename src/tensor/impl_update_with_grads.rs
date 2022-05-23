use crate::prelude::*;

/// Subtracts the gradient for the tensor from [HasArrayData::mut_data].
impl<T: Tensor<Dtype = f32>> CanUpdateWithGradients for T {
    fn update<G: GradientProvider>(&mut self, grads: &mut G) {
        let gradient = grads.gradient(self).unwrap();
        <Self as HasDevice>::Device::sub_assign(self.mut_data(), gradient.as_ref());
    }
}
