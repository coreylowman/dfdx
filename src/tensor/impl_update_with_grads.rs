use crate::devices::Device;
use crate::gradients::{CanUpdateWithGradients, GradientProvider, UnusedTensors};
use crate::prelude::*;

impl<T: Tensor<Dtype = f32>> CanUpdateWithGradients for T {
    /// Subtracts the gradient for the tensor from [HasArrayData::mut_data].
    fn update<G: GradientProvider>(&mut self, grads: &mut G, unused: &mut UnusedTensors) {
        match grads.gradient(self) {
            Some(gradient) => <Self as HasDevice>::Device::sub(self.mut_data(), gradient.as_ref()),
            None => unused.add(self),
        }
    }
}
