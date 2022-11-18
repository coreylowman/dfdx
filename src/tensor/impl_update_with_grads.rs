use crate::{
    arrays::{Dtype, Shape},
    devices::Device,
    gradients::{CanUpdateWithGradients, GradientProvider, UnusedTensors},
};

use super::Tensor;

impl<S: Shape, E: Dtype, D: Device, T> CanUpdateWithGradients for Tensor<S, E, D, T> {
    /// Subtracts the gradient for the tensor from [HasArrayData::mut_data].
    fn update<G: GradientProvider>(&mut self, grads: &mut G, unused: &mut UnusedTensors) {
        match grads.gradient::<D, Self>(self) {
            Some(grad) => self.device.sub_assign(&mut self.storage, &grad),
            None => unused.add(self),
        }
    }
}
