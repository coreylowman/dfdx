use dfdx::prelude::{Device, Dtype, Shape, Tape, Tensor};

#[derive(Default, Debug, Clone, Copy, crate::CustomModule)]
pub struct ReLU;

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>> crate::Module<Tensor<S, E, D, T>> for ReLU {
    type Output = Tensor<S, E, D, T>;
    type Error = D::Err;
    fn try_forward(&self, x: Tensor<S, E, D, T>) -> Result<Self::Output, Self::Error> {
        x.try_relu()
    }
}
