use dfdx::optim::Sgd;
use dfdx_macros::CanUpdateWithGradients;
use dfdx::tensor::*;



#[derive(Default, CanUpdateWithGradients)]
pub struct Linear<const I: usize, const O: usize> {
    // Transposed weight matrix, shape (O, I)
    pub weight: Tensor2D<O, I>,

    // Bias vector, shape (O, )
    pub bias: Tensor1D<O>,
}


#[test]
fn test_derive_can_update_with_gradients() {
    let mut model: Linear<5, 2> = Linear::default();
    let mut gradients_provider: Sgd<f32> = Default::default();
    let mut unused: UnusedTensors = Default::default();
    model.update(&mut gradients_provider, &mut unused);
}