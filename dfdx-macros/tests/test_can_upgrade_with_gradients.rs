// Test macro expansion with `cargo expand --test test_can_upgrade_with_gradients`
use dfdx::optim::Sgd;
use dfdx::tensor::*;

use dfdx_macros::CanUpdateWithGradients;
use dfdx::gradients::{UnusedTensors, GradientProvider, CanUpdateWithGradients};
use dfdx::nn::Linear;

#[test]
fn test_named_fields() {

    #[derive(CanUpdateWithGradients, Default)]
    pub struct Linear<const I: usize, const O: usize> {
        // Transposed weight matrix, shape (O, I)
        pub weight: Tensor2D<O, I>,

        // Bias vector, shape (O, )
        pub bias: Tensor1D<O>,
    }
    let mut model: Linear<5, 2> = Linear::default();
    let mut gradients_provider: Sgd<f32> = Default::default();
    let mut unused: UnusedTensors = Default::default();
    model.update(&mut gradients_provider, &mut unused);
}

#[test]
fn test_unnamed_fields() {

    #[derive(CanUpdateWithGradients, Default)]
    pub struct Residual<F>(pub F);

    let mut model: Residual<Linear<2, 5>> = Default::default();
    let mut gradients_provider: Sgd<f32> = Default::default();
    let mut unused: UnusedTensors = Default::default();
    model.update(&mut gradients_provider, &mut unused);
}