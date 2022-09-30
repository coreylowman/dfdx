//! Intro to dfdx::nn

use rand::prelude::*;

use dfdx::nn::{Linear, Module, ReLU, ResetParams};
use dfdx::tensor::{Tensor1D, Tensor2D, TensorCreator};

fn main() {
    // nn exposes many different neural network types, like the Linear layer!
    let mut m: Linear<4, 2> = Default::default();

    // at first they are initialized to zeros, but you can randomize them too
    let mut rng = StdRng::seed_from_u64(0);
    m.reset_params(&mut rng);

    // they act on tensors using the forward method
    let x: Tensor1D<4> = TensorCreator::zeros();
    let _: Tensor1D<2> = m.forward(x);

    // most of them can also act on many different shapes of tensors
    let x: Tensor2D<10, 4> = TensorCreator::zeros();
    let _: Tensor2D<10, 2> = m.forward(x);

    // you can also combine multiple modules with tuples
    let mlp: (Linear<4, 2>, ReLU, Linear<2, 1>) = Default::default();

    let x: Tensor1D<4> = TensorCreator::zeros();
    let _: Tensor1D<1> = mlp.forward(x);
}
