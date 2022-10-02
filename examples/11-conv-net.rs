//! Demonstrates how to build a neural network with convolution
//! layers on nightly rust.

#![cfg_attr(feature = "nightly", feature(generic_const_exprs))]

#[cfg(not(feature = "nightly"))]
fn main() {
    panic!("`+nightly` required to run this example.")
}

#[cfg(feature = "nightly")]
fn main() {
    use dfdx::prelude::*;
    use rand::thread_rng;

    type Model = (
        (Conv2D<3, 4, 3>, ReLU),
        (Conv2D<4, 8, 3>, ReLU),
        (Conv2D<8, 16, 3>, ReLU),
        FlattenImage,
        Linear<7744, 10>,
    );

    let mut rng = thread_rng();
    let mut m: Model = Default::default();
    m.reset_params(&mut rng);

    // single image forward
    let x: Tensor3D<3, 28, 28> = TensorCreator::randn(&mut rng);
    let _: Tensor1D<10> = m.forward(x);

    // batched image forward
    let x: Tensor4D<32, 3, 28, 28> = TensorCreator::randn(&mut rng);
    let _: Tensor2D<32, 10> = m.forward(x);
}
