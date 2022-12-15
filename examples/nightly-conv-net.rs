//! Demonstrates how to build a neural network with convolution
//! layers on nightly rust.
#![cfg_attr(feature = "nightly", feature(generic_const_exprs))]

#[cfg(feature = "nightly")]
fn main() {
    use dfdx::prelude::*;

    type Model = (
        (Conv2D<3, 4, 3>, ReLU),
        (Conv2D<4, 8, 3>, ReLU),
        (Conv2D<8, 16, 3>, ReLU),
        Flatten2D,
        Linear<7744, 10>,
    );

    let dev: Cpu = Default::default();
    let m: Model = dev.build_module();

    // single image forward
    let x: Tensor<Rank3<3, 28, 28>> = dev.randn();
    let _: Tensor<Rank1<10>> = m.forward(x);

    // batched image forward
    let x: Tensor<Rank4<32, 3, 28, 28>> = dev.randn();
    let _: Tensor<Rank2<32, 10>> = m.forward(x);
}

#[cfg(not(feature = "nightly"))]
fn main() {
    panic!("Run with `cargo +nightly run ...` to run this example.");
}
