//! Demonstrates how to build a neural network with convolution
//! layers on nightly rust.
#![cfg_attr(feature = "nightly", feature(generic_const_exprs))]

#[cfg(feature = "nightly")]
fn main() {
    use dfdx::{prelude::*, tensor::AutoDevice};

    type Model = (
        (Conv2D<3, 4, 3>, ReLU),
        (Conv2D<4, 8, 3>, ReLU),
        (Conv2D<8, 16, 3>, ReLU),
        Flatten2D,
        Linear<7744, 10>,
    );

    let dev = AutoDevice::default();
    let m = dev.build_module::<Model, f32>();

    // single image forward
    let x: Tensor<Rank3<3, 28, 28>, f32, _> = dev.sample_normal();
    let _y: Tensor<Rank1<10>, f32, _> = m.forward(x);

    // batched image forward
    let x: Tensor<Rank4<32, 3, 28, 28>, f32, _> = dev.sample_normal();
    let _y: Tensor<Rank2<32, 10>, f32, _> = m.forward(x);
}

#[cfg(not(feature = "nightly"))]
fn main() {
    panic!("Run with `cargo +nightly run ...` to run this example.");
}
