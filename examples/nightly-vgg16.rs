//! This example implements VGG16, a deep convolutional network popular in facial recognition
#![cfg_attr(feature = "nightly", feature(generic_const_exprs))]

#[cfg(feature = "nightly")]
use dfdx::prelude::*;

#[cfg(feature = "nightly")]
type Vgg = (
    // conv block 1
    (
        (Conv2D<3, 64, 3, 1, 1>, ReLU),
        (Conv2D<64, 64, 3, 1, 1>, ReLU),
        MaxPool2D<2, 2>,
    ),
    // conv block 2
    (
        (Conv2D<64, 128, 3, 1, 1>, ReLU),
        (Conv2D<128, 128, 3, 1, 1>, ReLU),
        MaxPool2D<2, 2>,
    ),
    // conv block 3
    (
        (Conv2D<128, 256, 3, 1, 1>, ReLU),
        (Conv2D<256, 256, 3, 1, 1>, ReLU),
        (Conv2D<256, 256, 3, 1, 1>, ReLU),
        MaxPool2D<2, 2>,
    ),
    // conv block 4
    (
        (Conv2D<256, 512, 3, 1, 1>, ReLU),
        (Conv2D<512, 512, 3, 1, 1>, ReLU),
        (Conv2D<512, 512, 3, 1, 1>, ReLU),
        MaxPool2D<2, 2>,
    ),
    // conv block 5
    (
        (Conv2D<512, 512, 3, 1, 1>, ReLU),
        (Conv2D<512, 512, 3, 1, 1>, ReLU),
        (Conv2D<512, 512, 3, 1, 1>, ReLU),
        MaxPool2D<2, 2>,
    ),
    // head
    (
        (Conv2D<512, 4096, 7, 1, 0>, ReLU),
        (Conv2D<4096, 4096, 1, 1, 0>, ReLU),
        (Conv2D<4096, 2622, 1, 1, 0>, Flatten2D, Softmax),
    ),
);

#[cfg(feature = "nightly")]
fn main() {
    // the way we load the input would overflow the stack in windows
    // so we just make a new thread with larger stack, and run there
    let child = std::thread::Builder::new()
        .stack_size(2 * 1024 * 1024)
        .spawn(run)
        .unwrap();

    child.join().unwrap();
}

#[cfg(feature = "nightly")]
fn run() {
    // setup, takes about 5 seconds
    let dev: Cpu = Default::default();
    let vgg: Vgg = dev.build_module();

    // load weights, if you have them
    // vgg.load("vgg_weights.npz").unwrap()

    // this should be a 224x224 RGB image, but for demonstration let's just use zeros
    let image = [[[0.0f32; 224]; 224]; 3];
    let x: Tensor<Rank3<3, 224, 224>> = dev.tensor(image);

    // inference, takes about 1-2 seconds on a CPU
    let start = std::time::Instant::now();
    let y = vgg.forward(x);
    let result = y.array();
    println!(
        "inference ran in {:.2?}, first element of result is {:.2}",
        start.elapsed(),
        result[0]
    );
}

#[cfg(not(feature = "nightly"))]
fn main() {
    panic!("Run with `cargo +nightly run ...` to run this example.");
}
