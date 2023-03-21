//! Demonstrates how to build a neural network that has multiple
//! outputs using `SplitInto`.

use dfdx::{
    nn::builders::{DeviceBuildExt, Linear, Module, SplitInto},
    shapes::Rank1,
    tensor::{AsArray, AutoDevice, Tensor, TensorFrom},
};

fn main() {
    let dev = AutoDevice::default();

    // SplitInto accepts a tuple of modules. Each one of the items in the
    // tuple must accept the same type of input.
    // Note that here, both of the linears have the same size input (1)
    type Model = SplitInto<(Linear<1, 3>, Linear<1, 5>)>;
    let m = dev.build_module::<Model, f32>();

    // when we forward data through, we get a tuple back!
    let (y1, y2): (Tensor<Rank1<3>, f32, _>, Tensor<Rank1<5>, f32, _>) =
        m.forward(dev.tensor([1.0]));
    println!("Split 1: {:?}, Split 2: {:?}", y1.array(), y2.array());
}
