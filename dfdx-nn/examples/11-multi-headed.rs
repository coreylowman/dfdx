//! Demonstrates how to build a neural network that has multiple
//! outputs using `SplitInto`.

use dfdx::{
    shapes::Rank1,
    tensor::{AsArray, AutoDevice, Tensor, TensorFrom},
};

use dfdx_nn::{BuildModuleExt, LinearConstConfig, Module, SplitInto};

fn main() {
    let dev = AutoDevice::default();

    // SplitInto accepts a tuple of modules. Each one of the items in the
    // tuple must accept the same type of input.
    // Note that here, both of the linears have the same size input (1)
    type Model = SplitInto<(LinearConstConfig<1, 3>, LinearConstConfig<1, 5>)>;
    let m = dev.build_module_ext::<f32>(Model::default());

    // when we forward data through, we get a tuple back!
    let (y1, y2): (Tensor<Rank1<3>, f32, _>, Tensor<Rank1<5>, f32, _>) =
        m.forward(dev.tensor([1.0]));
    println!("Split 1: {:?}, Split 2: {:?}", y1.array(), y2.array());
}