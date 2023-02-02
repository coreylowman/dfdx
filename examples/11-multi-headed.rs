//! Demonstrates how to build a neural network that has multiple
//! outputs using `SplitInto`.

use dfdx::{
    nn::{BuildOnDevice, Linear, Module, SplitInto},
    shapes::Rank1,
    tensor::{Cpu, Tensor, TensorFrom},
};

fn main() {
    let dev: Cpu = Default::default();

    // SplitInto accepts a tuple of modules. Each one of the items in the
    // tuple must accept the same type of input.
    // Note that here, both of the linears have the same size input (1)
    type Model = SplitInto<(Linear<1, 3>, Linear<1, 5>)>;
    let m = Model::build_on_device(&dev);

    // when we forward data through, we get a tuple back!
    let _: (Tensor<Rank1<3>>, Tensor<Rank1<5>>) = m.forward(dev.tensor([1.0]));
}
