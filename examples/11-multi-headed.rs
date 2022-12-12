//! Demonstrates how to build a neural network that has multiple
//! outputs using `SplitInto`.

use dfdx::{
    nn::{Linear, Module, ModuleBuilder, SplitInto},
    shapes::Rank1,
    tensor::{Cpu, Tensor, TensorFromArray},
};

fn main() {
    let dev: Cpu = Default::default();

    // SplitInto accepts a tuple of modules. Each one of the items in the
    // tuple must accept the same type of input.
    // Note that here, both of the linears have the same size input (1)
    let m: SplitInto<(Linear<1, 3>, Linear<1, 5>)> = dev.build_module();

    // when we forward data through, we get a tuple back!
    let _: (Tensor<Rank1<3>, f32>, Tensor<Rank1<5>, f32>) = m.forward(dev.tensor([1.0]));
}
