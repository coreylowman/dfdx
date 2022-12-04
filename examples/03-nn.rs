//! Intro to dfdx::nn

use dfdx::{
    arrays::{Rank1, Rank2},
    nn::{BuildModule, Linear, Module, ModuleMut, ReLU},
    tensor::{AsArray, Cpu, RandnTensor, Tensor, ZerosTensor},
};

fn main() {
    let dev: Cpu = Default::default();

    // nn exposes many different neural network types, like the Linear layer!
    // you can use BuildModule::build to construct an initialized model
    let mut m: Linear<4, 2> = BuildModule::build(&dev);

    // BuildModule::reset_params also allows you to re-randomize the weights
    m.reset_params();

    // Modules act on tensors using either:
    // 1. `Module::forward`, which does not mutate the module
    let _: Tensor<Rank1<2>, f32, Cpu> = m.forward(dev.zeros::<Rank1<4>>());

    // 2. `ModuleMut::forward_mut()`, which may mutate the module
    let _: Tensor<Rank1<2>, f32, Cpu> = m.forward_mut(dev.zeros::<Rank1<4>>());

    // most of them can also act on many different shapes of tensors
    // here we see that Linear can also accept batches of inputs
    // Note: the Rank2 with a batch size of 10 in the input
    //       AND the output
    let _: Tensor<Rank2<10, 2>, f32, Cpu> = m.forward(dev.zeros::<Rank2<10, 4>>());

    // you can also combine multiple modules with tuples
    let mlp: (Linear<4, 2>, ReLU, Linear<2, 1>) = BuildModule::build(&dev);

    // and of course forward passes the input through each module sequentially:
    let x = dev.randn::<Rank1<4>>();
    let a: Tensor<Rank1<1>, f32, Cpu> = mlp.forward(x.clone());
    let b = mlp.2.forward(mlp.1.forward(mlp.0.forward(x)));
    assert_eq!(a.array(), b.array());
}
